import logging
import math
import pathlib
import shutil
import subprocess
import sys
import json

import pandas as pd
import sub_manip as sm
from config.settings import Config
from data.dataframe_manager import ScenesDataFrameManager
from enums.SceneCols import SceneCols

config = Config()
df_manager = ScenesDataFrameManager()


def find_videos(path):
    root_path = pathlib.Path(path).resolve()
    return sorted(
        (
            video
            for video in pathlib.Path(path).rglob("*")
            if video.suffix.lower() in config.VIDEO_EXTENSIONS
            and not video.name.startswith("censored_")
            and "temp" not in {part.lower() for part in video.resolve().relative_to(root_path).parts[:-1]}
        ),
        key=lambda p: str(p).lower(),
    )


def create_video_to_id_mappings(video_list):
    config.video_to_id = {}
    config.id_to_video = {}
    for idx, video_path in enumerate(video_list, start=1):
        config.video_to_id[video_path] = idx
        config.id_to_video[idx] = video_path


def split_into_scenes(video_id):
    current_scenes_df = pd.DataFrame()
    should_not_split = False
    detection_exists = (config._temp_folder_path / f"{video_id}/scenes.csv").exists()
    if detection_exists:
        current_scenes_df = pd.read_csv(config._temp_folder_path / f"{video_id}/scenes.csv", skiprows=1)
        should_not_split = math.isclose(
            len(current_scenes_df) * config.NUMBER_OF_IMAGES_PER_SCENE,
            len(list((config._temp_folder_path / f"{video_id}").glob("*.jpg"))),
            rel_tol=0.01,
        )

    if should_not_split:
        logging.info("Scenes already split for video %s, skipping...", video_id)
        add_image_paths_and_scene_subs(video_id)
        return

    logging.info("Detection exists: %s, should split: %s", detection_exists, should_not_split)
    video_path = config.id_to_video[video_id]
    logging.info("Splitting video %s into scenes...", video_path.name)

    command = [
        sys.executable,
        "-m",
        "scenedetect",
        "--verbosity",
        "error",
        "--input",
        str(video_path),
        "list-scenes",
        "-f",
        f"{config._temp_folder_path}/{video_id}/scenes",
        "save-images",
        "--num-images",
        f"{config.NUMBER_OF_IMAGES_PER_SCENE}",
        "--filename",
        "$SCENE_NUMBER-$IMAGE_NUMBER-$TIMESTAMP_MS",
        "--output",
        f"{config._temp_folder_path}/{video_id}",
    ]

    if detection_exists:
        command.extend(["load-scenes", "-i", f"{config._temp_folder_path}/{video_id}/scenes.csv"])

    try:
        subprocess.run(command, check=True)
        add_image_paths_and_scene_subs(video_id)
    except subprocess.CalledProcessError as exc:
        logging.error("Error splitting scenes for %s: %s", video_path.name, exc)
        raise


def add_image_paths_and_scene_subs(video_id):
    images = list((config._temp_folder_path / f"{video_id}").glob("*.jpg"))
    if not images:
        logging.warning("No images found in %s. Skipping scene update.", config._temp_folder_path / str(video_id))
        return

    df_manager.init_scene_images_df()
    for image in images:
        scene_number, image_number, timestamp = map(int, image.stem.split("-"))
        df_manager.scene_images_df.loc[len(df_manager.scene_images_df)] = [
            scene_number,
            video_id,
            image_number,
            timestamp,
            str(image),
        ]

    video_mask = ~(
        (df_manager.all_scenes_df["video_id"] == video_id)
        & (df_manager.all_scenes_df["scene_snapshot_path"].notna())
    )
    df_manager.all_scenes_df = df_manager.all_scenes_df[video_mask]
    df_manager.all_scenes_df = pd.concat([df_manager.all_scenes_df, df_manager.scene_images_df], ignore_index=True)
    df_manager.all_scenes_df.sort_values(by=["video_id", "timestamp"], inplace=True, ignore_index=True)
    df_manager.all_scenes_df.reset_index(drop=True, inplace=True)
    df_manager.all_scenes_df.drop_duplicates(subset=["timestamp", "video_id"], inplace=True)

    assign_scene_numbers(video_id)

    df_manager.all_scenes_df.to_pickle(config._temp_folder_path / f"{video_id}/all_scenes_df.pkl")
    df_manager.scene_images_df.to_pickle(config._temp_folder_path / f"{video_id}/scene_images_db.pkl")
    config.save_checkpoint()


def assign_scene_numbers(video_id):
    scene_csv_path = config._temp_folder_path / f"{video_id}/scenes.csv"
    if not scene_csv_path.exists():
        logging.warning("Scene CSV not found for video %s. Subtitle-to-scene mapping skipped.", video_id)
        return

    scene_ranges = pd.read_csv(scene_csv_path, skiprows=1)
    scene_ranges["start_ms"] = (scene_ranges["Start Time (seconds)"] * 1000).round().astype(int)
    scene_ranges["end_ms"] = (scene_ranges["End Time (seconds)"] * 1000).round().astype(int)

    subtitle_mask = (
        (df_manager.all_scenes_df["video_id"] == video_id)
        & (df_manager.all_scenes_df["subtitle"].notna())
    )
    for idx, row in df_manager.all_scenes_df[subtitle_mask].iterrows():
        subtitle_start = int(row["subtitle_start_time"])
        matching_scenes = scene_ranges[
            (scene_ranges["start_ms"] <= subtitle_start)
            & (scene_ranges["end_ms"] > subtitle_start)
        ]
        if matching_scenes.empty:
            continue
        df_manager.all_scenes_df.at[idx, "scene_number"] = int(matching_scenes.iloc[0]["Scene Number"])


def mute_audio(video_id):
    video_path = config.id_to_video[video_id]
    output_path = config.muted_audio_root_path / video_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profane_audio_timestamps = df_manager.all_scenes_df[
        (df_manager.all_scenes_df["profanity_present"] == True)
        & (df_manager.all_scenes_df["video_id"] == video_id)
    ]

    if profane_audio_timestamps.empty:
        logging.info("No profane audio timestamps found for video %s. Copying original audio/video.", video_id)
        shutil.copy2(video_path, output_path)
        return

    intervals = list(
        zip(
            profane_audio_timestamps[SceneCols.SUBTITLE_START_TIME.value[0]] / 1000.0,
            profane_audio_timestamps[SceneCols.SUBTITLE_END_TIME.value[0]] / 1000.0,
        )
    )

    volume_expr = "+".join([f"between(t,{start},{end})" for start, end in intervals])
    volume_filter = f"volume=enable='{volume_expr}':volume=0"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-af",
        volume_filter,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True)
        logging.info("Muted profane segments for %s. Output saved to %s.", video_path.name, output_path)
    except subprocess.CalledProcessError as exc:
        logging.error("Error muting audio for %s: %s", video_path.name, exc)
        raise


def remove_scenes_and_generate_final_video(video_id, subtitle_path):
    video_path = config.id_to_video[video_id]
    input_path = config.muted_audio_root_path / video_path.name
    source_for_copy = input_path if input_path.exists() else video_path
    output_path = config.media_folder_path / f"censored_{video_path.name}"
    subtitle_output_path = config.media_folder_path / f"censored_{video_path.stem}.srt"

    scenes_to_censor = df_manager.all_scenes_df[
        (df_manager.all_scenes_df["should_censor"] == True)
        & (df_manager.all_scenes_df["video_id"] == video_id)
    ][["scene_number"]].drop_duplicates()

    if scenes_to_censor.empty:
        logging.info("No scenes to censor for video %s. Copying video without visual changes.", video_id)
        shutil.copy2(source_for_copy, output_path)
        if subtitle_path is not None:
            sm.write_censored_subtitles(video_id, subtitle_path, subtitle_output_path)
        return

    scene_csv_path = config._temp_folder_path / f"{video_id}/scenes.csv"
    if scene_csv_path.exists():
        scene_ranges = pd.read_csv(scene_csv_path, skiprows=1)
        scene_ranges = scene_ranges[scene_ranges["Scene Number"].isin(scenes_to_censor["scene_number"])]
        remove_intervals = [
            (float(row["Start Time (seconds)"]), float(row["End Time (seconds)"]))
            for _, row in scene_ranges.iterrows()
        ]
    else:
        remove_intervals = []
        censored_rows = df_manager.all_scenes_df[
            (df_manager.all_scenes_df["should_censor"] == True)
            & (df_manager.all_scenes_df["video_id"] == video_id)
            & (df_manager.all_scenes_df["scene_number"].notna())
        ]
        for _scene_number, group in censored_rows.groupby("scene_number"):
            start_ms = float(group["timestamp"].min())
            end_ms = float(group["timestamp"].max())
            remove_intervals.append((start_ms / 1000.0, end_ms / 1000.0))
    remove_intervals.sort(key=lambda interval: interval[0])

    merged_intervals = []
    for start_time, end_time in remove_intervals:
        if not merged_intervals or start_time > merged_intervals[-1][1]:
            merged_intervals.append([start_time, end_time])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], end_time)

    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(source_for_copy),
    ]
    probe_result = subprocess.run(probe_command, check=True, capture_output=True, text=True)
    duration = float(json.loads(probe_result.stdout)["format"]["duration"])

    keep_intervals = []
    cursor = 0.0
    for start_time, end_time in merged_intervals:
        if start_time > cursor:
            keep_intervals.append((cursor, start_time))
        cursor = max(cursor, end_time)
    if cursor < duration:
        keep_intervals.append((cursor, duration))

    keep_intervals = [
        (start_time, end_time)
        for start_time, end_time in keep_intervals
        if end_time - start_time > 0.05
    ]

    if not keep_intervals:
        raise RuntimeError("All video intervals were marked for censoring; refusing to create an empty output.")

    filter_parts = []
    concat_inputs = []
    for index, (start_time, end_time) in enumerate(keep_intervals):
        filter_parts.append(
            f"[0:v]trim=start={start_time:.3f}:end={end_time:.3f},setpts=PTS-STARTPTS[v{index}]"
        )
        filter_parts.append(
            f"[0:a]atrim=start={start_time:.3f}:end={end_time:.3f},asetpts=PTS-STARTPTS[a{index}]"
        )
        concat_inputs.append(f"[v{index}][a{index}]")
    filter_parts.append(
        "".join(concat_inputs) + f"concat=n={len(keep_intervals)}:v=1:a=1[v][a]"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_for_copy),
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True)
        if subtitle_path is not None:
            sm.write_censored_subtitles(video_id, subtitle_path, subtitle_output_path)
        logging.info("Final video generated at %s.", output_path)
    except subprocess.CalledProcessError as exc:
        logging.error("Error generating final video for %s: %s", video_path.name, exc)
        raise
