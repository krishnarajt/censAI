import json
import logging
import math
import pathlib
import shutil
import subprocess
import sys

import pandas as pd
from tqdm import tqdm

import sub_manip as sm
import util
from config.settings import Config
from db.store import Store

config = Config()
store = Store()


def find_videos(path):
    root_path = pathlib.Path(path).resolve()
    return sorted(
        (
            video
            for video in pathlib.Path(path).rglob("*")
            if video.suffix.lower() in config.VIDEO_EXTENSIONS
            and not video.name.startswith("censored_")
            and "temp"
            not in {part.lower() for part in video.resolve().relative_to(root_path).parts[:-1]}
        ),
        key=lambda p: str(p).lower(),
    )


def create_video_to_id_mappings(video_list):
    config.video_to_id = {}
    config.id_to_video = {}
    for idx, video_path in enumerate(video_list, start=1):
        config.video_to_id[video_path] = idx
        config.id_to_video[idx] = video_path
        store.upsert_video(idx, str(video_path), video_path.name)


def split_into_scenes(video_id):
    video_path = config.id_to_video[video_id]
    scene_csv_path = config._temp_folder_path / f"{video_id}/scenes.csv"
    detection_exists = scene_csv_path.exists()

    expected_images = 0
    should_not_split = False
    if detection_exists:
        scenes_df = pd.read_csv(scene_csv_path, skiprows=1)
        expected_images = len(scenes_df) * config.NUMBER_OF_IMAGES_PER_SCENE
        actual_images = len(list((config._temp_folder_path / f"{video_id}").glob("*.jpg")))
        should_not_split = math.isclose(expected_images, actual_images, rel_tol=0.01)

    if should_not_split:
        logging.info("Scenes already split for video %s, skipping split step.", video_id)
        index_scene_images(video_id)
        return

    logging.info(
        "Detection exists: %s, splitting required.", detection_exists
    )
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
        index_scene_images(video_id)
    except subprocess.CalledProcessError as exc:
        logging.error("Error splitting scenes for %s: %s", video_path.name, exc)
        raise


def index_scene_images(video_id):
    """Compute sha256 + phash for every snapshot frame and upsert into the DB.
    Re-runs are cheap because INSERT ... ON CONFLICT keeps existing hashes."""
    images = sorted((config._temp_folder_path / f"{video_id}").glob("*.jpg"))
    if not images:
        logging.warning(
            "No images found in %s. Skipping frame indexing.",
            config._temp_folder_path / str(video_id),
        )
        return

    for image in tqdm(images, desc=f"Hashing frames for {video_id}", unit="frame"):
        try:
            scene_number, image_number, timestamp = map(int, image.stem.split("-"))
        except ValueError:
            logging.warning("Skipping unrecognised frame filename: %s", image.name)
            continue
        try:
            sha = util.sha256_file(image)
            phash = util.phash_image(image)
        except Exception as exc:  # noqa: BLE001 -- frame may be partially written
            logging.warning("Failed to hash %s: %s", image.name, exc)
            sha = None
            phash = None
        store.upsert_frame(
            video_id=video_id,
            scene_number=scene_number,
            snapshot_number=image_number,
            timestamp_ms=timestamp,
            snapshot_path=str(image),
            image_sha256=sha,
            image_phash=phash,
        )

    assign_scene_numbers_to_subtitles(video_id)


def assign_scene_numbers_to_subtitles(video_id):
    scene_csv_path = config._temp_folder_path / f"{video_id}/scenes.csv"
    if not scene_csv_path.exists():
        logging.warning(
            "Scene CSV not found for video %s. Subtitle-to-scene mapping skipped.", video_id
        )
        return

    scene_ranges = pd.read_csv(scene_csv_path, skiprows=1)
    scene_ranges["start_ms"] = (scene_ranges["Start Time (seconds)"] * 1000).round().astype(int)
    scene_ranges["end_ms"] = (scene_ranges["End Time (seconds)"] * 1000).round().astype(int)

    subtitle_rows = store.get_subtitles(video_id)
    mappings = []
    for row in subtitle_rows:
        start_ms = int(row["start_ms"])
        matches = scene_ranges[
            (scene_ranges["start_ms"] <= start_ms) & (scene_ranges["end_ms"] > start_ms)
        ]
        if matches.empty:
            continue
        mappings.append((int(row["id"]), int(matches.iloc[0]["Scene Number"])))
    if mappings:
        store.assign_subtitle_scenes(video_id, mappings)


def _scene_csv_ranges(video_id):
    scene_csv_path = config._temp_folder_path / f"{video_id}/scenes.csv"
    if not scene_csv_path.exists():
        return None
    return pd.read_csv(scene_csv_path, skiprows=1)


def mute_audio(video_id):
    video_path = config.id_to_video[video_id]
    output_path = config.muted_audio_root_path / video_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    intervals = store.get_profane_subtitle_intervals(video_id)
    if not intervals:
        logging.info(
            "No profane audio timestamps found for video %s. Copying original audio/video.",
            video_id,
        )
        shutil.copy2(video_path, output_path)
        return

    interval_s = [(s / 1000.0, e / 1000.0) for s, e in intervals]
    volume_expr = "+".join([f"between(t,{s:.3f},{e:.3f})" for s, e in interval_s])
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
        logging.info(
            "Muted profane segments for %s. Output saved to %s.", video_path.name, output_path
        )
    except subprocess.CalledProcessError as exc:
        logging.error("Error muting audio for %s: %s", video_path.name, exc)
        raise


def remove_scenes_and_generate_final_video(video_id, subtitle_path):
    video_path = config.id_to_video[video_id]
    input_path = config.muted_audio_root_path / video_path.name
    source_for_copy = input_path if input_path.exists() else video_path
    output_path = config.media_folder_path / f"censored_{video_path.name}"
    subtitle_output_path = config.media_folder_path / f"censored_{video_path.stem}.srt"

    censored_scenes = store.get_censored_scene_numbers(video_id)
    if not censored_scenes:
        logging.info(
            "No scenes to censor for video %s. Copying video without visual changes.", video_id
        )
        shutil.copy2(source_for_copy, output_path)
        if subtitle_path is not None:
            sm.write_censored_subtitles(video_id, subtitle_path, subtitle_output_path)
        return

    scene_ranges = _scene_csv_ranges(video_id)
    remove_intervals = []
    if scene_ranges is not None:
        scene_ranges = scene_ranges[scene_ranges["Scene Number"].isin(censored_scenes)]
        for _, row in scene_ranges.iterrows():
            remove_intervals.append(
                (float(row["Start Time (seconds)"]), float(row["End Time (seconds)"]))
            )
    else:
        # Fallback: use frame timestamps. Less accurate, no end-of-scene gap.
        for scene_number in censored_scenes:
            frames = [
                row
                for row in store.get_scene_frames_with_caches(video_id, scene_number)
            ]
            if not frames:
                continue
            start_ms = min(int(row["timestamp_ms"]) for row in frames)
            end_ms = max(int(row["timestamp_ms"]) for row in frames)
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
        (s, e) for s, e in keep_intervals if e - s > 0.05
    ]

    if not keep_intervals:
        raise RuntimeError(
            "All video intervals were marked for censoring; refusing to create an empty output."
        )

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
    filter_parts.append("".join(concat_inputs) + f"concat=n={len(keep_intervals)}:v=1:a=1[v][a]")

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


def print_stats(video_id):
    rows = store.get_subtitles(video_id)
    censored_scenes = store.get_censored_scene_numbers(video_id)
    cleaned_subs = sum(1 for r in rows if r["cleaned_text"])
    profane_subs = sum(1 for r in rows if r["profanity_present"])

    # Use scenedetect's CSV for accurate scene boundaries.
    total_time_cut_s = 0.0
    scene_ranges = _scene_csv_ranges(video_id)
    if scene_ranges is not None and censored_scenes:
        filtered = scene_ranges[scene_ranges["Scene Number"].isin(censored_scenes)]
        total_time_cut_s = float(
            (filtered["End Time (seconds)"] - filtered["Start Time (seconds)"]).sum()
        )

    print(f"Total Time Cut:       {total_time_cut_s:.2f} seconds")
    print(f"Censored Scenes:      {len(censored_scenes)}")
    print(f"Subtitle Lines:       {len(rows)}")
    print(f"Profane Subtitles:    {profane_subs}")
    print(f"Cleaned Subtitles:    {cleaned_subs}")
