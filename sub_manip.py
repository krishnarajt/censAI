import logging
import pathlib
import re
import subprocess
from pathlib import Path

import profanity as pf
import pysrt
from config.settings import Config
from data.dataframe_manager import ScenesDataFrameManager
from enums.SceneCols import SceneCols
from rapidfuzz import fuzz, process
from safetext import SafeText
from tqdm import tqdm

config = Config()
df_manager = ScenesDataFrameManager()


def custom_scorer(choice, query, **kwargs):
    base_score = fuzz.ratio(choice, query)
    query_episode = re.search(r"S\d{2}E\d{2}", query)
    choice_episode = re.search(r"S\d{2}E\d{2}", choice)
    if query_episode and choice_episode and query_episode.group() == choice_episode.group():
        return 100
    return base_score


def find_subtitles(path):
    return sorted(
        (sub for sub in pathlib.Path(path).rglob("*") if sub.suffix.lower() in config.SUBTITLE_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )


def match_video_and_subtitles(videos, subtitles):
    if not videos:
        logging.warning("No videos found.")
        return {}
    if not subtitles:
        logging.warning("No subtitles found. Proceeding with video-only censorship.")
        return {config.video_to_id[video]: None for video in videos}

    subtitle_names = [sub.name for sub in subtitles]
    media_files = {}
    for video in videos:
        best_match, _score, _ = process.extractOne(video.name, subtitle_names, scorer=custom_scorer)
        subtitle = next((s for s in subtitles if s.name == best_match), None)
        media_files[config.video_to_id[video]] = subtitle
        subtitle_names.remove(best_match)

    if len(media_files.keys()) != len(videos):
        logging.warning("Some videos do not have matching subtitles.")
    else:
        logging.info("All videos have matching subtitles.")
    return media_files


def align_subtitles(video_id, subtitle_path):
    video_path = config.id_to_video[video_id]
    if subtitle_path is None:
        logging.warning("No matching subtitle for video: %s", video_path.name)
        return

    subtitle_synced_path = subtitle_path.with_stem(subtitle_path.stem + ".synced")
    if subtitle_synced_path.exists():
        logging.info("Synced subtitle already exists: %s", subtitle_synced_path.name)
        config.video_and_subtitle_files[video_id] = subtitle_synced_path
        return

    command = [
        "ffsubsync",
        str(video_path),
        "-i",
        str(subtitle_path),
        "--vad",
        "webrtc",
        "-o",
        str(subtitle_synced_path),
    ]

    try:
        subprocess.run(command, check=True)
        config.video_and_subtitle_files[video_id] = subtitle_synced_path
    except subprocess.CalledProcessError as exc:
        logging.error("Error aligning subtitles for %s: %s", video_path.name, exc)
        raise


def clean_subtitles(video_id, subtitle_path):
    if subtitle_path is None:
        logging.info("No subtitle file for video %s. Skipping subtitle cleanup.", video_id)
        if video_id not in config.subtitles_processed_video_ids:
            config.subtitles_processed_video_ids.append(video_id)
        return

    existing_rows = df_manager.all_scenes_df[
        (df_manager.all_scenes_df["video_id"] == video_id)
        & (df_manager.all_scenes_df["subtitle"].notna())
    ]
    if video_id in config.subtitles_processed_video_ids or not existing_rows.empty:
        logging.info("Skipping video %s subtitle processing as they have already been processed.", video_id)
        if video_id not in config.subtitles_processed_video_ids:
            config.subtitles_processed_video_ids.append(video_id)
        return

    st = SafeText(language="en")
    subs = pysrt.open(str(subtitle_path))

    try:
        for sub in tqdm(subs, desc="Cleaning subtitles", unit="sub"):
            profane = st.check_profanity(sub.text)
            df_manager.all_scenes_df.loc[len(df_manager.all_scenes_df)] = {
                str(SceneCols.TIMESTAMP): sub.start.ordinal,
                str(SceneCols.SUBTITLE_START_TIME): sub.start.ordinal,
                str(SceneCols.SUBTITLE_END_TIME): sub.end.ordinal,
                str(SceneCols.VIDEO_ID): video_id,
                str(SceneCols.SCENE_NUMBER): None,
                str(SceneCols.SCENE_SNAPSHOT_NUMBER): None,
                str(SceneCols.SCENE_SNAPSHOT_PATH): None,
                str(SceneCols.SUBTITLE): sub.text,
                str(SceneCols.CLEANED_SUBTITLE): pf.clean_text(sub.text) if profane else None,
                str(SceneCols.SNAPSHOT_DESC): None,
                str(SceneCols.DETECTOR_LABELS): None,
                str(SceneCols.DETECTOR_MAX_SCORE): None,
                str(SceneCols.DETECTOR_RAW): None,
                str(SceneCols.FRAME_ROLE): None,
                str(SceneCols.VISIBLE_NUDITY): None,
                str(SceneCols.EXPLICIT_EXPOSURE): None,
                str(SceneCols.SEXUAL_ACTIVITY): None,
                str(SceneCols.VISION_CONFIDENCE): None,
                str(SceneCols.VISION_REASON): None,
                str(SceneCols.VISION_RAW): None,
                str(SceneCols.PROFANITY_PRESENT): bool(profane),
                str(SceneCols.NUDITY_PRESENT): None,
                str(SceneCols.SHOULD_CENSOR): None,
            }
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Saving checkpoint before exiting...")
        config.save_checkpoint()
        raise

    config.subtitles_processed_video_ids.append(video_id)
    config.save_checkpoint()


def write_censored_subtitles(video_id, source_subtitle_path, output_subtitle_path):
    if source_subtitle_path is None:
        logging.info("No source subtitle file for video %s. Skipping subtitle output.", video_id)
        return

    subs = pysrt.open(str(source_subtitle_path))
    subtitle_rows = df_manager.all_scenes_df[
        (df_manager.all_scenes_df["video_id"] == video_id)
        & (df_manager.all_scenes_df["subtitle"].notna())
    ].copy()

    if subtitle_rows.empty:
        logging.warning("No subtitle rows found for video %s. Skipping subtitle output.", video_id)
        return

    rows_by_start = {
        int(row["subtitle_start_time"]): row
        for _, row in subtitle_rows.iterrows()
    }

    for sub in subs:
        row = rows_by_start.get(sub.start.ordinal)
        if row is None:
            continue
        if row.get("should_censor") is True:
            sub.text = "[scene censored]"
        elif isinstance(row.get("cleaned_subtitle"), str) and row["cleaned_subtitle"].strip():
            sub.text = row["cleaned_subtitle"]

    Path(output_subtitle_path).parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(output_subtitle_path), encoding="utf-8")
