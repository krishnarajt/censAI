import logging
import pathlib
import re
import subprocess
from pathlib import Path

import profanity as pf
import pysrt
from rapidfuzz import fuzz, process
from safetext import SafeText
from tqdm import tqdm

import util
from config.settings import Config
from db.store import Store

config = Config()
store = Store()


def custom_scorer(choice, query, **kwargs):
    base_score = fuzz.ratio(choice, query)
    query_episode = re.search(r"S\d{2}E\d{2}", query)
    choice_episode = re.search(r"S\d{2}E\d{2}", choice)
    if query_episode and choice_episode and query_episode.group() == choice_episode.group():
        return 100
    return base_score


def find_subtitles(path):
    return sorted(
        (
            sub
            for sub in pathlib.Path(path).rglob("*")
            if sub.suffix.lower() in config.SUBTITLE_EXTENSIONS
        ),
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
    unmatched_videos = []
    for video in videos:
        if not subtitle_names:
            media_files[config.video_to_id[video]] = None
            unmatched_videos.append(video.name)
            continue

        match = process.extractOne(video.name, subtitle_names, scorer=custom_scorer)
        if match is None:
            media_files[config.video_to_id[video]] = None
            unmatched_videos.append(video.name)
            continue

        best_match, _score, _ = match
        subtitle = next((s for s in subtitles if s.name == best_match), None)
        media_files[config.video_to_id[video]] = subtitle
        if subtitle is None:
            unmatched_videos.append(video.name)
            continue
        subtitle_names.remove(best_match)

    if unmatched_videos:
        logging.warning(
            "Some videos do not have matching subtitles: %s",
            ", ".join(unmatched_videos[:10]),
        )
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
        return

    if store.has_subtitles(video_id):
        logging.info(
            "Subtitles for video %s already loaded into the DB. Skipping re-parse.",
            video_id,
        )
        return

    st = SafeText(language="en")
    subs = pysrt.open(str(subtitle_path))

    try:
        rows = []
        profane_items = []
        for sub in tqdm(subs, desc="Scanning subtitles", unit="sub"):
            text = (sub.text or "").strip()
            if not text:
                continue
            text_hash = util.sha256_text(text)
            profane = bool(st.check_profanity(text))
            if profane:
                profane_items.append((text_hash, text))
            rows.append(
                {
                    "start_ms": sub.start.ordinal,
                    "end_ms": sub.end.ordinal,
                    "text": text,
                    "text_hash": text_hash,
                    "profanity_present": profane,
                }
            )

        cleaned_by_hash = pf.clean_texts_cached(profane_items) if profane_items else {}
        for row in tqdm(rows, desc="Persisting subtitles", unit="sub"):
            store.insert_subtitle(
                video_id=video_id,
                start_ms=row["start_ms"],
                end_ms=row["end_ms"],
                text=row["text"],
                text_hash=row["text_hash"],
                cleaned_text=cleaned_by_hash.get(row["text_hash"]),
                profanity_present=row["profanity_present"],
            )
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Subtitle progress is already persisted.")
        raise


def write_censored_subtitles(video_id, source_subtitle_path, output_subtitle_path):
    if source_subtitle_path is None:
        logging.info("No source subtitle file for video %s. Skipping subtitle output.", video_id)
        return

    subs = pysrt.open(str(source_subtitle_path))
    rows = store.get_subtitles(video_id)
    if not rows:
        logging.warning("No subtitle rows found for video %s. Skipping subtitle output.", video_id)
        return

    rows_by_start = {int(r["start_ms"]): r for r in rows}
    censored_scenes = set(store.get_censored_scene_numbers(video_id))

    for sub in subs:
        row = rows_by_start.get(sub.start.ordinal)
        if row is None:
            continue
        scene_number = row["scene_number"]
        if scene_number is not None and int(scene_number) in censored_scenes:
            sub.text = "[scene censored]"
            continue
        cleaned = row["cleaned_text"]
        if isinstance(cleaned, str) and cleaned.strip():
            sub.text = cleaned

    Path(output_subtitle_path).parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(output_subtitle_path), encoding="utf-8")
