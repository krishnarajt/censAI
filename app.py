import argparse
import logging
from pathlib import Path

import censor as cn
import sub_manip as sm
import util
import video_manip as vm
from config.settings import Config
from db.central import CentralStore, is_rate_limit_error
from db.store import Store
from enums.CensorshipStrength import CensorshipStrength
from enums.LoggingColors import LoggingColors

BLUE_INFO = LoggingColors.BLUE_INFO.value
RESET = LoggingColors.RESET.value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-folder", dest="media_folder")
    parser.add_argument("--strength", choices=["moderate", "strict"])
    return parser.parse_args()


def configure_logging():
    config = Config()
    logging.basicConfig(
        format=config.LOGGING_FORMAT,
        datefmt=config.LOGGING_DATE_FORMAT,
        level=logging.INFO,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)
    logging.getLogger("safetext").setLevel(logging.ERROR)


def _set_strength(strength: str | None, interactive: bool = False):
    config = Config()
    if strength:
        config.censorship_strength = (
            CensorshipStrength.MODERATE
            if strength.lower() == "moderate"
            else CensorshipStrength.STRICT
        )
        return

    if interactive:
        util.get_censorship_strength()
        return

    default_strength = config.DEFAULT_CENSORSHIP_STRENGTH
    config.censorship_strength = (
        CensorshipStrength.STRICT
        if default_strength == "strict"
        else CensorshipStrength.MODERATE
    )


def _central_store() -> CentralStore:
    central = CentralStore(Config())
    central.init_db()
    return central


def scan_media_folder(media_folder_path: str, strength: str | None = None) -> dict:
    """Open the local SQLite store, scan media/subtitles, and sync main DB."""
    config = Config()
    config.media_folder_path = media_folder_path
    Store().open(config.db_path)

    central = _central_store()
    if strength is None and central.enabled:
        strength = central.get_config_value(
            "censorship_strength",
            config.DEFAULT_CENSORSHIP_STRENGTH,
        )
    _set_strength(strength, interactive=False)

    logging.info(
        f"{BLUE_INFO}Using cache DB at: {config.db_path}{RESET}"
    )
    logging.info(
        "%sLLM provider: %s  |  Vision model: %s  |  Profanity model: %s%s",
        BLUE_INFO,
        config.llm_provider_label,
        config.vision_model,
        config.profanity_model,
        RESET,
    )

    logging.info(
        f"{BLUE_INFO}Looking for video and subtitle files{RESET}"
    )
    videos = vm.find_videos(config.media_folder_path)
    vm.create_video_to_id_mappings(videos)
    subtitles = sm.find_subtitles(config.media_folder_path)
    config.video_and_subtitle_files = sm.match_video_and_subtitles(videos, subtitles)

    for video_id, subtitle_path in config.video_and_subtitle_files.items():
        central.upsert_detected_video(
            path=config.id_to_video[video_id],
            media_folder=config.media_folder_path,
            local_video_id=video_id,
            subtitle_path=subtitle_path,
        )

    logging.info(
        f"{BLUE_INFO}Found {len(videos)} videos and {len(subtitles)} subtitles, matched {len(config.video_and_subtitle_files)} of them.{RESET}"
    )
    return {
        "videos": len(videos),
        "subtitles": len(subtitles),
        "matched": len(config.video_and_subtitle_files),
        "media_folder": str(config.media_folder_path),
        "central_db_enabled": central.enabled,
        "database_backend": central.backend_label,
    }


def scan_media_folders(media_folder_paths: list[str], strength: str | None = None) -> dict:
    """Scan one or more media roots and return an aggregate summary."""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_path in media_folder_paths:
        cleaned = str(raw_path).strip()
        if not cleaned:
            continue
        key = str(Path(cleaned).expanduser().resolve())
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)

    total_videos = 0
    total_subtitles = 0
    total_matched = 0
    scans: list[dict] = []
    for media_folder_path in normalized:
        result = scan_media_folder(media_folder_path, strength=strength)
        total_videos += int(result["videos"])
        total_subtitles += int(result["subtitles"])
        total_matched += int(result["matched"])
        scans.append(result)

    config = Config()
    central = _central_store()
    return {
        "media_folders": [str(Path(path).expanduser().resolve()) for path in normalized],
        "scans": scans,
        "videos": total_videos,
        "subtitles": total_subtitles,
        "matched": total_matched,
        "central_db_enabled": central.enabled,
        "database_backend": central.backend_label,
        "llm_provider": config.llm_provider_label,
        "vision_model": config.vision_model,
        "profanity_model": config.profanity_model,
    }


def process_video(video_id: int, subtitle_path) -> bool:
    config = Config()
    central = CentralStore(config)
    video_path = config.id_to_video[video_id]
    logging.info("Starting queued processing for %s.", video_path.name)
    central.mark_started(video_path)
    try:
        cn.censor(video_id, subtitle_path)
    except Exception as exc:
        central.mark_failed(video_path, exc)
        if is_rate_limit_error(exc):
            logging.error(
                "Rate limited while processing %s. It will be retried after %.1f hour(s).",
                video_path.name,
                config.RETRY_DELAY_HOURS,
            )
            return False
        raise
    central.mark_completed(video_path)
    logging.info("Completed queued processing for %s.", video_path.name)
    return True


def process_video_path(
    video_path: str,
    media_folder_path: str | None = None,
    strength: str | None = None,
) -> bool:
    target = Path(video_path).expanduser().resolve()
    media_root = media_folder_path or str(target.parent)
    scan_media_folder(media_root, strength=strength)
    config = Config()

    for video_id, path in config.id_to_video.items():
        if Path(path).expanduser().resolve() == target:
            return process_video(video_id, config.video_and_subtitle_files.get(video_id))
    raise FileNotFoundError(f"Video is not present under media folder: {target}")


def process_all(media_folder_path: str, strength: str | None, interactive_strength: bool) -> int:
    config = Config()
    if interactive_strength:
        config.media_folder_path = media_folder_path
        _set_strength(strength, interactive=True)
        strength = (
            "strict"
            if config.censorship_strength == CensorshipStrength.STRICT
            else "moderate"
        )

    scan_media_folder(media_folder_path, strength=strength)

    logging.info(
        f"{BLUE_INFO}Selected censorship strength: {Config()._censorship_strength}{RESET}"
    )
    print()
    util.print_censorship_message()

    processed = 0
    for video_id, subtitle_path in Config().video_and_subtitle_files.items():
        subtitle_label = subtitle_path.name if subtitle_path is not None else "None"
        logging.info(
            f"{BLUE_INFO}Processing video ID {video_id} with subtitle: {subtitle_label}{RESET}"
        )
        if process_video(video_id, subtitle_path):
            processed += 1
    return processed


if __name__ == "__main__":
    args = parse_args()
    configure_logging()
    config = Config()
    util.print_welcome_message()
    media_folder_path = args.media_folder or input(
        "\nEnter the path to the media folder (will be searched recursively): "
    ).strip()
    process_all(media_folder_path, args.strength, interactive_strength=args.strength is None)
    print("All done! Enjoy your media.")
