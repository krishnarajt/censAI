import argparse
import logging

import censor as cn
import sub_manip as sm
import util
import video_manip as vm
from config.settings import Config
from db.store import Store
from enums.CensorshipStrength import CensorshipStrength
from enums.LoggingColors import LoggingColors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-folder", dest="media_folder")
    parser.add_argument("--strength", choices=["moderate", "strict"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
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

    util.print_welcome_message()
    media_folder_path = args.media_folder or input(
        "\nEnter the path to the media folder (will be searched recursively): "
    ).strip()
    config.media_folder_path = media_folder_path

    # Open SQLite store. Deleting the file at config.db_path clears all caches.
    Store().open(config.db_path)
    logging.info(
        f"{LoggingColors.BLUE_INFO}Using cache DB at: {config.db_path}{LoggingColors.RESET}"
    )
    logging.info(
        f"{LoggingColors.BLUE_INFO}Vision model: {config.VISION_MODEL}  |  Profanity model: {config.PROFANITY_MODEL}{LoggingColors.RESET}"
    )

    if args.strength:
        config.censorship_strength = (
            CensorshipStrength.MODERATE
            if args.strength == "moderate"
            else CensorshipStrength.STRICT
        )
    else:
        util.get_censorship_strength()

    logging.info(
        f"{LoggingColors.BLUE_INFO}Selected censorship strength: {config._censorship_strength}{LoggingColors.RESET}"
    )
    print()

    logging.info(
        f"{LoggingColors.BLUE_INFO}Looking for video and subtitle files{LoggingColors.RESET}"
    )
    videos = vm.find_videos(config.media_folder_path)
    vm.create_video_to_id_mappings(videos)
    subtitles = sm.find_subtitles(config.media_folder_path)
    config.video_and_subtitle_files = sm.match_video_and_subtitles(videos, subtitles)
    logging.info(
        f"{LoggingColors.BLUE_INFO}Found {len(videos)} videos and {len(subtitles)} subtitles, matched {len(config.video_and_subtitle_files)} of them.{LoggingColors.RESET}"
    )
    print()

    util.print_censorship_message()

    for video_id, subtitle_path in config.video_and_subtitle_files.items():
        subtitle_label = subtitle_path.name if subtitle_path is not None else "None"
        logging.info(
            f"{LoggingColors.BLUE_INFO}Processing video ID {video_id} with subtitle: {subtitle_label}{LoggingColors.RESET}"
        )
        cn.censor(video_id, subtitle_path)

    print("All done! Enjoy your media.")
