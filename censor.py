import logging
import time

import nudity as nd
import profanity as pf
import sub_manip as sm
import video_manip as vm
from config.settings import Config
from data.dataframe_manager import ScenesDataFrameManager
from tqdm import tqdm

config = Config()
df_manager = ScenesDataFrameManager()


def censor(video_id, subtitle_path):
    steps = [
        "1. Aligning Subtitles",
        "2. Cleaning Subtitle File",
        "3. Splitting Video into Scenes",
        "4. Running NudeNet across saved scene frames",
        "5. Classifying representative frames with a local vision model",
        "6. Aggregating scene evidence into final censor decisions",
        "7. Muting audio",
        "8. Removing Scenes, Generating final video",
        "Final Stats",
    ]
    current_step = iter(range(len(steps)))
    video_path = config.id_to_video[video_id]
    logging.info("Starting the censorship process for : %s", video_path)

    total_start = time.perf_counter()

    with tqdm(total=len(steps), desc="Processing", unit="step") as pbar:
        step_start = time.perf_counter()
        tqdm.write(steps[next(current_step)])
        sm.align_subtitles(video_id, subtitle_path)
        tqdm.write(f"-> Done in {time.perf_counter() - step_start:.2f} sec")
        pbar.update(1)
        config.save_checkpoint()

        step_start = time.perf_counter()
        tqdm.write(steps[next(current_step)])
        current_subtitle_path = config.video_and_subtitle_files.get(video_id, subtitle_path)
        sm.clean_subtitles(video_id, current_subtitle_path)
        tqdm.write(f"-> Done in {time.perf_counter() - step_start:.2f} sec")
        pbar.update(1)
        config.save_checkpoint()

        step_start = time.perf_counter()
        tqdm.write(steps[next(current_step)])
        vm.split_into_scenes(video_id)
        tqdm.write(f"-> Done in {time.perf_counter() - step_start:.2f} sec")
        pbar.update(1)
        config.save_checkpoint()

        step_start = time.perf_counter()
        tqdm.write(steps[next(current_step)])
        nd.detect_nudity_in_video(video_id)
        tqdm.write(f"-> Done in {time.perf_counter() - step_start:.2f} sec")
        pbar.update(1)
        config.save_checkpoint()

        step_start = time.perf_counter()
        tqdm.write(steps[next(current_step)])
        nd.generate_descriptions_for_nude_scenes(video_id)
        tqdm.write(f"-> Done in {time.perf_counter() - step_start:.2f} sec")
        pbar.update(1)
        config.save_checkpoint()

        step_start = time.perf_counter()
        tqdm.write(steps[next(current_step)])
        pf.determine_if_should_censor_video(video_id)
        tqdm.write(f"-> Done in {time.perf_counter() - step_start:.2f} sec")
        pbar.update(1)
        config.save_checkpoint()

        step_start = time.perf_counter()
        tqdm.write(steps[next(current_step)])
        vm.mute_audio(video_id)
        tqdm.write(f"-> Done in {time.perf_counter() - step_start:.2f} sec")
        pbar.update(1)

        step_start = time.perf_counter()
        tqdm.write(steps[next(current_step)])
        vm.remove_scenes_and_generate_final_video(video_id, current_subtitle_path)
        tqdm.write(f"-> Done in {time.perf_counter() - step_start:.2f} sec")
        pbar.update(1)

        step_start = time.perf_counter()
        tqdm.write(steps[next(current_step)])
        df_manager.print_stats(video_id)
        tqdm.write(f"-> Done in {time.perf_counter() - step_start:.2f} sec")

    total_duration = time.perf_counter() - total_start
    tqdm.write(f"\nAll steps completed in {total_duration:.2f} seconds.")

    logging.info("Censorship process completed for : %s", video_path)
    logging.info("\n\n")
