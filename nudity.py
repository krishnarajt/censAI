import logging
import os
import re

from config.settings import Config
from data.dataframe_manager import ScenesDataFrameManager
from nudenet import NudeDetector
from ollama import Client
from tqdm import tqdm

config = Config()
df_manager = ScenesDataFrameManager()
detector = NudeDetector(model_path="models/640m.onnx", inference_resolution=640)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if OLLAMA_HOST == "0.0.0.0" or OLLAMA_HOST.startswith("0.0.0.0"):
    OLLAMA_HOST = "http://localhost:11434"
elif not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"
ollama_client = Client(host=OLLAMA_HOST)


def detect_nudity(image_path):
    results = detector.detect(image_path)
    results = [
        result
        for result in results
        if "face" not in result["class"].lower() and "feet" not in result["class"].lower()
    ]
    return len(results) > 0


def detect_nudity_in_video(video_id):
    rows = df_manager.all_scenes_df[
        (df_manager.all_scenes_df["video_id"] == video_id)
        & (df_manager.all_scenes_df["scene_snapshot_path"].notnull())
        & (df_manager.all_scenes_df["nudity_present"].isnull())
    ]

    if rows.empty:
        logging.info("Nudity detection already done for video %s or no scenes to process.", video_id)
        return

    try:
        for idx, scene_image in tqdm(
            zip(rows.index, rows.itertuples(index=False)),
            total=len(rows),
            desc=f"Checking nudity in {video_id}",
            unit="scene frames",
        ):
            nudity_present = detect_nudity(scene_image.scene_snapshot_path)
            df_manager.all_scenes_df.at[idx, "nudity_present"] = nudity_present
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt caught while detecting nudity. Saving checkpoint before exiting...")
        config.save_checkpoint()
        raise

    logging.info("Finished nudity detection for video %s. Saving checkpoint...", video_id)
    config.save_checkpoint()


def generate_detailed_caption(image_path):
    prompt = (
        "Describe this TV scene frame briefly and factually. "
        "Mention people, clothing coverage, body exposure, intimacy, and whether nudity is visible."
    )
    response = ollama_client.chat(
        model="qwen3-vl:4b",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
    )
    return response["message"]["content"].strip()


def generate_descriptions_for_nude_scenes(video_id):
    scene_rows = df_manager.all_scenes_df[
        (df_manager.all_scenes_df["video_id"] == video_id)
    ].copy()
    subtitle_rows = scene_rows[scene_rows["subtitle"].notna()].copy()
    sexual_pattern = re.compile(
        r"\b(naked|nude|breast|breasts|cock|dick|penis|vagina|ass|arse|fuck|fucking|whore|brothel|sex|sexual|kiss|bed|boned|virgin|cunt|slut|ride|undress|moan)\b",
        re.IGNORECASE,
    )
    risky_scene_numbers = set(
        subtitle_rows[
            subtitle_rows["subtitle"].astype(str).str.contains(sexual_pattern, na=False)
        ]["scene_number"].dropna().astype(int).tolist()
    )
    expanded_scene_numbers = set()
    for scene_number in risky_scene_numbers:
        for offset in range(-2, 3):
            expanded_scene_numbers.add(scene_number + offset)

    scene_number_series = scene_rows["scene_number"].astype("Int64")
    should_process_all_scenes = subtitle_rows.empty
    rows = scene_rows[
        (scene_rows["scene_snapshot_path"].notnull())
        & (scene_rows["scene_number"].notnull())
        & (scene_rows["snapshot_desc"].isnull())
        & (
            should_process_all_scenes
            | 
            (scene_rows["nudity_present"] == True)
            | (scene_number_series.isin(expanded_scene_numbers))
        )
    ].copy()

    if rows.empty:
        logging.info("No scene descriptions to generate for video %s.", video_id)
        return

    rows["scene_snapshot_number"] = rows["scene_snapshot_number"].astype(float)
    rows["distance_from_center"] = (rows["scene_snapshot_number"] - 4).abs()
    rows = rows.sort_values(
        by=["scene_number", "distance_from_center", "scene_snapshot_number"]
    ).drop_duplicates(subset=["scene_number"], keep="first")

    try:
        for idx, scene_image in tqdm(
            zip(rows.index, rows.itertuples(index=False)),
            total=len(rows),
            desc=f"Generating descriptions for {video_id}",
            unit="scene frames",
        ):
            description = generate_detailed_caption(scene_image.scene_snapshot_path)
            df_manager.all_scenes_df.at[idx, "snapshot_desc"] = description
    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        config.save_checkpoint()
        raise

    config.save_checkpoint()
