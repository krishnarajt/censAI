import logging
import os

import ollama
from config.settings import Config
from data.dataframe_manager import ScenesDataFrameManager

config = Config()
df_manager = ScenesDataFrameManager()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if OLLAMA_HOST == "0.0.0.0" or OLLAMA_HOST.startswith("0.0.0.0"):
    OLLAMA_HOST = "http://localhost:11434"
elif not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"
os.environ["OLLAMA_HOST"] = OLLAMA_HOST


def clean_text(text):
    prompt = f"""Return only the sanitized text. Remove all profanity while preserving the original meaning. If a single word is profane, replace it with a more appropriate alternative. For text with sexual implications, rewrite it in a kid-friendly manner. Do not provide explanations or advice. Only output the modified text. Give only 1 sentence.

Input: "Fuck!"
Output: "Shoot!"

Input: "This is fucking stupid!"
Output: "This is freaking stupid!"

Input: "Fuck you!"
Output: "Forget you!"

Input: "This is so fucked!"
Output: "This is a total mess!"

Input: "I wanna fuck you so badly"
Output: "I want us to be close."

Now sanitize:
{text}
"""
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def determine_if_should_censor_scene(image_descriptions: list, subtitles: list, nudity_present: bool):
    prompt = f"""You are given raw descriptions of frames from a TV scene plus subtitle lines.

Return only one word: true or false.

Return true if the scene likely contains nudity, sexual activity, explicit body exposure, or strongly sexual dialogue/themes.
Return false otherwise.

Frame Descriptions:
{", ".join(image_descriptions) if image_descriptions else "None"}

Subtitles:
{", ".join(subtitles) if subtitles else "None"}

Nude detector hint:
{"nudity detected" if nudity_present else "no nudity detected"}
"""
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip().lower() == "true"


def determine_if_should_censor_video(video_id: int):
    rows = df_manager.all_scenes_df[
        (df_manager.all_scenes_df["video_id"] == video_id)
        & (df_manager.all_scenes_df["scene_number"].notna())
    ]

    if rows.empty:
        logging.info("No scenes to process for video %s.", video_id)
        return False

    try:
        for scene_number, scene_rows in rows.groupby("scene_number"):
            if scene_rows["should_censor"].notna().all():
                continue

            image_descriptions = [
                desc for desc in scene_rows["snapshot_desc"].dropna().astype(str).tolist()
                if desc.strip()
            ]
            subtitles = [
                subtitle for subtitle in scene_rows["subtitle"].dropna().astype(str).tolist()
                if subtitle.strip()
            ]
            nudity_present = bool(scene_rows["nudity_present"].fillna(False).any())

            if not nudity_present and not image_descriptions:
                should_censor = False
            else:
                try:
                    should_censor = determine_if_should_censor_scene(
                        image_descriptions,
                        subtitles,
                        nudity_present,
                    )
                except Exception as exc:
                    logging.warning(
                        "Scene classification failed for video %s scene %s: %s. Falling back to nudity detection.",
                        video_id,
                        scene_number,
                        exc,
                    )
                    should_censor = nudity_present

            df_manager.all_scenes_df.loc[scene_rows.index, "should_censor"] = should_censor
    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        config.save_checkpoint()
        raise

    config.save_checkpoint()
    logging.info("Completed processing for video %s.", video_id)
    return True
