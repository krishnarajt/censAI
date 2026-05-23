import logging
import os
import re

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

EXPLICIT_SUBTITLE_PATTERN = re.compile(
    r"\b("
    r"naked|nude|breast|breasts|nipples?|genitals?|penis|vagina|cock|dick|ass|arse|"
    r"fuck|fucking|boned|whore|brothel|sex|sexual|kiss|bed|undress|moan|ride me|take off"
    r")\b",
    re.IGNORECASE,
)


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


def _scene_has_explicit_subtitles(subtitles):
    return any(EXPLICIT_SUBTITLE_PATTERN.search(subtitle or "") for subtitle in subtitles)


def _frame_vote(row):
    confidence = float(row.get("vision_confidence") or 0.0)
    visible_nudity = bool(row.get("visible_nudity") is True)
    sexual_activity = bool(row.get("sexual_activity") is True)
    exposure = str(row.get("explicit_exposure") or "none").lower()
    detector_score = float(row.get("detector_max_score") or 0.0)
    detector_labels = str(row.get("detector_labels") or "").lower()

    strong_detector = detector_score >= 0.50 and any(
        keyword in detector_labels
        for keyword in ["exposed_breast", "exposed_genitalia", "exposed_buttocks", "exposed_anus"]
    )
    soft_detector = detector_score >= 0.75 and "covered_breast" in detector_labels

    return {
        "high_conf_explicit": visible_nudity and exposure == "clear" and confidence >= 0.70,
        "visible_nudity": visible_nudity and confidence >= 0.55,
        "partial_nudity": visible_nudity and exposure == "partial" and confidence >= 0.60,
        "sexual_activity": sexual_activity and confidence >= 0.60,
        "negative_guard": (not visible_nudity) and (not sexual_activity) and exposure == "none" and confidence >= 0.65,
        "strong_detector": strong_detector,
        "soft_detector": soft_detector,
        "confidence": confidence,
    }


def determine_if_should_censor_video(video_id: int):
    rows = df_manager.all_scenes_df[
        (df_manager.all_scenes_df["video_id"] == video_id)
        & (df_manager.all_scenes_df["scene_number"].notna())
    ]

    if rows.empty:
        logging.info("No scenes to process for video %s.", video_id)
        return False

    is_strict = bool(config.censorship_strength and config.censorship_strength.name == "STRICT")

    try:
        for scene_number, scene_rows in rows.groupby("scene_number"):
            representative_rows = scene_rows[
                scene_rows["frame_role"].notna() & (scene_rows["frame_role"].astype(str) != "support")
            ].copy()

            subtitles = [
                subtitle for subtitle in scene_rows["subtitle"].dropna().astype(str).tolist()
                if subtitle.strip()
            ]
            explicit_subtitles = _scene_has_explicit_subtitles(subtitles)

            frame_votes = [_frame_vote(row) for _, row in representative_rows.iterrows()]
            visible_count = sum(vote["visible_nudity"] for vote in frame_votes)
            partial_count = sum(vote["partial_nudity"] for vote in frame_votes)
            sexual_count = sum(vote["sexual_activity"] for vote in frame_votes)
            negative_count = sum(vote["negative_guard"] for vote in frame_votes)
            strong_detector_count = sum(vote["strong_detector"] for vote in frame_votes)
            soft_detector_count = sum(vote["soft_detector"] for vote in frame_votes)
            has_high_conf_explicit = any(vote["high_conf_explicit"] for vote in frame_votes)

            if representative_rows.empty:
                should_censor = bool(explicit_subtitles and is_strict)
            elif (
                negative_count == len(frame_votes)
                and strong_detector_count == 0
                and soft_detector_count == 0
                and not explicit_subtitles
            ):
                should_censor = False
            elif is_strict:
                should_censor = bool(
                    visible_count >= 1
                    or partial_count >= 1
                    or sexual_count >= 1
                    or strong_detector_count >= 1
                    or soft_detector_count >= 1
                    or explicit_subtitles
                )
            else:
                should_censor = bool(
                    has_high_conf_explicit
                    or visible_count >= 2
                    or sexual_count >= 1 and visible_count >= 1
                    or strong_detector_count >= 2
                    or partial_count >= 2
                )

            df_manager.all_scenes_df.loc[scene_rows.index, "should_censor"] = should_censor
    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        config.save_checkpoint()
        raise

    config.save_checkpoint()
    logging.info("Completed processing for video %s.", video_id)
    return True
