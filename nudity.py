import json
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

IGNORED_DETECTOR_TERMS = {"face", "feet", "foot"}
STRONG_EXPLICIT_TERMS = {
    "exposed_breast",
    "exposed_genitalia",
    "exposed_buttocks",
    "exposed_anus",
}
SOFT_EXPOSURE_TERMS = {
    "covered_breast",
    "covered_buttocks",
}


def _normalize_detector_results(results):
    filtered = []
    for result in results:
        label = str(result.get("class", "")).strip()
        label_lower = label.lower()
        if any(term in label_lower for term in IGNORED_DETECTOR_TERMS):
            continue
        filtered.append(
            {
                "class": label,
                "score": float(result.get("score", 0.0)),
                "box": result.get("box", []),
            }
        )
    return filtered


def _detector_signal(results):
    max_score = max((result["score"] for result in results), default=0.0)
    labels = sorted({result["class"] for result in results})
    strong_hits = [
        result
        for result in results
        if any(term in result["class"].lower() for term in STRONG_EXPLICIT_TERMS)
        and result["score"] >= 0.35
    ]
    soft_hits = [
        result
        for result in results
        if any(term in result["class"].lower() for term in SOFT_EXPOSURE_TERMS)
        and result["score"] >= 0.70
    ]
    nudity_present = bool(strong_hits or soft_hits)
    return {
        "labels": labels,
        "max_score": max_score,
        "nudity_present": nudity_present,
        "raw": json.dumps(results, ensure_ascii=True),
    }


def detect_nudity(image_path):
    raw_results = detector.detect(image_path)
    results = _normalize_detector_results(raw_results)
    return _detector_signal(results)


def detect_nudity_in_video(video_id):
    rows = df_manager.all_scenes_df[
        (df_manager.all_scenes_df["video_id"] == video_id)
        & (df_manager.all_scenes_df["scene_snapshot_path"].notnull())
        & (
            df_manager.all_scenes_df["detector_raw"].isnull()
            | df_manager.all_scenes_df["detector_raw"].astype(str).eq("")
        )
    ]

    if rows.empty:
        logging.info("Nudity detection already done for video %s or no scenes to process.", video_id)
        mark_scene_representative_frames(video_id)
        return

    try:
        for idx, scene_image in tqdm(
            zip(rows.index, rows.itertuples(index=False)),
            total=len(rows),
            desc=f"Checking nudity in {video_id}",
            unit="scene frames",
        ):
            detector_signal = detect_nudity(scene_image.scene_snapshot_path)
            df_manager.all_scenes_df.at[idx, "detector_labels"] = ", ".join(detector_signal["labels"])
            df_manager.all_scenes_df.at[idx, "detector_max_score"] = detector_signal["max_score"]
            df_manager.all_scenes_df.at[idx, "detector_raw"] = detector_signal["raw"]
            df_manager.all_scenes_df.at[idx, "nudity_present"] = detector_signal["nudity_present"]
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt caught while detecting nudity. Saving checkpoint before exiting...")
        config.save_checkpoint()
        raise

    mark_scene_representative_frames(video_id)
    logging.info("Finished nudity detection for video %s. Saving checkpoint...", video_id)
    config.save_checkpoint()


def _select_representative_indices(scene_rows):
    ordered = scene_rows.sort_values(by=["timestamp", "scene_snapshot_number"]).copy()
    indices = []
    count = len(ordered)
    if count == 0:
        return indices

    center_index = ordered.index[count // 2]
    indices.append(center_index)

    scored = ordered.sort_values(
        by=["detector_max_score", "distance_from_center"],
        ascending=[False, True],
    )
    for idx in scored.index.tolist():
        if idx not in indices:
            indices.append(idx)
        if len(indices) >= 3:
            break

    if len(indices) < min(3, count):
        for idx in ordered.index.tolist():
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= min(3, count):
                break

    return indices[:3]


def mark_scene_representative_frames(video_id):
    scene_rows = df_manager.all_scenes_df[
        (df_manager.all_scenes_df["video_id"] == video_id)
        & (df_manager.all_scenes_df["scene_snapshot_path"].notnull())
        & (df_manager.all_scenes_df["scene_number"].notnull())
    ].copy()

    if scene_rows.empty:
        return

    df_manager.all_scenes_df.loc[scene_rows.index, "frame_role"] = "support"
    scene_rows["scene_snapshot_number"] = scene_rows["scene_snapshot_number"].astype(float)

    for scene_number, group in scene_rows.groupby("scene_number"):
        group = group.copy()
        group["distance_from_center"] = (group["scene_snapshot_number"] - 4).abs()
        representative_indices = _select_representative_indices(group)
        role_names = ["middle", "detector_top", "detector_backup"]
        for role_name, idx in zip(role_names, representative_indices):
            df_manager.all_scenes_df.at[idx, "frame_role"] = role_name


def _strip_code_fences(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _parse_json_response(text):
    stripped = _strip_code_fences(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _parse_non_json_classification(text):
    stripped = _strip_code_fences(text)
    lowered = stripped.lower()

    def contains_any(phrases):
        return any(phrase in lowered for phrase in phrases)

    no_nudity = contains_any(
        [
            "no nudity",
            "nudity is not visible",
            "no visible nudity",
            "fully clothed",
            "no explicit exposure",
        ]
    )
    visible_nudity = (
        not no_nudity
        and contains_any(
            [
                "topless",
                "bare breasts",
                "exposed breasts",
                "nude",
                "nudity is visible",
                "genitals visible",
                "buttocks visible",
            ]
        )
    )

    if contains_any(["genitals visible", "full nudity", "clear nudity", "explicit nudity"]):
        exposure = "clear"
    elif visible_nudity or contains_any(["partial nudity", "topless", "exposed chest", "bare chest"]):
        exposure = "partial"
    else:
        exposure = "none"

    sexual_activity = contains_any(
        [
            "sexual activity",
            "sex act",
            "having sex",
            "intercourse",
        ]
    )

    confidence_match = re.search(r"\b(?:confidence|score)\D{0,10}([01](?:\.\d+)?)", lowered)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.55

    summary = stripped.splitlines()[0].strip()[:160] if stripped.strip() else "Unstructured model response"
    reason_short = summary
    return {
        "summary": summary,
        "visible_nudity": visible_nudity,
        "explicit_body_exposure": exposure,
        "sexual_activity": sexual_activity,
        "intimacy_level": "sexual" if sexual_activity else ("suggestive" if visible_nudity else "none"),
        "confidence": confidence,
        "reason_short": reason_short,
    }


def classify_frame(image_path, censorship_strength):
    if censorship_strength and censorship_strength.name == "STRICT":
        policy_line = (
            "STRICT policy: return visible_nudity=true for any toplessness, underwear-only sexual framing, "
            "clear body exposure, or scene likely to be sexually intimate even if genitals are not visible."
        )
    else:
        policy_line = (
            "MODERATE policy: return visible_nudity=true only for clearly visible breasts, genitals, buttocks, anus, "
            "or unmistakable sexual activity. Do not mark clothed, armored, romantic, or merely close scenes as nudity."
        )

    prompt = f"""Classify this single TV frame for censorship.

{policy_line}

Return JSON only with this exact schema:
{{
  "summary": "short factual description",
  "visible_nudity": true,
  "explicit_body_exposure": "none",
  "sexual_activity": false,
  "intimacy_level": "none",
  "confidence": 0.0,
  "reason_short": "brief reason"
}}

Rules:
- explicit_body_exposure must be one of: "none", "partial", "clear"
- intimacy_level must be one of: "none", "suggestive", "sexual"
- Be conservative. Do not hallucinate body exposure or identify actors.
- If unsure, lower confidence instead of guessing.
"""
    last_content = ""
    for _attempt in range(3):
        response = ollama_client.chat(
            model="qwen3-vl:4b",
            format="json",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_path)],
                }
            ],
        )
        last_content = response["message"]["content"]
        if not last_content or not last_content.strip():
            continue
        try:
            parsed = _parse_json_response(last_content)
            break
        except json.JSONDecodeError:
            parsed = _parse_non_json_classification(last_content)
            break
    else:
        parsed = _parse_non_json_classification(last_content)

    parsed["summary"] = str(parsed.get("summary", "")).strip()
    parsed["visible_nudity"] = bool(parsed.get("visible_nudity", False))
    parsed["explicit_body_exposure"] = str(parsed.get("explicit_body_exposure", "none")).strip().lower()
    if parsed["explicit_body_exposure"] not in {"none", "partial", "clear"}:
        parsed["explicit_body_exposure"] = "none"
    parsed["sexual_activity"] = bool(parsed.get("sexual_activity", False))
    parsed["intimacy_level"] = str(parsed.get("intimacy_level", "none")).strip().lower()
    if parsed["intimacy_level"] not in {"none", "suggestive", "sexual"}:
        parsed["intimacy_level"] = "none"
    try:
        parsed["confidence"] = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        parsed["confidence"] = 0.0
    parsed["confidence"] = max(0.0, min(1.0, parsed["confidence"]))
    parsed["reason_short"] = str(parsed.get("reason_short", "")).strip()
    return parsed


def generate_descriptions_for_nude_scenes(video_id):
    mark_scene_representative_frames(video_id)
    rows = df_manager.all_scenes_df[
        (df_manager.all_scenes_df["video_id"] == video_id)
        & (df_manager.all_scenes_df["scene_snapshot_path"].notnull())
        & (df_manager.all_scenes_df["frame_role"].notnull())
        & (
            df_manager.all_scenes_df["vision_raw"].isnull()
            | df_manager.all_scenes_df["vision_raw"].astype(str).eq("")
        )
        & (df_manager.all_scenes_df["frame_role"].astype(str) != "support")
    ].copy()

    if rows.empty:
        logging.info("No representative scene frames left to classify for video %s.", video_id)
        return

    rows = rows.sort_values(by=["scene_number", "timestamp"])

    try:
        for idx, scene_image in tqdm(
            zip(rows.index, rows.itertuples(index=False)),
            total=len(rows),
            desc=f"Classifying frames for {video_id}",
            unit="frames",
        ):
            classification = classify_frame(
                scene_image.scene_snapshot_path,
                config.censorship_strength,
            )
            df_manager.all_scenes_df.at[idx, "snapshot_desc"] = classification["summary"]
            df_manager.all_scenes_df.at[idx, "visible_nudity"] = classification["visible_nudity"]
            df_manager.all_scenes_df.at[idx, "explicit_exposure"] = classification["explicit_body_exposure"]
            df_manager.all_scenes_df.at[idx, "sexual_activity"] = classification["sexual_activity"]
            df_manager.all_scenes_df.at[idx, "vision_confidence"] = classification["confidence"]
            df_manager.all_scenes_df.at[idx, "vision_reason"] = classification["reason_short"]
            df_manager.all_scenes_df.at[idx, "vision_raw"] = json.dumps(classification, ensure_ascii=True)
            config.save_checkpoint()
    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        config.save_checkpoint()
        raise
    except Exception:
        config.save_checkpoint()
        raise

    config.save_checkpoint()
