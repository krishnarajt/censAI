"""Nudity / vision pipeline.

Two stages:
  1. NudeNet (ONNX) runs on every snapshot frame. Output is cached by image
     sha256.
  2. A local vision LLM (default qwen3-vl:4b) classifies a subset of frames
     per scene: the middle frame plus every frame whose NudeNet score crossed
     a threshold, capped at MAX_VISION_FRAMES_PER_SCENE. Output is cached by
     (sha256, model), with a (phash, model) -> sha256 alias so near-identical
     frames hit the cache for free.

Aggregation (per-scene censor / no-censor decision) lives in profanity.py
alongside the subtitle-based signal, since the two are combined together.
"""

import json
import logging
import os
import re

from nudenet import NudeDetector
from ollama import Client
from tqdm import tqdm

from config.settings import Config
from db.store import Store

config = Config()
store = Store()
os.environ["OLLAMA_HOST"] = config.ollama_host
detector = NudeDetector(model_path="models/640m.onnx", inference_resolution=640)
ollama_client = Client(host=config.ollama_host)

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
DETECTOR_LABEL_ALIASES = {
    "female_breast_exposed": "exposed_breast",
    "male_breast_exposed": "male_breast_exposed",
    "female_genitalia_exposed": "exposed_genitalia",
    "male_genitalia_exposed": "exposed_genitalia",
    "buttocks_exposed": "exposed_buttocks",
    "anus_exposed": "exposed_anus",
    "female_breast_covered": "covered_breast",
    "buttocks_covered": "covered_buttocks",
}

VISION_JSON_SCHEMA = {
    "type": "object",
    "required": [
        "summary",
        "visible_nudity",
        "explicit_body_exposure",
        "sexual_activity",
        "intimacy_level",
        "confidence",
        "reason_short",
    ],
    "properties": {
        "summary": {"type": "string"},
        "visible_nudity": {"type": "boolean"},
        "explicit_body_exposure": {
            "type": "string",
            "enum": ["none", "partial", "clear"],
        },
        "sexual_activity": {"type": "boolean"},
        "intimacy_level": {
            "type": "string",
            "enum": ["none", "suggestive", "sexual"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason_short": {"type": "string"},
    },
}

VISION_PROMPT = """You are a content moderation reviewer for TV and film.
Look at this single video frame and classify ONLY what is visually present.
Base every answer strictly on what is actually visible in this one image.
Do not infer from context, do not identify actors, do not guess.

Definitions:
- visible_nudity: TRUE only if you can clearly see a bare female nipple/areola,
  bare buttocks, bare genitals, or a bare anus. Cleavage, swimwear, lingerie,
  shirtless men, tight clothing, and underwear-only shots are NOT visible_nudity.
- explicit_body_exposure:
    "clear"   = unmistakable bare nipple/genitals/buttocks/anus visible.
    "partial" = bare exposure of those areas is partially visible or partly obscured.
    "none"    = no bare exposure of those areas.
- sexual_activity: TRUE only if people are clearly engaged in or actively
  simulating a sex act (intercourse, oral sex, masturbation, etc.). Kissing
  alone, hugging, and dancing are NOT sexual_activity.
- intimacy_level:
    "sexual"     = a sex act is visible.
    "suggestive" = visibly sensual/intimate without an actual sex act
                   (e.g. two undressed-but-covered people in bed, heavy makeout
                   with partial undress).
    "none"       = neither of the above.
- confidence: 0.0 to 1.0. If you are unsure, output a LOW confidence value.
  Never set confidence high just to look certain.
- summary: 2 to 4 factual sentences describing the frame in useful detail.
  Mention people, pose/action, clothing/undress state, visible body exposure,
  and setting/objects when relevant. Stay literal and avoid guessing.
- reason_short: one short sentence explaining your flag values.

Return JSON only. Do not include any other text.
"""


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def _normalize_detector_results(results):
    filtered = []
    for result in results:
        raw_label = str(result.get("class", "")).strip()
        label = raw_label.lower()
        if any(term in label for term in IGNORED_DETECTOR_TERMS):
            continue
        canonical = DETECTOR_LABEL_ALIASES.get(label, label)
        filtered.append(
            {
                "class": canonical,
                "raw_class": raw_label,
                "score": float(result.get("score", 0.0)),
                "box": result.get("box", []),
            }
        )
    return filtered


def _detector_signal(results):
    max_score = max((r["score"] for r in results), default=0.0)
    labels = sorted({r["class"] for r in results})
    strong_hits = [
        r
        for r in results
        if r["class"] in STRONG_EXPLICIT_TERMS
        and r["score"] >= config.NUDENET_STRONG_THRESHOLD
    ]
    soft_hits = [
        r
        for r in results
        if r["class"] in SOFT_EXPOSURE_TERMS
        and r["score"] >= config.NUDENET_SOFT_THRESHOLD
    ]
    return {
        "labels": labels,
        "max_score": max_score,
        "nudity_present": bool(strong_hits or soft_hits),
        "raw": results,
    }


def _signal_from_cached_detector_row(cached):
    raw_json = cached["raw_json"] if cached is not None else None
    if raw_json:
        try:
            raw_results = json.loads(raw_json)
        except json.JSONDecodeError:
            raw_results = []
        normalized = _normalize_detector_results(raw_results)
        return _detector_signal(normalized)

    labels = []
    for label in str(cached["labels"] or "").split(","):
        normalized = label.strip().lower()
        if not normalized:
            continue
        labels.append(DETECTOR_LABEL_ALIASES.get(normalized, normalized))
    max_score = float(cached["max_score"] or 0.0)
    strong = max_score >= config.NUDENET_STRONG_THRESHOLD and any(
        label in STRONG_EXPLICIT_TERMS for label in labels
    )
    soft = max_score >= config.NUDENET_SOFT_THRESHOLD and any(
        label in SOFT_EXPOSURE_TERMS for label in labels
    )
    return {
        "labels": labels,
        "max_score": max_score,
        "nudity_present": bool(strong or soft),
        "raw": [],
    }


def detect_nudity_in_video(video_id):
    frames = [row for row in store.get_frames(video_id) if row["image_sha256"]]
    if not frames:
        logging.info("No frames to scan for video %s.", video_id)
        mark_scene_representative_frames(video_id)
        return

    new_calls = 0
    repaired_cached_rows = 0
    for frame in tqdm(frames, desc=f"NudeNet on {video_id}", unit="frame"):
        sha = frame["image_sha256"]
        cached = store.get_nudity_cache(sha)
        if cached is not None:
            refreshed = _signal_from_cached_detector_row(cached)
            cached_flag = bool(cached["nudity_present"])
            cached_labels = str(cached["labels"] or "")
            desired_labels = ", ".join(refreshed["labels"])
            if (
                cached_flag != refreshed["nudity_present"]
                or cached_labels != desired_labels
            ):
                store.set_nudity_cache(
                    image_sha256=sha,
                    labels=refreshed["labels"],
                    max_score=refreshed["max_score"],
                    raw=refreshed["raw"],
                    nudity_present=refreshed["nudity_present"],
                )
                repaired_cached_rows += 1
            continue
        raw = detector.detect(frame["snapshot_path"])
        results = _normalize_detector_results(raw)
        signal = _detector_signal(results)
        store.set_nudity_cache(
            image_sha256=sha,
            labels=signal["labels"],
            max_score=signal["max_score"],
            raw=signal["raw"],
            nudity_present=signal["nudity_present"],
        )
        new_calls += 1

    mark_scene_representative_frames(video_id)
    logging.info(
        "NudeNet pass for video %s: %d new detections cached, %d cached rows repaired, %d frames total.",
        video_id,
        new_calls,
        repaired_cached_rows,
        len(frames),
    )


# ---------------------------------------------------------------------------
# Representative frame selection
# ---------------------------------------------------------------------------


def _select_vision_frame_ids(frames_with_cache):
    """Pick which frames in a scene should be sent to the vision LLM.

    Strategy:
      - Always include the middle snapshot for context.
      - Include every frame whose NudeNet max_score >= NUDENET_STRONG_THRESHOLD.
      - Sort by score desc; cap at MAX_VISION_FRAMES_PER_SCENE total.
    """
    frames = list(frames_with_cache)
    if not frames:
        return []

    middle_idx = len(frames) // 2
    selected = {frames[middle_idx]["id"]: ("middle", 0.0)}

    flagged = [
        f
        for f in frames
        if (f["det_max_score"] or 0.0) >= config.NUDENET_STRONG_THRESHOLD
    ]
    flagged.sort(key=lambda f: float(f["det_max_score"] or 0.0), reverse=True)
    for f in flagged:
        if f["id"] in selected:
            continue
        if len(selected) >= config.MAX_VISION_FRAMES_PER_SCENE:
            break
        selected[f["id"]] = ("detector_hit", float(f["det_max_score"] or 0.0))

    return list(selected.items())  # [(frame_id, (role, score)), ...]


def mark_scene_representative_frames(video_id):
    """Tag the frames we intend to classify with vision so the next stage
    knows what to process. The role column is informational only."""
    store.clear_frame_roles(video_id)
    scene_numbers = store.get_scene_numbers(video_id)
    for scene_number in scene_numbers:
        frames = store.get_scene_frames_with_caches(video_id, scene_number)
        for frame_id, (role, _score) in _select_vision_frame_ids(frames):
            store.set_frame_role(frame_id, role)


# ---------------------------------------------------------------------------
# Vision LLM
# ---------------------------------------------------------------------------


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


def _coerce_parsed(parsed):
    """Defensive coercion so a malformed model response can never silently
    elevate confidence or flags."""
    out = {}
    out["summary"] = str(parsed.get("summary", "")).strip()
    out["visible_nudity"] = bool(parsed.get("visible_nudity", False))
    exposure = str(parsed.get("explicit_body_exposure", "none")).strip().lower()
    out["explicit_body_exposure"] = exposure if exposure in {"none", "partial", "clear"} else "none"
    out["sexual_activity"] = bool(parsed.get("sexual_activity", False))
    intimacy = str(parsed.get("intimacy_level", "none")).strip().lower()
    out["intimacy_level"] = intimacy if intimacy in {"none", "suggestive", "sexual"} else "none"
    try:
        out["confidence"] = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))
    out["reason_short"] = str(parsed.get("reason_short", "")).strip()
    return out


def _classify_frame_via_llm(image_path):
    """Call the vision model. Returns (parsed_dict, raw_text)."""
    last_content = ""
    parsed = None
    for _ in range(3):
        response = ollama_client.chat(
            model=config.VISION_MODEL,
            format=VISION_JSON_SCHEMA,
            options={"temperature": 0.0},
            messages=[
                {
                    "role": "user",
                    "content": VISION_PROMPT,
                    "images": [str(image_path)],
                }
            ],
        )
        last_content = response["message"]["content"] or ""
        if not last_content.strip():
            continue
        try:
            parsed = _parse_json_response(last_content)
            break
        except json.JSONDecodeError:
            parsed = None
            continue

    if parsed is None:
        # Schema-constrained call failed three times. Fail closed: confidence 0,
        # all flags false. The aggregator will treat this as "no signal" rather
        # than a false positive.
        parsed = {
            "summary": "Model produced no structured output.",
            "visible_nudity": False,
            "explicit_body_exposure": "none",
            "sexual_activity": False,
            "intimacy_level": "none",
            "confidence": 0.0,
            "reason_short": "fallback: unparseable response",
        }

    return _coerce_parsed(parsed), last_content


def classify_frame(image_path, image_sha256, image_phash):
    """Cached frame classification. Order of lookup:
        1. vision_cache by sha256+model
        2. vision_phash_alias by phash+model (near-duplicate frame within a scene)
        3. Call the model and store both.
    """
    model = config.VISION_MODEL
    cached = store.get_vision_cache(image_sha256, model)
    if cached is not None:
        return {
            "summary": cached["summary"],
            "visible_nudity": bool(cached["visible_nudity"]),
            "explicit_body_exposure": cached["explicit_exposure"] or "none",
            "sexual_activity": bool(cached["sexual_activity"]),
            "intimacy_level": cached["intimacy_level"] or "none",
            "confidence": float(cached["confidence"] or 0.0),
            "reason_short": cached["reason"] or "",
        }

    if image_phash:
        aliased = store.get_vision_cache_by_phash(image_phash, model)
        if aliased is not None:
            parsed = {
                "summary": aliased["summary"],
                "visible_nudity": bool(aliased["visible_nudity"]),
                "explicit_body_exposure": aliased["explicit_exposure"] or "none",
                "sexual_activity": bool(aliased["sexual_activity"]),
                "intimacy_level": aliased["intimacy_level"] or "none",
                "confidence": float(aliased["confidence"] or 0.0),
                "reason_short": aliased["reason"] or "",
            }
            store.set_vision_cache(
                image_sha256=image_sha256,
                model=model,
                parsed=parsed,
                raw=aliased["raw_json"] or "",
                image_phash=image_phash,
            )
            return parsed

    parsed, raw = _classify_frame_via_llm(image_path)
    store.set_vision_cache(
        image_sha256=image_sha256,
        model=model,
        parsed=parsed,
        raw=raw,
        image_phash=image_phash,
    )
    return parsed


def generate_descriptions_for_nude_scenes(video_id):
    mark_scene_representative_frames(video_id)
    scene_numbers = store.get_scene_numbers(video_id)
    if not scene_numbers:
        logging.info("No scenes to classify for video %s.", video_id)
        return

    # Build a flat work list of frames-to-classify, in scene order.
    work = []
    for scene_number in scene_numbers:
        for frame in store.get_scene_frames_with_caches(video_id, scene_number):
            if not frame["frame_role"] or frame["frame_role"] == "support":
                continue
            if not frame["image_sha256"]:
                continue
            work.append(frame)

    if not work:
        logging.info("No representative frames left to classify for video %s.", video_id)
        return

    new_calls = 0
    try:
        for frame in tqdm(work, desc=f"Vision on {video_id}", unit="frame"):
            sha = frame["image_sha256"]
            cached_before = store.get_vision_cache(sha, config.VISION_MODEL)
            classify_frame(
                frame["snapshot_path"],
                sha,
                frame["image_phash"],
            )
            if cached_before is None:
                # check if we ended up filling the cache via phash alias vs
                # actually calling the LLM
                new_calls += 1
    except KeyboardInterrupt:
        print("\nInterrupted! Cache so far is already persisted.")
        raise

    logging.info(
        "Vision pass for video %s: ~%d new/aliased classifications, %d frames considered.",
        video_id,
        new_calls,
        len(work),
    )
