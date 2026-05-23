"""Subtitle rewriting (profanity sanitization) + per-scene censor decision.

The per-scene aggregation pulls together NudeNet + vision LLM signals from
the DB and combines them with subtitle context to produce a single
should_censor decision. Aggregation logic is kept here, alongside the
subtitle pieces, because subtitle text is one of the signals fed into it.
"""

import logging
import os
import re

import ollama

import util
from config.settings import Config
from db.store import Store

config = Config()
store = Store()
os.environ["OLLAMA_HOST"] = config.ollama_host

STRONG_DETECTOR_LABELS = {
    "exposed_breast",
    "exposed_genitalia",
    "exposed_buttocks",
    "exposed_anus",
    "female_breast_exposed",
    "female_genitalia_exposed",
    "male_genitalia_exposed",
    "buttocks_exposed",
    "anus_exposed",
}
SOFT_DETECTOR_LABELS = {
    "covered_breast",
    "covered_buttocks",
    "female_breast_covered",
    "buttocks_covered",
}

# Tightened. The previous list flagged "kiss", "bed", "ass", "sex" which fire
# on innocent text like "we went to bed", "kick ass", etc. We keep only words
# that are essentially never used in a non-sexual context in TV dialogue.
EXPLICIT_SUBTITLE_PATTERN = re.compile(
    r"\b("
    r"naked|nude|topless|"
    r"breasts?|nipples?|genitals?|penis|vagina|clitoris|"
    r"cock|dick|cunt|"
    r"fuck(?:ing|s|ed)?|"
    r"intercourse|orgasm|"
    r"undress(?:ed|ing)?|"
    r"masturbat(?:e|ing|ion)|"
    r"horny|aroused"
    r")\b",
    re.IGNORECASE,
)

PROFANITY_PROMPT_TEMPLATE = """Return only the sanitized text. Remove all profanity while preserving the original meaning. If a single word is profane, replace it with a more appropriate alternative. For text with sexual implications, rewrite it in a kid-friendly manner. Do not provide explanations or advice. Only output the modified text. Give only 1 sentence.

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


def _call_profanity_llm(text: str) -> str:
    response = ollama.chat(
        model=config.PROFANITY_MODEL,
        options={"temperature": 0.1},
        messages=[
            {"role": "user", "content": PROFANITY_PROMPT_TEMPLATE.format(text=text)},
        ],
    )
    return response["message"]["content"].strip()


def clean_text_cached(text: str, text_hash: str | None = None) -> str:
    """Cached subtitle rewrite. Same text + same model => zero LLM calls."""
    if text_hash is None:
        text_hash = util.sha256_text(text)
    cached = store.get_profanity_cache(text_hash, config.PROFANITY_MODEL)
    if cached is not None:
        return cached
    cleaned = _call_profanity_llm(text)
    store.set_profanity_cache(text_hash, config.PROFANITY_MODEL, cleaned)
    return cleaned


# Back-compat name used by sub_manip.
def clean_text(text):
    return clean_text_cached(text)


# ---------------------------------------------------------------------------
# Per-scene aggregation
# ---------------------------------------------------------------------------


def _scene_subtitle_texts(video_id: int, scene_number: int) -> list[str]:
    rows = store.conn.execute(
        "SELECT text FROM subtitles WHERE video_id = ? AND scene_number = ?",
        (video_id, scene_number),
    ).fetchall()
    return [r["text"] for r in rows if r["text"]]


def _scene_has_explicit_subtitles(subtitles):
    return any(EXPLICIT_SUBTITLE_PATTERN.search(s or "") for s in subtitles)


def _frame_vote(row):
    confidence = float(row["v_confidence"] or 0.0)
    visible_nudity = bool(row["v_visible_nudity"])
    sexual_activity = bool(row["v_sexual_activity"])
    exposure = (row["v_explicit_exposure"] or "none").lower()
    intimacy = (row["v_intimacy_level"] or "none").lower()
    detector_score = float(row["det_max_score"] or 0.0)
    detector_labels = (row["det_labels"] or "").lower()

    detector_label_set = {
        label.strip().lower()
        for label in detector_labels.split(",")
        if label.strip()
    }
    strong_detector = (
        detector_score >= 0.50
        and bool(detector_label_set & STRONG_DETECTOR_LABELS)
    )
    soft_detector = (
        detector_score >= 0.75
        and bool(detector_label_set & SOFT_DETECTOR_LABELS)
    )

    return {
        "high_conf_explicit": visible_nudity and exposure == "clear" and confidence >= 0.70,
        "visible_nudity": visible_nudity and confidence >= 0.55,
        "partial_nudity": visible_nudity and exposure == "partial" and confidence >= 0.60,
        "sexual_activity": sexual_activity and confidence >= 0.60,
        "suggestive": intimacy == "suggestive" and confidence >= 0.55,
        "negative_guard": (
            (not visible_nudity)
            and (not sexual_activity)
            and exposure == "none"
            and intimacy == "none"
            and confidence >= 0.65
        ),
        "strong_detector": strong_detector,
        "soft_detector": soft_detector,
    }


def _decide_should_censor(
    is_strict: bool,
    has_classified_frames: bool,
    votes: list[dict],
    explicit_subtitles: bool,
) -> tuple[bool, str]:
    if not has_classified_frames:
        # No vision evidence at all. Subtitles alone only trigger a cut in
        # strict mode, because the audio mute step already handles dialogue.
        if explicit_subtitles and is_strict:
            return True, "no-frames + strict + explicit subtitle keywords"
        return False, "no-frames + no strong subtitle signal"

    visible_count = sum(v["visible_nudity"] for v in votes)
    partial_count = sum(v["partial_nudity"] for v in votes)
    sexual_count = sum(v["sexual_activity"] for v in votes)
    suggestive_count = sum(v["suggestive"] for v in votes)
    negative_count = sum(v["negative_guard"] for v in votes)
    strong_det = sum(v["strong_detector"] for v in votes)
    soft_det = sum(v["soft_detector"] for v in votes)
    high_conf_explicit = any(v["high_conf_explicit"] for v in votes)

    # Hard negative: every representative frame confidently said "nothing
    # nudity-related is visible" AND the detector saw nothing AND the
    # subtitles aren't loud about it. Trust it.
    if (
        negative_count == len(votes)
        and strong_det == 0
        and soft_det == 0
        and not explicit_subtitles
    ):
        return False, "all-frames negative + clean detector + clean subtitles"

    if high_conf_explicit:
        return True, "high-confidence explicit frame"
    if strong_det >= 1 and (visible_count >= 1 or partial_count >= 1):
        return True, "strong detector hit confirmed by vision"
    if sexual_count >= 1:
        return True, "vision detected sexual activity"

    if is_strict:
        if (
            visible_count >= 1
            or partial_count >= 1
            or strong_det >= 1
            or soft_det >= 1
            or suggestive_count >= 2
            or explicit_subtitles
        ):
            return True, "strict: any visible/partial/detector/suggestive/subtitle signal"
        return False, "strict: no signal"

    # Moderate: require multiple independent signals OR a high-confidence one.
    if visible_count >= 2:
        return True, "moderate: >=2 visible-nudity frames"
    if strong_det >= 2:
        return True, "moderate: >=2 strong detector hits"
    if partial_count >= 2 and (strong_det >= 1 or explicit_subtitles):
        return True, "moderate: partial nudity confirmed by detector or subtitles"
    return False, "moderate: signal too weak"


def determine_if_should_censor_video(video_id: int):
    scene_numbers = store.get_scene_numbers(video_id)
    if not scene_numbers:
        logging.info("No scenes to process for video %s.", video_id)
        return False

    is_strict = config.is_strict

    for scene_number in scene_numbers:
        frames = store.get_scene_frames_with_caches(video_id, scene_number)
        representative = [
            f for f in frames if f["frame_role"] and f["frame_role"] != "support"
        ]
        votes = [_frame_vote(f) for f in representative if f["v_confidence"] is not None]
        subtitles = _scene_subtitle_texts(video_id, scene_number)
        explicit_subs = _scene_has_explicit_subtitles(subtitles)

        should_censor, reason = _decide_should_censor(
            is_strict=is_strict,
            has_classified_frames=bool(votes),
            votes=votes,
            explicit_subtitles=explicit_subs,
        )
        store.upsert_decision(video_id, scene_number, should_censor, reason)

    logging.info("Scene censor decisions persisted for video %s.", video_id)
    return True
