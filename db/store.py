"""SQLite-backed persistent store for CensAI.

Replaces the previous pandas DataFrame / pickle / CSV checkpoint system. All
LLM and detector calls are cached here so that re-running on the same media
folder is free, and only changed work is performed.

Cache keys:
  - nudity_cache:    image_sha256          (NudeNet output is deterministic)
  - vision_cache:    image_sha256 + model  (vision LLM output)
  - vision_phash:    image_phash + model -> image_sha256 (near-dupe alias)
  - profanity_cache: text_hash + model    (subtitle rewrite output)

Delete the censai.sqlite file to flush all caches.
"""

import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS videos (
    video_id   INTEGER PRIMARY KEY,
    path       TEXT    NOT NULL UNIQUE,
    name       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS subtitles (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id           INTEGER NOT NULL,
    start_ms           INTEGER NOT NULL,
    end_ms             INTEGER NOT NULL,
    text               TEXT    NOT NULL,
    text_hash          TEXT    NOT NULL,
    cleaned_text       TEXT,
    profanity_present  INTEGER NOT NULL DEFAULT 0,
    scene_number       INTEGER,
    UNIQUE (video_id, start_ms),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_subs_video_scene ON subtitles(video_id, scene_number);

CREATE TABLE IF NOT EXISTS scene_frames (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id         INTEGER NOT NULL,
    scene_number     INTEGER NOT NULL,
    snapshot_number  INTEGER NOT NULL,
    timestamp_ms     INTEGER NOT NULL,
    snapshot_path    TEXT    NOT NULL,
    image_sha256     TEXT,
    image_phash      TEXT,
    frame_role       TEXT,
    UNIQUE (video_id, scene_number, snapshot_number),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_frames_video_scene ON scene_frames(video_id, scene_number);
CREATE INDEX IF NOT EXISTS idx_frames_sha          ON scene_frames(image_sha256);

CREATE TABLE IF NOT EXISTS scene_decisions (
    video_id      INTEGER NOT NULL,
    scene_number  INTEGER NOT NULL,
    should_censor INTEGER NOT NULL,
    reason        TEXT,
    PRIMARY KEY (video_id, scene_number),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS nudity_cache (
    image_sha256    TEXT PRIMARY KEY,
    labels          TEXT,
    max_score       REAL,
    raw_json        TEXT,
    nudity_present  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS vision_cache (
    image_sha256       TEXT NOT NULL,
    model              TEXT NOT NULL,
    summary            TEXT,
    visible_nudity     INTEGER,
    explicit_exposure  TEXT,
    sexual_activity    INTEGER,
    intimacy_level     TEXT,
    confidence         REAL,
    reason             TEXT,
    raw_json           TEXT,
    PRIMARY KEY (image_sha256, model)
);

CREATE TABLE IF NOT EXISTS vision_phash_alias (
    image_phash   TEXT NOT NULL,
    model         TEXT NOT NULL,
    image_sha256  TEXT NOT NULL,
    PRIMARY KEY (image_phash, model)
);

CREATE TABLE IF NOT EXISTS profanity_cache (
    text_hash     TEXT NOT NULL,
    model         TEXT NOT NULL,
    cleaned_text  TEXT NOT NULL,
    PRIMARY KEY (text_hash, model)
);

CREATE VIEW IF NOT EXISTS vision_frame_insights AS
SELECT
    f.id AS frame_id,
    f.video_id,
    vd.name AS video_name,
    f.scene_number,
    f.snapshot_number,
    f.timestamp_ms,
    f.snapshot_path,
    f.image_sha256,
    f.image_phash,
    f.frame_role,
    n.labels AS detector_labels,
    n.max_score AS detector_max_score,
    n.nudity_present AS detector_nudity_present,
    v.model,
    v.summary,
    v.visible_nudity,
    v.explicit_exposure,
    v.sexual_activity,
    v.intimacy_level,
    v.confidence,
    v.reason,
    v.raw_json
FROM scene_frames f
JOIN videos vd ON vd.video_id = f.video_id
LEFT JOIN nudity_cache n ON n.image_sha256 = f.image_sha256
LEFT JOIN vision_cache v ON v.image_sha256 = f.image_sha256;
"""


class Store:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._conn: sqlite3.Connection | None = None
        self._db_path: Path | None = None

    # ------------------------------------------------------------------ open
    def open(self, db_path):
        with self._lock:
            if self._conn is not None:
                self._conn.close()
            self._db_path = Path(db_path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                self._db_path, check_same_thread=False, isolation_level=None
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Store not opened. Call store.open(db_path) first.")
        return self._conn

    @property
    def db_path(self) -> Path | None:
        return self._db_path

    def close(self):
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ---------------------------------------------------------------- videos
    def upsert_video(self, video_id: int, path: str, name: str) -> int:
        self.conn.execute(
            "INSERT INTO videos(video_id, path, name) VALUES (?, ?, ?) "
            "ON CONFLICT(video_id) DO UPDATE SET path=excluded.path, name=excluded.name",
            (video_id, path, name),
        )
        return video_id

    def get_video_path(self, video_id: int) -> str | None:
        row = self.conn.execute(
            "SELECT path FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        return row["path"] if row else None

    # ------------------------------------------------------------- subtitles
    def has_subtitles(self, video_id: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM subtitles WHERE video_id = ? LIMIT 1", (video_id,)
        ).fetchone()
        return row is not None

    def insert_subtitle(
        self,
        video_id: int,
        start_ms: int,
        end_ms: int,
        text: str,
        text_hash: str,
        cleaned_text: str | None,
        profanity_present: bool,
    ):
        self.conn.execute(
            "INSERT OR IGNORE INTO subtitles "
            "(video_id, start_ms, end_ms, text, text_hash, cleaned_text, profanity_present) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                video_id,
                int(start_ms),
                int(end_ms),
                text,
                text_hash,
                cleaned_text,
                1 if profanity_present else 0,
            ),
        )

    def get_subtitles(self, video_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM subtitles WHERE video_id = ? ORDER BY start_ms",
            (video_id,),
        ).fetchall()

    def get_profane_subtitle_intervals(self, video_id: int) -> list[tuple[int, int]]:
        rows = self.conn.execute(
            "SELECT start_ms, end_ms FROM subtitles "
            "WHERE video_id = ? AND profanity_present = 1 ORDER BY start_ms",
            (video_id,),
        ).fetchall()
        return [(int(r["start_ms"]), int(r["end_ms"])) for r in rows]

    def assign_subtitle_scenes(self, video_id: int, mappings):
        """mappings: iterable of (subtitle_id, scene_number)."""
        self.conn.executemany(
            "UPDATE subtitles SET scene_number = ? WHERE id = ?",
            [(int(sn), int(sid)) for sid, sn in mappings],
        )

    # ---------------------------------------------------------- scene_frames
    def count_frames(self, video_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM scene_frames WHERE video_id = ?", (video_id,)
        ).fetchone()
        return int(row["n"]) if row else 0

    def count_hashed_frames(self, video_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM scene_frames "
            "WHERE video_id = ? AND image_sha256 IS NOT NULL AND image_phash IS NOT NULL",
            (video_id,),
        ).fetchone()
        return int(row["n"]) if row else 0

    def upsert_frame(
        self,
        video_id: int,
        scene_number: int,
        snapshot_number: int,
        timestamp_ms: int,
        snapshot_path: str,
        image_sha256: str | None,
        image_phash: str | None,
    ):
        self.conn.execute(
            "INSERT INTO scene_frames "
            "(video_id, scene_number, snapshot_number, timestamp_ms, snapshot_path, image_sha256, image_phash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(video_id, scene_number, snapshot_number) DO UPDATE SET "
            "  timestamp_ms = excluded.timestamp_ms, "
            "  snapshot_path = excluded.snapshot_path, "
            "  image_sha256 = excluded.image_sha256, "
            "  image_phash = excluded.image_phash",
            (
                int(video_id),
                int(scene_number),
                int(snapshot_number),
                int(timestamp_ms),
                snapshot_path,
                image_sha256,
                image_phash,
            ),
        )

    def get_frames(self, video_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM scene_frames WHERE video_id = ? "
            "ORDER BY scene_number, snapshot_number",
            (video_id,),
        ).fetchall()

    def get_scene_frames_with_caches(self, video_id: int, scene_number: int):
        """Return frames for a scene joined with nudity & vision caches."""
        return self.conn.execute(
            """
            SELECT f.*,
                   n.labels         AS det_labels,
                   n.max_score      AS det_max_score,
                   n.nudity_present AS det_nudity_present,
                   v.summary        AS v_summary,
                   v.visible_nudity AS v_visible_nudity,
                   v.explicit_exposure AS v_explicit_exposure,
                   v.sexual_activity   AS v_sexual_activity,
                   v.intimacy_level    AS v_intimacy_level,
                   v.confidence        AS v_confidence,
                   v.reason            AS v_reason
              FROM scene_frames f
              LEFT JOIN nudity_cache n ON n.image_sha256 = f.image_sha256
              LEFT JOIN vision_cache v ON v.image_sha256 = f.image_sha256
             WHERE f.video_id = ? AND f.scene_number = ?
             ORDER BY f.snapshot_number
            """,
            (video_id, scene_number),
        ).fetchall()

    def set_frame_role(self, frame_id: int, role: str | None):
        self.conn.execute(
            "UPDATE scene_frames SET frame_role = ? WHERE id = ?",
            (role, int(frame_id)),
        )

    def clear_frame_roles(self, video_id: int):
        self.conn.execute(
            "UPDATE scene_frames SET frame_role = NULL WHERE video_id = ?", (video_id,)
        )

    def get_scene_numbers(self, video_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT DISTINCT scene_number FROM scene_frames WHERE video_id = ? "
            "ORDER BY scene_number",
            (video_id,),
        ).fetchall()
        return [int(r["scene_number"]) for r in rows]

    # ----------------------------------------------------------- nudity cache
    def get_nudity_cache(self, image_sha256: str):
        return self.conn.execute(
            "SELECT * FROM nudity_cache WHERE image_sha256 = ?", (image_sha256,)
        ).fetchone()

    def set_nudity_cache(
        self,
        image_sha256: str,
        labels: list,
        max_score: float,
        raw,
        nudity_present: bool,
    ):
        self.conn.execute(
            "INSERT INTO nudity_cache (image_sha256, labels, max_score, raw_json, nudity_present) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(image_sha256) DO UPDATE SET "
            "  labels = excluded.labels, max_score = excluded.max_score, "
            "  raw_json = excluded.raw_json, nudity_present = excluded.nudity_present",
            (
                image_sha256,
                ", ".join(labels) if isinstance(labels, (list, tuple)) else str(labels or ""),
                float(max_score or 0.0),
                json.dumps(raw, ensure_ascii=True),
                1 if nudity_present else 0,
            ),
        )

    # ----------------------------------------------------------- vision cache
    def get_vision_cache(self, image_sha256: str, model: str):
        return self.conn.execute(
            "SELECT * FROM vision_cache WHERE image_sha256 = ? AND model = ?",
            (image_sha256, model),
        ).fetchone()

    def get_vision_cache_by_phash(self, image_phash: str, model: str):
        row = self.conn.execute(
            "SELECT v.* FROM vision_phash_alias a "
            "  JOIN vision_cache v ON v.image_sha256 = a.image_sha256 AND v.model = a.model "
            " WHERE a.image_phash = ? AND a.model = ? LIMIT 1",
            (image_phash, model),
        ).fetchone()
        return row

    def set_vision_cache(
        self,
        image_sha256: str,
        model: str,
        parsed: dict,
        raw: str,
        image_phash: str | None,
    ):
        self.conn.execute(
            "INSERT INTO vision_cache "
            "(image_sha256, model, summary, visible_nudity, explicit_exposure, "
            " sexual_activity, intimacy_level, confidence, reason, raw_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(image_sha256, model) DO UPDATE SET "
            "  summary = excluded.summary, "
            "  visible_nudity = excluded.visible_nudity, "
            "  explicit_exposure = excluded.explicit_exposure, "
            "  sexual_activity = excluded.sexual_activity, "
            "  intimacy_level = excluded.intimacy_level, "
            "  confidence = excluded.confidence, "
            "  reason = excluded.reason, "
            "  raw_json = excluded.raw_json",
            (
                image_sha256,
                model,
                parsed.get("summary"),
                1 if parsed.get("visible_nudity") else 0,
                parsed.get("explicit_body_exposure", "none"),
                1 if parsed.get("sexual_activity") else 0,
                parsed.get("intimacy_level", "none"),
                float(parsed.get("confidence") or 0.0),
                parsed.get("reason_short"),
                raw,
            ),
        )
        if image_phash:
            self.conn.execute(
                "INSERT OR REPLACE INTO vision_phash_alias "
                "(image_phash, model, image_sha256) VALUES (?, ?, ?)",
                (image_phash, model, image_sha256),
            )

    # -------------------------------------------------------- profanity cache
    def get_profanity_cache(self, text_hash: str, model: str) -> str | None:
        row = self.conn.execute(
            "SELECT cleaned_text FROM profanity_cache WHERE text_hash = ? AND model = ?",
            (text_hash, model),
        ).fetchone()
        return row["cleaned_text"] if row else None

    def set_profanity_cache(self, text_hash: str, model: str, cleaned_text: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO profanity_cache (text_hash, model, cleaned_text) "
            "VALUES (?, ?, ?)",
            (text_hash, model, cleaned_text),
        )

    # ---------------------------------------------------------------- decisions
    def upsert_decision(
        self, video_id: int, scene_number: int, should_censor: bool, reason: str = ""
    ):
        self.conn.execute(
            "INSERT INTO scene_decisions (video_id, scene_number, should_censor, reason) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(video_id, scene_number) DO UPDATE SET "
            "  should_censor = excluded.should_censor, reason = excluded.reason",
            (int(video_id), int(scene_number), 1 if should_censor else 0, reason),
        )

    def get_censored_scene_numbers(self, video_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT scene_number FROM scene_decisions "
            "WHERE video_id = ? AND should_censor = 1 ORDER BY scene_number",
            (video_id,),
        ).fetchall()
        return [int(r["scene_number"]) for r in rows]

    def get_decision(self, video_id: int, scene_number: int):
        return self.conn.execute(
            "SELECT should_censor, reason FROM scene_decisions "
            "WHERE video_id = ? AND scene_number = ?",
            (video_id, scene_number),
        ).fetchone()
