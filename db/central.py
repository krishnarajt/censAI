"""Main tracking/config store for CensAI.

The local per-media SQLite database remains responsible for detector/LLM
caches. This module tracks cross-run video job status, retry timing, and small
developer-editable config values in Postgres or root-level SQLite.
"""

from __future__ import annotations

import hashlib
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    or_,
    select,
)
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from config.settings import Config

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://") and "+" not in url.split("://", 1)[0]:
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


def video_key_for_path(path: str | Path) -> str:
    resolved = str(Path(path).expanduser().resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def is_rate_limit_error(exc: BaseException | str) -> bool:
    text = str(exc).lower()
    return any(
        needle in text
        for needle in (
            "429",
            "rate limit",
            "rate_limit",
            "too many requests",
            "resource exhausted",
            "quota",
        )
    )


class Base(DeclarativeBase):
    pass


def _table_args() -> dict[str, str]:
    config = Config()
    if config.DB_SCHEMA and not config.DATABASE_URL.startswith("sqlite"):
        return {"schema": config.DB_SCHEMA}
    return {}


class CentralVideo(Base):
    __tablename__ = "censai_videos"
    __table_args__ = (
        UniqueConstraint("video_key", name="uq_censai_videos_video_key"),
        _table_args(),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    media_folder: Mapped[str] = mapped_column(Text, nullable=False, default="")
    subtitle_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    local_video_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="detected")
    censored: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
    )


class CentralConfig(Base):
    __tablename__ = "censai_configs"
    __table_args__ = (_table_args(),)

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
    )


class CentralStore:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.enabled = bool(self.config.CENTRAL_DB_ENABLED and self.config.DATABASE_URL)
        self.engine = None
        self.SessionLocal = None
        if not self.enabled:
            return

        database_url = normalize_database_url(self.config.DATABASE_URL)
        engine_kwargs = {"pool_pre_ping": True, "future": True}
        if database_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            self._ensure_sqlite_parent(database_url)
        self.engine = create_engine(database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )

    @property
    def backend_label(self) -> str:
        if not self.enabled or self.engine is None:
            return "disabled"
        return self.engine.dialect.name

    @staticmethod
    def _ensure_sqlite_parent(database_url: str) -> None:
        url = make_url(database_url)
        if not url.database or url.database == ":memory:":
            return
        Path(url.database).expanduser().parent.mkdir(parents=True, exist_ok=True)

    def init_db(self) -> None:
        if not self.enabled or self.engine is None:
            logger.info("Main DB disabled; video retry/config tracking is local-only.")
            return
        Base.metadata.create_all(bind=self.engine)
        self.seed_default_configs()

    @contextmanager
    def session(self) -> Iterator[Session]:
        if not self.enabled or self.SessionLocal is None:
            raise RuntimeError("Central DB is not enabled.")
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def seed_default_configs(self) -> None:
        defaults = self.config.central_config_defaults()
        with self.session() as db:
            existing_keys = {
                key
                for key, in db.execute(select(CentralConfig.key))
            }
            existing_generic_vision = db.get(CentralConfig, "vision_model")
            existing_generic_profanity = db.get(CentralConfig, "profanity_model")
            for key, value in defaults.items():
                if key not in existing_keys:
                    db.add(CentralConfig(key=key, value=value))
                    existing_keys.add(key)
            # Backfill gateway-specific keys from older generic rows so current
            # effective defaults stay stable after upgrading.
            if "llm_gateway_vision_model" not in existing_keys and existing_generic_vision:
                db.add(
                    CentralConfig(
                        key="llm_gateway_vision_model",
                        value=existing_generic_vision.value,
                    )
                )
                existing_keys.add("llm_gateway_vision_model")
            if (
                "llm_gateway_profanity_model" not in existing_keys
                and existing_generic_profanity
            ):
                db.add(
                    CentralConfig(
                        key="llm_gateway_profanity_model",
                        value=existing_generic_profanity.value,
                    )
                )
                existing_keys.add("llm_gateway_profanity_model")

    def upsert_detected_video(
        self,
        path: str | Path,
        media_folder: str | Path,
        local_video_id: int | None = None,
        subtitle_path: str | Path | None = None,
    ) -> None:
        if not self.enabled:
            return
        path_obj = Path(path)
        key = video_key_for_path(path_obj)
        with self.session() as db:
            row = db.execute(
                select(CentralVideo).where(CentralVideo.video_key == key)
            ).scalar_one_or_none()
            subtitle = str(subtitle_path) if subtitle_path is not None else None
            if row is None:
                db.add(
                    CentralVideo(
                        video_key=key,
                        path=str(path_obj),
                        name=path_obj.name,
                        media_folder=str(media_folder),
                        subtitle_path=subtitle,
                        local_video_id=local_video_id,
                        status="detected",
                    )
                )
                return
            row.path = str(path_obj)
            row.name = path_obj.name
            row.media_folder = str(media_folder)
            row.subtitle_path = subtitle
            row.local_video_id = local_video_id
            if row.status in {"missing"}:
                row.status = "detected"

    def mark_started(self, path: str | Path) -> None:
        if not self.enabled:
            return
        key = video_key_for_path(path)
        with self.session() as db:
            row = db.execute(
                select(CentralVideo).where(CentralVideo.video_key == key)
            ).scalar_one_or_none()
            if row is None:
                return
            row.status = "processing"
            row.censored = False
            row.attempts = int(row.attempts or 0) + 1
            row.last_started_at = utcnow()
            row.last_error = None
            row.next_retry_at = None

    def mark_completed(self, path: str | Path) -> None:
        if not self.enabled:
            return
        key = video_key_for_path(path)
        with self.session() as db:
            row = db.execute(
                select(CentralVideo).where(CentralVideo.video_key == key)
            ).scalar_one_or_none()
            if row is None:
                return
            row.status = "censored"
            row.censored = True
            row.last_finished_at = utcnow()
            row.last_error = None
            row.next_retry_at = None

    def mark_failed(self, path: str | Path, exc: BaseException | str) -> None:
        if not self.enabled:
            return
        key = video_key_for_path(path)
        rate_limited = is_rate_limit_error(exc)
        retry_delay_hours = self.config.RETRY_DELAY_HOURS
        try:
            retry_delay_hours = float(
                self.get_config_value("retry_delay_hours", str(retry_delay_hours))
                or retry_delay_hours
            )
        except ValueError:
            retry_delay_hours = self.config.RETRY_DELAY_HOURS
        retry_at = (
            utcnow() + timedelta(hours=retry_delay_hours)
            if rate_limited
            else None
        )
        with self.session() as db:
            row = db.execute(
                select(CentralVideo).where(CentralVideo.video_key == key)
            ).scalar_one_or_none()
            if row is None:
                return
            row.status = "rate_limited" if rate_limited else "failed"
            row.censored = False
            row.last_error = str(exc)[:4000]
            row.next_retry_at = retry_at
            row.last_finished_at = utcnow()

    def queue_video(self, video_id: int) -> bool:
        if not self.enabled:
            return False
        with self.session() as db:
            row = db.get(CentralVideo, video_id)
            if row is None:
                return False
            row.status = "queued"
            row.censored = False
            row.next_retry_at = None
            row.last_error = None
            return True

    def queue_videos(self, video_ids: list[int]) -> int:
        if not self.enabled:
            return 0
        queued = 0
        with self.session() as db:
            rows = db.execute(
                select(CentralVideo).where(CentralVideo.id.in_(video_ids))
            ).scalars()
            for row in rows:
                row.status = "queued"
                row.censored = False
                row.next_retry_at = None
                row.last_error = None
                queued += 1
        return queued

    def queue_folder(self, folder_path: str | Path) -> int:
        if not self.enabled:
            return 0
        folder = str(Path(folder_path).expanduser().resolve())
        prefix = folder.rstrip(os.sep) + os.sep
        queued = 0
        with self.session() as db:
            rows = db.execute(select(CentralVideo)).scalars()
            for row in rows:
                video_path = str(Path(row.path).expanduser().resolve())
                if video_path == folder or video_path.startswith(prefix):
                    row.status = "queued"
                    row.censored = False
                    row.next_retry_at = None
                    row.last_error = None
                queued += 1
        return queued

    def requeue_stale_processing(self) -> int:
        if not self.enabled:
            return 0
        requeued = 0
        with self.session() as db:
            rows = db.execute(
                select(CentralVideo).where(CentralVideo.status == "processing")
            ).scalars()
            for row in rows:
                row.status = "queued"
                row.censored = False
                row.next_retry_at = None
                row.last_error = None
                requeued += 1
        return requeued

    def claim_next_due_video(self) -> dict | None:
        if not self.enabled:
            return None
        now = utcnow()
        with self.session() as db:
            row = db.execute(
                select(CentralVideo)
                .where(
                    CentralVideo.status.in_(("queued", "rate_limited")),
                    or_(
                        CentralVideo.next_retry_at.is_(None),
                        CentralVideo.next_retry_at <= now,
                    ),
                )
                .order_by(CentralVideo.next_retry_at.asc().nullsfirst(), CentralVideo.id.asc())
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            row.status = "processing"
            row.censored = False
            row.last_error = None
            row.next_retry_at = None
            return self.serialize_video(row)

    def list_videos(self, limit: int = 5000) -> list[dict]:
        if not self.enabled:
            return []
        with self.session() as db:
            rows = db.execute(
                select(CentralVideo)
                .order_by(CentralVideo.updated_at.desc(), CentralVideo.id.desc())
                .limit(limit)
            ).scalars()
            return [self.serialize_video(row) for row in rows]

    def video_stats(self) -> dict:
        videos = self.list_videos(limit=10000)
        statuses: dict[str, int] = {}
        for video in videos:
            status = str(video["status"])
            statuses[status] = statuses.get(status, 0) + 1
        return {
            "total": len(videos),
            "statuses": statuses,
            "with_subtitles": sum(1 for video in videos if video["subtitle_path"]),
            "queued_or_processing": sum(
                1
                for video in videos
                if video["status"] in {"queued", "processing", "rate_limited"}
            ),
        }

    def list_folders(self) -> list[dict]:
        videos = self.list_videos(limit=10000)
        folders: dict[str, dict] = {}
        for video in videos:
            media_folder = Path(video["media_folder"] or self.config.MEDIA_FOLDER).expanduser()
            parent = Path(video["path"]).expanduser().parent
            try:
                relative_parent = parent.resolve().relative_to(media_folder.resolve())
                ancestors = [media_folder.resolve()]
                cursor = media_folder.resolve()
                for part in relative_parent.parts:
                    cursor = cursor / part
                    ancestors.append(cursor)
            except ValueError:
                ancestors = [parent.resolve()]

            for folder in ancestors:
                key = str(folder)
                item = folders.setdefault(
                    key,
                    {
                        "path": key,
                        "label": self._folder_label(key, str(media_folder)),
                        "total": 0,
                        "statuses": {},
                    },
                )
                item["total"] += 1
                status = video["status"]
                item["statuses"][status] = item["statuses"].get(status, 0) + 1

        return sorted(folders.values(), key=lambda item: item["label"].lower())

    @staticmethod
    def _folder_label(folder: str, media_folder: str) -> str:
        try:
            rel = Path(folder).resolve().relative_to(Path(media_folder).resolve())
        except ValueError:
            return folder
        rel_text = str(rel)
        return "Media root" if rel_text == "." else rel_text

    def list_configs(self) -> list[dict]:
        if not self.enabled:
            return []
        with self.session() as db:
            rows = db.execute(select(CentralConfig).order_by(CentralConfig.key)).scalars()
            return [
                {
                    "key": row.key,
                    "value": row.value,
                    "updated_at": self._iso(row.updated_at),
                }
                for row in rows
            ]

    def set_config(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        cleaned_key = key.strip()
        if not cleaned_key:
            raise ValueError("Config key is required.")
        with self.session() as db:
            row = db.get(CentralConfig, cleaned_key)
            if row is None:
                db.add(CentralConfig(key=cleaned_key, value=value))
            else:
                row.value = value

    def get_config_value(self, key: str, default: str | None = None) -> str | None:
        if not self.enabled:
            return default
        with self.session() as db:
            row = db.get(CentralConfig, key)
            return row.value if row is not None else default

    @staticmethod
    def _iso(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    def serialize_video(self, row: CentralVideo) -> dict:
        return {
            "id": row.id,
            "video_key": row.video_key,
            "path": row.path,
            "name": row.name,
            "media_folder": row.media_folder,
            "subtitle_path": row.subtitle_path,
            "local_video_id": row.local_video_id,
            "folder_path": str(Path(row.path).parent),
            "relative_path": self._relative_video_path(row.path, row.media_folder),
            "file_exists": Path(row.path).exists(),
            "has_subtitle": bool(row.subtitle_path),
            "status": row.status,
            "censored": bool(row.censored),
            "attempts": int(row.attempts or 0),
            "last_error": row.last_error,
            "next_retry_at": self._iso(row.next_retry_at),
            "last_started_at": self._iso(row.last_started_at),
            "last_finished_at": self._iso(row.last_finished_at),
            "created_at": self._iso(row.created_at),
            "updated_at": self._iso(row.updated_at),
        }

    @staticmethod
    def _relative_video_path(path: str, media_folder: str) -> str:
        try:
            return str(Path(path).resolve().relative_to(Path(media_folder).resolve()))
        except ValueError:
            return Path(path).name


@event.listens_for(Base.metadata, "before_create")
def _ensure_schema(target, connection, **kw):
    config = Config()
    if (
        connection.dialect.name == "postgresql"
        and config.DB_SCHEMA
        and config.DB_SCHEMA != "public"
    ):
        connection.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{config.DB_SCHEMA}"')
