"""Background processor for main DB retry/queue rows."""

from __future__ import annotations

import logging
import threading
import time

from config.settings import Config
from db.central import CentralStore

logger = logging.getLogger(__name__)


class RetryWorker:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.central = CentralStore(self.config)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        if not self.central.enabled:
            logger.info("Retry worker disabled because the main DB is not configured.")
            return
        requeued = self.central.requeue_stale_processing()
        if requeued:
            logger.warning(
                "Recovered %d stale processing video(s) from a previous run and re-queued them.",
                requeued,
            )
        self._thread = threading.Thread(target=self._run, name="censai-retry-worker", daemon=True)
        self._thread.start()
        logger.info("CensAI retry worker started.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _sleep(self) -> None:
        self._stop.wait(max(5, int(self.config.WORKER_POLL_SECONDS)))

    def _run(self) -> None:
        try:
            from app import process_video_path, scan_media_folder
        except Exception as exc:  # noqa: BLE001
            logger.critical(
                "Retry worker could not import the processing pipeline: %s",
                exc,
                exc_info=True,
            )
            return

        while not self._stop.is_set():
            try:
                scan_media_folder(self.config.MEDIA_FOLDER)
                job = self.central.claim_next_due_video()
                if job is None:
                    self._sleep()
                    continue

                logger.info("Processing queued video: %s", job["path"])
                try:
                    process_video_path(
                        job["path"],
                        media_folder_path=job.get("media_folder") or self.config.MEDIA_FOLDER,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.central.mark_failed(job["path"], exc)
                    logger.error("Queued video failed: %s", exc, exc_info=True)
                time.sleep(1)
            except Exception as exc:  # noqa: BLE001
                logger.error("Retry worker loop failed: %s", exc, exc_info=True)
                self._sleep()
