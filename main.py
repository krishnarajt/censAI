"""Long-running CensAI pod service."""

from __future__ import annotations

import logging
import signal
import sys
import threading

from config.settings import Config
from db.central import CentralStore
from ui.server import UIServer
from worker import RetryWorker


def configure_logging() -> None:
    config = Config()
    logging.basicConfig(
        format=config.LOGGING_FORMAT,
        datefmt=config.LOGGING_DATE_FORMAT,
        level=logging.INFO,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)
    logging.getLogger("safetext").setLevel(logging.ERROR)


def main() -> int:
    configure_logging()
    config = Config()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Starting CensAI service")
    logger.info("=" * 60)
    logger.info(
        "LLM provider=%s vision_model=%s profanity_model=%s",
        config.llm_provider_label,
        config.vision_model,
        config.profanity_model,
    )

    central = CentralStore(config)
    try:
        central.init_db()
    except Exception as exc:  # noqa: BLE001
        logger.critical("Main DB init failed: %s", exc, exc_info=True)
        return 2

    ui_server = UIServer(config.UI_HOST, config.UI_PORT)
    ui_server.start()

    worker = RetryWorker(config)
    if config.AUTO_PROCESS_DUE:
        worker.start()

    stop_event = threading.Event()

    def _stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    logger.info("CensAI is ready.")
    try:
        stop_event.wait()
    finally:
        logger.info("Stopping CensAI service...")
        worker.stop()
        ui_server.stop()
        logger.info("CensAI stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
