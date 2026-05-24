import os
import pathlib
import time

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared for runtime images
    def load_dotenv():
        return False

from enums.CensorshipStrength import CensorshipStrength

load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _split_media_folders(value: str) -> list[str]:
    paths: list[str] = []
    for raw in value.replace(";", "\n").replace(",", "\n").splitlines():
        cleaned = raw.strip()
        if cleaned:
            paths.append(cleaned)
    return paths


def _get_database_url() -> str:
    configured = (os.environ.get("DATABASE_URL") or "").strip()
    if configured:
        return configured
    sqlite_path = (os.environ.get("CENSAI_MAIN_SQLITE_PATH") or "censai.sqlite3").strip()
    if sqlite_path.startswith("sqlite:"):
        return sqlite_path
    return f"sqlite:///{sqlite_path}"


class Config:
    _instance = None

    LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
    LOGGING_DATE_FORMAT = "%H:%M:%S"
    VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv"]
    SUBTITLE_EXTENSIONS = [".srt", ".ass", ".vtt", ".sub", ".idx"]
    NUMBER_OF_IMAGES_PER_SCENE = 7

    # Tunable runtime defaults. These remain readable from env so first boot
    # can seed the shared main DB, but runtime reads prefer central config.
    _ENV_VISION_MODEL = os.environ.get("CENSAI_VISION_MODEL", "qwen3-vl:4b")
    _ENV_PROFANITY_MODEL = os.environ.get("CENSAI_PROFANITY_MODEL", "mistral")
    _ENV_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    _ENV_USE_LLM_GATEWAY = _get_bool(
        "CENSAI_USE_LLM_GATEWAY",
        _get_bool("USE_LLM_GATEWAY", False),
    )
    _ENV_LLM_GATEWAY_URL = os.environ.get(
        "LLM_GATEWAY_URL",
        "https://llmgateway.krishnarajthadesar.in",
    )
    _ENV_LLM_GATEWAY_API_KEY = os.environ.get("LLM_GATEWAY_API_KEY", "")
    _ENV_LLM_GATEWAY_CHAT_PATH = os.environ.get("LLM_GATEWAY_CHAT_PATH", "/api/chat")
    _ENV_LLM_GATEWAY_DEFAULT_MODEL = os.environ.get("LLM_GATEWAY_DEFAULT_MODEL", "gemma4:27b")
    _ENV_LLM_GATEWAY_VISION_MODEL = os.environ.get(
        "LLM_GATEWAY_VISION_MODEL",
        _ENV_LLM_GATEWAY_DEFAULT_MODEL,
    )
    _ENV_LLM_GATEWAY_PROFANITY_MODEL = os.environ.get(
        "LLM_GATEWAY_PROFANITY_MODEL",
        _ENV_LLM_GATEWAY_DEFAULT_MODEL,
    )
    _ENV_LLM_GATEWAY_TIMEOUT_SECONDS = _get_float("LLM_GATEWAY_TIMEOUT_SECONDS", 300.0)
    _ENV_LLM_GATEWAY_MAX_PARALLEL_CALLS = max(
        1,
        _get_int("CENSAI_LLM_GATEWAY_MAX_PARALLEL_CALLS", 3),
    )
    _ENV_OLLAMA_MAX_PARALLEL_CALLS = max(
        1,
        _get_int("CENSAI_OLLAMA_MAX_PARALLEL_CALLS", 1),
    )

    # Main tracking/config DB. If Postgres creds are not provided, fall back to
    # a root-level SQLite DB. The local per-folder SQLite cache remains the
    # source of truth for expensive detector/model outputs.
    DATABASE_URL = _get_database_url()
    DB_SCHEMA = os.environ.get(
        "DB_SCHEMA",
        "" if DATABASE_URL.startswith("sqlite") else "censai",
    )
    CENTRAL_DB_ENABLED = _get_bool("CENSAI_CENTRAL_DB_ENABLED", True)

    # Long-running pod/UI defaults.
    _ENV_MEDIA_FOLDER = os.environ.get("CENSAI_MEDIA_FOLDER", "/media")
    _ENV_MEDIA_FOLDERS = os.environ.get("CENSAI_MEDIA_FOLDERS", "").strip()
    _ENV_UI_HOST = os.environ.get("CENSAI_UI_HOST", "0.0.0.0")
    _ENV_UI_PORT = _get_int("CENSAI_UI_PORT", 8000)
    _ENV_AUTO_PROCESS_DUE = _get_bool("CENSAI_AUTO_PROCESS_DUE", True)
    _ENV_WORKER_POLL_SECONDS = _get_int("CENSAI_WORKER_POLL_SECONDS", 60)
    _ENV_RETRY_DELAY_HOURS = _get_float("CENSAI_RETRY_DELAY_HOURS", 24.0)
    _ENV_DEFAULT_CENSORSHIP_STRENGTH = os.environ.get(
        "CENSAI_DEFAULT_CENSORSHIP_STRENGTH",
        "moderate",
    ).strip().lower()

    # NudeNet thresholds. Tuned to reduce false positives while still
    # catching anything genuinely explicit.
    NUDENET_STRONG_THRESHOLD = 0.40
    NUDENET_SOFT_THRESHOLD = 0.75

    # Vision sampling per scene. Phash dedup means actual LLM calls are
    # fewer than this for visually static scenes.
    MAX_VISION_FRAMES_PER_SCENE = 5

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._media_folder_path = pathlib.Path()
            self._temp_folder_path = None
            self._censorship_strength = None
            self.muted_audio_root_path = None
            self.video_and_subtitle_files = {}
            self.video_to_id = {}
            self.id_to_video = {}
            self.db_path = None
            self._runtime_config_cache = {}
            self._runtime_config_loaded_at = 0.0
            self._runtime_config_ttl_seconds = 2.0

    def invalidate_runtime_config_cache(self) -> None:
        self._runtime_config_cache = {}
        self._runtime_config_loaded_at = 0.0

    def _runtime_configs(self) -> dict[str, str]:
        if not self.CENTRAL_DB_ENABLED or not self.DATABASE_URL:
            return {}
        now = time.monotonic()
        if (
            self._runtime_config_cache
            and now - self._runtime_config_loaded_at < self._runtime_config_ttl_seconds
        ):
            return self._runtime_config_cache
        try:
            from db.central import CentralStore

            central = CentralStore(self)
            if not central.enabled:
                self._runtime_config_cache = {}
            else:
                self._runtime_config_cache = {
                    item["key"]: item["value"]
                    for item in central.list_configs()
                }
            self._runtime_config_loaded_at = now
        except Exception:
            if not self._runtime_config_cache:
                self._runtime_config_cache = {}
        return self._runtime_config_cache

    def _runtime_value(self, key: str, default: str) -> str:
        value = self._runtime_configs().get(key)
        if value is None:
            return default
        return str(value).strip()

    def _runtime_bool(self, key: str, default: bool) -> bool:
        value = self._runtime_configs().get(key)
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _runtime_int(self, key: str, default: int, minimum: int | None = None) -> int:
        value = self._runtime_configs().get(key)
        try:
            parsed = int(str(value).strip()) if value is not None else default
        except ValueError:
            parsed = default
        return max(minimum, parsed) if minimum is not None else parsed

    def _runtime_float(self, key: str, default: float) -> float:
        value = self._runtime_configs().get(key)
        try:
            return float(str(value).strip()) if value is not None else default
        except ValueError:
            return default

    def central_config_defaults(self) -> dict[str, str]:
        return {
            "ui_host": self._ENV_UI_HOST,
            "ui_port": str(self._ENV_UI_PORT),
            "auto_process_due": str(self._ENV_AUTO_PROCESS_DUE).lower(),
            "worker_poll_seconds": str(self._ENV_WORKER_POLL_SECONDS),
            "retry_delay_hours": str(self._ENV_RETRY_DELAY_HOURS),
            "censorship_strength": self._ENV_DEFAULT_CENSORSHIP_STRENGTH,
            "use_llm_gateway": str(self._ENV_USE_LLM_GATEWAY).lower(),
            "ollama_host": self._ENV_OLLAMA_HOST,
            "vision_model": self._ENV_VISION_MODEL,
            "profanity_model": self._ENV_PROFANITY_MODEL,
            "ollama_max_parallel_calls": str(self._ENV_OLLAMA_MAX_PARALLEL_CALLS),
            "llm_gateway_url": self._ENV_LLM_GATEWAY_URL,
            "llm_gateway_chat_path": self._ENV_LLM_GATEWAY_CHAT_PATH,
            "llm_gateway_default_model": self._ENV_LLM_GATEWAY_DEFAULT_MODEL,
            "llm_gateway_vision_model": self._ENV_LLM_GATEWAY_VISION_MODEL,
            "llm_gateway_profanity_model": self._ENV_LLM_GATEWAY_PROFANITY_MODEL,
            "llm_gateway_timeout_seconds": str(self._ENV_LLM_GATEWAY_TIMEOUT_SECONDS),
            "llm_gateway_max_parallel_calls": str(self._ENV_LLM_GATEWAY_MAX_PARALLEL_CALLS),
        }

    @property
    def media_folder_path(self):
        return self._media_folder_path

    @media_folder_path.setter
    def media_folder_path(self, path):
        if not isinstance(path, str):
            raise ValueError("Path must be a string.")
        path = pathlib.Path(path)
        if not path.exists() or not path.is_dir():
            raise ValueError("The path does not exist or is not a directory.")

        self._media_folder_path = path
        self._temp_folder_path = path / "temp"
        self.muted_audio_root_path = self._temp_folder_path / "muted_audio"
        self._temp_folder_path.mkdir(parents=True, exist_ok=True)
        self.muted_audio_root_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self._temp_folder_path / "censai.sqlite"

    @property
    def temp_path(self):
        return self._temp_folder_path

    @temp_path.setter
    def temp_path(self, path):
        if not isinstance(path, pathlib.Path):
            raise ValueError("Path must be a pathlib.Path object.")
        self._temp_folder_path = path

    @property
    def censorship_strength(self):
        return self._censorship_strength

    @censorship_strength.setter
    def censorship_strength(self, strength):
        if strength is not None and not isinstance(strength, CensorshipStrength):
            raise ValueError("Censorship strength must be a CensorshipStrength Enum value.")
        self._censorship_strength = strength

    @property
    def is_strict(self) -> bool:
        return self._censorship_strength == CensorshipStrength.STRICT

    @property
    def ollama_host(self) -> str:
        host = self._runtime_value("ollama_host", self._ENV_OLLAMA_HOST)
        if host == "0.0.0.0" or host.startswith("0.0.0.0"):
            host = "http://localhost:11434"
        elif not host.startswith("http"):
            host = f"http://{host}"
        return host

    @property
    def vision_model(self) -> str:
        return self.LLM_GATEWAY_VISION_MODEL if self.USE_LLM_GATEWAY else self.VISION_MODEL

    @property
    def profanity_model(self) -> str:
        return (
            self.LLM_GATEWAY_PROFANITY_MODEL
            if self.USE_LLM_GATEWAY
            else self.PROFANITY_MODEL
        )

    @property
    def llm_provider_label(self) -> str:
        return "LLM Gateway" if self.USE_LLM_GATEWAY else "Ollama"

    @property
    def VISION_MODEL(self) -> str:
        return self._runtime_value("vision_model", self._ENV_VISION_MODEL)

    @property
    def PROFANITY_MODEL(self) -> str:
        return self._runtime_value("profanity_model", self._ENV_PROFANITY_MODEL)

    @property
    def USE_LLM_GATEWAY(self) -> bool:
        return self._runtime_bool("use_llm_gateway", self._ENV_USE_LLM_GATEWAY)

    @property
    def LLM_GATEWAY_URL(self) -> str:
        return self._runtime_value("llm_gateway_url", self._ENV_LLM_GATEWAY_URL)

    @property
    def LLM_GATEWAY_API_KEY(self) -> str:
        return self._ENV_LLM_GATEWAY_API_KEY

    @property
    def LLM_GATEWAY_CHAT_PATH(self) -> str:
        return self._runtime_value("llm_gateway_chat_path", self._ENV_LLM_GATEWAY_CHAT_PATH)

    @property
    def LLM_GATEWAY_DEFAULT_MODEL(self) -> str:
        return self._runtime_value(
            "llm_gateway_default_model",
            self._ENV_LLM_GATEWAY_DEFAULT_MODEL,
        )

    @property
    def LLM_GATEWAY_VISION_MODEL(self) -> str:
        return self._runtime_value(
            "llm_gateway_vision_model",
            self._runtime_value(
                "llm_gateway_default_model",
                self._ENV_LLM_GATEWAY_VISION_MODEL,
            ),
        )

    @property
    def LLM_GATEWAY_PROFANITY_MODEL(self) -> str:
        return self._runtime_value(
            "llm_gateway_profanity_model",
            self._runtime_value(
                "llm_gateway_default_model",
                self._ENV_LLM_GATEWAY_PROFANITY_MODEL,
            ),
        )

    @property
    def LLM_GATEWAY_TIMEOUT_SECONDS(self) -> float:
        return self._runtime_float(
            "llm_gateway_timeout_seconds",
            self._ENV_LLM_GATEWAY_TIMEOUT_SECONDS,
        )

    @property
    def LLM_GATEWAY_MAX_PARALLEL_CALLS(self) -> int:
        return self._runtime_int(
            "llm_gateway_max_parallel_calls",
            self._ENV_LLM_GATEWAY_MAX_PARALLEL_CALLS,
            minimum=1,
        )

    @property
    def OLLAMA_MAX_PARALLEL_CALLS(self) -> int:
        return self._runtime_int(
            "ollama_max_parallel_calls",
            self._ENV_OLLAMA_MAX_PARALLEL_CALLS,
            minimum=1,
        )

    @property
    def MEDIA_FOLDER(self) -> str:
        folders = self.MEDIA_FOLDERS
        if folders:
            return folders[0]
        return self._ENV_MEDIA_FOLDER

    @property
    def MEDIA_FOLDERS(self) -> list[str]:
        configured = self._ENV_MEDIA_FOLDERS or self._ENV_MEDIA_FOLDER
        folders = _split_media_folders(configured)
        if folders:
            return folders
        return [self._ENV_MEDIA_FOLDER]

    @property
    def UI_HOST(self) -> str:
        return self._runtime_value("ui_host", self._ENV_UI_HOST)

    @property
    def UI_PORT(self) -> int:
        return self._runtime_int("ui_port", self._ENV_UI_PORT, minimum=1)

    @property
    def AUTO_PROCESS_DUE(self) -> bool:
        return self._runtime_bool("auto_process_due", self._ENV_AUTO_PROCESS_DUE)

    @property
    def WORKER_POLL_SECONDS(self) -> int:
        return self._runtime_int(
            "worker_poll_seconds",
            self._ENV_WORKER_POLL_SECONDS,
            minimum=1,
        )

    @property
    def RETRY_DELAY_HOURS(self) -> float:
        return self._runtime_float("retry_delay_hours", self._ENV_RETRY_DELAY_HOURS)

    @property
    def DEFAULT_CENSORSHIP_STRENGTH(self) -> str:
        value = self._runtime_value(
            "censorship_strength",
            self._ENV_DEFAULT_CENSORSHIP_STRENGTH,
        ).lower()
        return value if value in {"moderate", "strict"} else self._ENV_DEFAULT_CENSORSHIP_STRENGTH

    def cleanup_temp_folder(self, video_id):
        temp_folder = self._temp_folder_path / str(video_id)
        if temp_folder.exists() and temp_folder.is_dir():
            for item in temp_folder.iterdir():
                if item.is_file():
                    item.unlink()
            temp_folder.rmdir()
