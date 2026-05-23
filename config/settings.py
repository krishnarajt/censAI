import os
import pathlib

from enums.CensorshipStrength import CensorshipStrength


class Config:
    _instance = None

    LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
    LOGGING_DATE_FORMAT = "%H:%M:%S"
    VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv"]
    SUBTITLE_EXTENSIONS = [".srt", ".ass", ".vtt", ".sub", ".idx"]
    NUMBER_OF_IMAGES_PER_SCENE = 7

    # Tunable models -- override via environment variable.
    # Defaults are chosen for a 4060Ti 8GB / 32GB DDR5 box.
    VISION_MODEL = os.environ.get("CENSAI_VISION_MODEL", "qwen3-vl:4b")
    PROFANITY_MODEL = os.environ.get("CENSAI_PROFANITY_MODEL", "mistral")
    _OLLAMA_HOST_RAW = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

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
        host = self._OLLAMA_HOST_RAW
        if host == "0.0.0.0" or host.startswith("0.0.0.0"):
            host = "http://localhost:11434"
        elif not host.startswith("http"):
            host = f"http://{host}"
        return host

    def cleanup_temp_folder(self, video_id):
        temp_folder = self._temp_folder_path / str(video_id)
        if temp_folder.exists() and temp_folder.is_dir():
            for item in temp_folder.iterdir():
                if item.is_file():
                    item.unlink()
            temp_folder.rmdir()
