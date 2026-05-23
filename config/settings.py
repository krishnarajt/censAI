import pathlib

from enums.CensorshipStrength import CensorshipStrength


class Config:
    _instance = None

    LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
    LOGGING_DATE_FORMAT = "%H:%M:%S"
    VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv"]
    SUBTITLE_EXTENSIONS = [".srt", ".ass", ".vtt", ".sub", ".idx"]
    NUMBER_OF_IMAGES_PER_SCENE = 7

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
            self.subtitles_processed_video_ids = []
            self.video_to_id = {}
            self.id_to_video = {}

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

    def cleanup_temp_folder(self, video_id):
        temp_folder = self._temp_folder_path / str(video_id)
        if temp_folder.exists() and temp_folder.is_dir():
            for item in temp_folder.iterdir():
                if item.is_file():
                    item.unlink()
            temp_folder.rmdir()

    def save_checkpoint(self):
        from data.dataframe_manager import ScenesDataFrameManager

        ScenesDataFrameManager().save_checkpoint()
