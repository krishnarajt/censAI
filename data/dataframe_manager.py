import pandas as pd

from config.settings import Config
from enums.SceneCols import SceneCols
from enums.SceneImageCols import SceneImageCols


class ScenesDataFrameManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self.config = Config()
            self._initialized = True
            self._init_empty_frames()

    def _init_empty_frames(self):
        self.all_scenes_df = pd.DataFrame(
            columns=[col.name_str for col in SceneCols],
        ).astype({col.name_str: col.dtype for col in SceneCols})
        self.scene_images_df = pd.DataFrame(
            columns=[col.name_str for col in SceneImageCols],
        ).astype({col.name_str: col.dtype for col in SceneImageCols})

    def init_scene_images_df(self):
        self.scene_images_df = pd.DataFrame(
            columns=[col.name_str for col in SceneImageCols],
        ).astype({col.name_str: col.dtype for col in SceneImageCols})

    def save_checkpoint(self):
        if self.config._temp_folder_path is None:
            return
        self.config._temp_folder_path.mkdir(parents=True, exist_ok=True)
        self.all_scenes_df.to_csv(self.config._temp_folder_path / "scenes.csv", index=False)
        self.all_scenes_df.to_pickle(self.config._temp_folder_path / "scenes.pkl")

    def load_checkpoint(self):
        if self.config._temp_folder_path is None:
            self._init_empty_frames()
            return

        checkpoint_path = self.config._temp_folder_path / "scenes.pkl"
        if checkpoint_path.exists():
            self.all_scenes_df = pd.read_pickle(checkpoint_path)
            print("Loaded scenes DataFrame from existing file.")
        else:
            self._init_empty_frames()
            print("Initialized new scenes DataFrame.")

    def print_stats(self, video_id: int):
        if self.all_scenes_df.empty:
            print("No scenes to display.")
            return

        video_rows = self.all_scenes_df[self.all_scenes_df["video_id"] == video_id]
        subtitle_rows = video_rows[video_rows["subtitle"].notna()]
        censored_rows = video_rows[video_rows["should_censor"] == True]
        changed_subtitles = subtitle_rows[subtitle_rows["cleaned_subtitle"].notna()]

        total_time_cut = 0.0
        for scene_number in censored_rows["scene_number"].dropna().unique().tolist():
            scene_df = censored_rows[censored_rows["scene_number"] == scene_number]
            if not scene_df.empty:
                start_time = scene_df["timestamp"].min()
                end_time = scene_df["timestamp"].max()
                total_time_cut += (end_time - start_time) / 1000.0

        print(f"Total Time Cut: {total_time_cut:.2f} seconds")
        print(f"Total Rows in Video: {len(video_rows)}")
        print(f"Total Subtitle Rows: {len(subtitle_rows)}")
        print(f"Total Censored Rows: {len(censored_rows)}")
        print(f"Total Changed Subtitles: {len(changed_subtitles)}")
