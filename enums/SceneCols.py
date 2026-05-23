from enum import Enum
import numpy as np

class SceneCols(Enum):
    TIMESTAMP = ("timestamp", "float64")
    SUBTITLE_START_TIME = ("subtitle_start_time", "float64")
    SUBTITLE_END_TIME = ("subtitle_end_time", "float64")
    VIDEO_ID = ("video_id", "int32")
    SCENE_NUMBER = ("scene_number", "int32")
    SCENE_SNAPSHOT_NUMBER = ("scene_snapshot_number", "int32")
    SCENE_SNAPSHOT_PATH = ("scene_snapshot_path", "string")
    SUBTITLE = ("subtitle", "string")
    CLEANED_SUBTITLE = ("cleaned_subtitle", "string")
    SNAPSHOT_DESC = ("snapshot_desc", "string")
    PROFANITY_PRESENT = ("profanity_present", "bool")
    NUDITY_PRESENT = ("nudity_present", "bool")
    SHOULD_CENSOR = ("should_censor", "bool")

    def __init__(self, col_name, dtype):
        self._col_name = col_name
        self._dtype = dtype

    @property
    def name_str(self):
        return self._col_name

    @property
    def dtype(self):
        return self._dtype

    def __str__(self):
        return self._col_name

    def __repr__(self):
        return self._col_name
