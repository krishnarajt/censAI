from enum import Enum
from enums.SceneCols import SceneCols

class SceneImageCols(Enum):
    SCENE_NUMBER = SceneCols.SCENE_NUMBER.value
    VIDEO_ID =  SceneCols.VIDEO_ID.value
    SCENE_SNAPSHOT_NUMBER = SceneCols.SCENE_SNAPSHOT_NUMBER.value
    TIMESTAMP = SceneCols.TIMESTAMP.value
    SCENE_SNAPSHOT_PATH = SceneCols.SCENE_SNAPSHOT_PATH.value

    def __init__(self, col_name, dtype):
        self._col_name = col_name
        self._dtype = dtype

    def __str__(self):
        return self._col_name

    @property
    def name_str(self):
        return self._col_name

    @property
    def dtype(self):
        return self._dtype

    @classmethod
    def as_dtype_dict(cls):
        return {str(col): col.dtype for col in cls}
