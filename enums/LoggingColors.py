from enum import Enum


class LoggingColors(Enum):
    BLUE_INFO = '\033[94m'  # Blue
    YELLOW_WARNING = '\033[93m'  # Yellow
    RED_ERROR = '\033[91m'  # Red
    GREEN = "\033[92m"
    RESET = '\033[0m'  # Reset to default color



