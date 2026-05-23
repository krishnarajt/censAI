import logging
import time

# Custom formatter class to align log level and file details
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Format timestamp
        time_str = f"[{self.formatTime(record, '%H:%M:%S')}]"

        # Align log level to the right with a fixed width
        level = f"{record.levelname:<6}"

        # Main message (supports multi-line)
        message = record.getMessage()

        # Right-aligned file and line number
        file_info = f"{record.filename}:{record.lineno}".rjust(30)

        return f"{time_str} {level} {message}\n{file_info}"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # We handle formatting in CustomFormatter
    datefmt="%H:%M:%S"
)

# Apply custom formatter
logger = logging.getLogger()

for handler in logger.handlers:
    handler.setFormatter(CustomFormatter())

# Example function to simulate logging output
def process_video(file_name):
    logger.info(f"Extracting speech segments from reference '{file_name}'...")
    time.sleep(1)

    logger.info("Checking video for subtitles stream...")
    time.sleep(1)

    logger.info("Detected encoding: ASCII")
    time.sleep(1)

    logger.info("...success!")
    time.sleep(1)

    logger.info("...done")

# Example usage
process_video("S:\\Shows\\Arrow.S02.1080p.Bluray.x265-HiQVE.mkv")
