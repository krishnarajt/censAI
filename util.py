import hashlib
import time
from pathlib import Path

from PIL import Image

from config.settings import Config
from enums.CensorshipStrength import CensorshipStrength


def print_welcome_message():
    print("Welcome to CensAI!")
    print(
        """

    ░▒▓██████▓▒░░▒▓████████▓▒░▒▓███████▓▒░ ░▒▓███████▓▒░░▒▓██████▓▒░░▒▓█▓▒░ 
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░ 
    ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░ 
    ░▒▓█▓▒░      ░▒▓██████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░ 
    ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░ 
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░ 
    ░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░ 

    Censorship Model for TV shows and Movies

    """
    )


def print_censorship_message():
    print()
    print(
        "-------------------------------------------------------------------------------------------"
    )
    print("Starting the censorship process.")
    print(
        "-------------------------------------------------------------------------------------------"
    )
    print()


def get_censorship_strength():
    time.sleep(0.1)
    config = Config()
    print("Enter censorship strength (1 or 2):")
    print(
        f"1. {CensorshipStrength.MODERATE} - Only explicit on-screen exposed nudity is removed."
    )
    print(
        f"2. {CensorshipStrength.STRICT} - Almost all on-screen nudity is removed. Profane dialogues are muted and their subtitles replaced by AI generated sentences with similar meaning."
    )

    choice = input("Enter your choice: ").strip()
    if choice not in ["1", "2"]:
        print("Invalid choice. Please enter 1 or 2.")
        return get_censorship_strength()
    config._censorship_strength = (
        CensorshipStrength.MODERATE if choice == "1" else CensorshipStrength.STRICT
    )


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------


def sha256_file(path) -> str:
    """SHA-256 of the raw file bytes. Used as the cache key for LLM/detector
    calls so that re-running on identical frames is free."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def phash_image(path) -> str:
    """Perceptual hash (pHash) as hex string. Used to dedupe near-identical
    frames within a scene so the vision model isn't called repeatedly on
    visually identical snapshots."""
    import imagehash  # lazy: imagehash drags in numpy/scipy

    with Image.open(path) as img:
        return str(imagehash.phash(img.convert("RGB"), hash_size=16))


def file_exists_nonempty(path) -> bool:
    p = Path(path)
    return p.is_file() and p.stat().st_size > 0
