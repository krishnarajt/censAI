import time

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
        "-------------------------------------------------------------------------------------------")
    print("Starting the censorship process.")
    print(
        "-------------------------------------------------------------------------------------------")
    print()

def get_censorship_strength():
    time.sleep(0.1)
    config = Config()
    print("Enter censorship strength (1 or 2):")
    print(f"1. {CensorshipStrength.MODERATE} - Only Explicit on screen exposed nudity is removed.  ")
    print(
        f"2. {CensorshipStrength.STRICT} - Almost all on screen nudity is removed. If any story is present in said scenes, an AI generated summary is provided on a black screen later. Profane dialogues are muted and their subtitles replaced by AI generated sentences with similar meaning. "
    )

    choice = input("Enter your choice: ").strip()
    if choice not in ['1', '2']:
        print("Invalid choice. Please enter 1 or 2.")
        return get_censorship_strength()
    config._censorship_strength = CensorshipStrength.MODERATE if choice == '1' else CensorshipStrength.STRICT
