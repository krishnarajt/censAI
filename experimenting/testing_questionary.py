import questionary


def main_menu():
    while True:
        choice = questionary.select(
            "Main Menu - Choose an option:", choices=["Option 1", "Option 2", "Exit"]
        ).ask()

        if choice == "Exit":
            break
        elif choice == "Option 1":
            submenu_1()
        elif choice == "Option 2":
            submenu_2()


def submenu_1():
    choice = questionary.select(
        "Submenu 1 - Choose an option:",
        choices=["Suboption 1.1", "Suboption 1.2", "Back"],
    ).ask()

    if choice == "Back":
        return
    elif choice == "Suboption 1.1":
        submenu_1_1()
    elif choice == "Suboption 1.2":
        submenu_1_2()


def submenu_1_1():
    choice = questionary.select(
        "Submenu 1.1 - Choose an option:",
        choices=["Final Option 1.1.1", "Final Option 1.1.2", "Back"],
    ).ask()

    if choice == "Back":
        return
    print(f"You selected {choice}")


def submenu_1_2():
    choice = questionary.select(
        "Submenu 1.2 - Choose an option:",
        choices=["Final Option 1.2.1", "Final Option 1.2.2", "Back"],
    ).ask()

    if choice == "Back":
        return
    print(f"You selected {choice}")


def submenu_2():
    choice = questionary.select(
        "Submenu 2 - Choose an option:",
        choices=["Suboption 2.1", "Suboption 2.2", "Back"],
    ).ask()

    if choice == "Back":
        return
    print(f"You selected {choice}")


if __name__ == "__main__":
    main_menu()
