import curses
import time

menu = {
    "Main Menu": ["Option 1", "Option 2", "Exit"],
    "Option 1": ["Suboption 1.1", "Suboption 1.2", "Back"],
    "Suboption 1.1": ["Final Option 1.1.1", "Final Option 1.1.2", "Back"],
    "Suboption 1.2": ["Final Option 1.2.1", "Final Option 1.2.2", "Back"],
    "Option 2": ["Suboption 2.1", "Suboption 2.2", "Back"],
}


def menu_navigation(stdscr, current_menu):
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    k = 0
    selected = 0

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, f"Menu: {current_menu}", curses.A_BOLD)

        for idx, item in enumerate(menu[current_menu]):
            if idx == selected:
                stdscr.addstr(idx + 2, 2, f"> {item}", curses.A_REVERSE)
            else:
                stdscr.addstr(idx + 2, 2, item)

        stdscr.refresh()
        k = stdscr.getch()

        if k == curses.KEY_UP and selected > 0:
            selected -= 1
        elif k == curses.KEY_DOWN and selected < len(menu[current_menu]) - 1:
            selected += 1
        elif k == 10:  # Enter key
            choice = menu[current_menu][selected]

            if choice == "Back":
                return
            elif choice == "Exit":
                return
            elif choice in menu:
                menu_navigation(stdscr, choice)
            else:
                stdscr.addstr(
                    len(menu[current_menu]) + 3, 2, f"Selected: {choice}", curses.A_BOLD
                )
                stdscr.refresh()
                stdscr.getch()
                return


def main(stdscr):
    menu_navigation(stdscr, "Main Menu")


def progress_bar(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(100)

    height, width = stdscr.getmaxyx()
    progress_width = width - 10
    for i in range(101):
        stdscr.clear()
        stdscr.addstr(2, 2, f"Progress: {i}%")
        stdscr.addstr(
            4,
            2,
            "["
            + "#" * (i * progress_width // 100)
            + " " * ((100 - i) * progress_width // 100)
            + "]",
        )
        stdscr.refresh()
        time.sleep(0.05)


if __name__ == "__main__":
    curses.wrapper(main)
    curses.wrapper(progress_bar)
