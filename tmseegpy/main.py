# main.py
#!/usr/bin/env python3
import argparse
import tkinter as tk
import builtins

from .gui.gui_app import TMSEEG_GUI  # or wherever TMSEEG_GUI is defined

def main():
    parser = argparse.ArgumentParser(description='Launch TMS-EEG Analysis GUI')
    parser.add_argument('--width', type=int, default=1000,
                       help='Initial window width (default: 1000)')
    parser.add_argument('--height', type=int, default=1000,
                       help='Initial window height (default: 1000)')

    args = parser.parse_args()

    # Create the Tk root
    root = tk.Tk()
    root.title("TMS-EEG Analysis")
    root.geometry(f"{args.width}x{args.height}")

    # **Important**: Let schedule_on_main_thread() find the main root
    builtins.GUI_MAIN_ROOT = root

    # Start the GUI
    app = TMSEEG_GUI(root)
    root.mainloop()


# If a user calls `python -m tmseegpy.main`, this block is used:
if __name__ == "__main__":
    main()
