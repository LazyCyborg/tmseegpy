# main.py
#!/usr/bin/env python3
import argparse
import tkinter as tk
from .gui import TMSEEG_GUI

def main():
    parser = argparse.ArgumentParser(description='Launch TMS-EEG Analysis GUI')
    parser.add_argument('--width', type=int, default=1000,
                       help='Initial window width (default: 1000)')
    parser.add_argument('--height', type=int, default=1000,
                       help='Initial window height (default: 1000)')

    args = parser.parse_args()

    try:
        root = tk.Tk()
        root.title("TMS-EEG Analysis")
        root.geometry(f"{args.width}x{args.height}")
        app = TMSEEG_GUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error initializing GUI: {e}")
        return 1

    return 0

if __name__ == "__main__":
    main()