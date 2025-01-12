# tmseegpy/gui/__init__.py

from .gui_app import TMSEEG_GUI
from .ica_handler import select_ica_components, ICAComponentSelector

def main():
    """Entry point for the GUI application"""
    import tkinter as tk
    import sys
    import traceback
    import builtins
    from tkinter import messagebox

    def show_error(error_msg):
        """Display error in a simple tkinter window"""
        temp_root = tk.Tk()
        temp_root.withdraw()
        messagebox.showerror("Error", f"Failed to start GUI:\n{error_msg}")
        temp_root.destroy()

    try:
        root = tk.Tk()
        root.title("TMS-EEG Analysis")

        # Store root in builtins for access from other modules
        builtins.GUI_MAIN_ROOT = root

        app = TMSEEG_GUI(root)
        root.mainloop()
    except Exception as e:
        error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
        show_error(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()