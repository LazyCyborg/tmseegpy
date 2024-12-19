from .gui_app import TMSEEG_GUI

def main():
    """Entry point for the GUI application"""
    import tkinter as tk
    import sys
    import traceback
    from tkinter import messagebox
    
    def show_error(error_msg):
        """Display error in a simple tkinter window"""
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"Failed to start GUI:\n{error_msg}")
        root.destroy()
    
    try:
        root = tk.Tk()
        root.title("TMS-EEG Analysis")
        app = TMSEEG_GUI(root)
        root.mainloop()
    except Exception as e:
        error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
        show_error(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()