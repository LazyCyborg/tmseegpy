#!/usr/bin/env python
# gui_launcher.py

import os
import sys
from pathlib import Path
import streamlit.web.cli as stcli


def find_main_script():
    """Find the path to the main.py script in the main_gui folder."""
    try:
        # First try to get it from the installed package
        import tmseegpy
        package_path = Path(tmseegpy.__file__).parent

        # Look in main_gui directory
        main_script = package_path / "main_gui" / "main.py"

        if main_script.exists():
            return str(main_script)

        # If not found, try alternative locations
        possible_locations = [
            Path(__file__).parent / "main_gui" / "main.py",
            Path(__file__).parent.parent / "main_gui" / "main.py",
            Path(__file__).parent / "main.py",
            Path(__file__).parent.parent / "main.py",
            package_path / "app.py",
        ]

        for location in possible_locations:
            if location.exists():
                return str(location)

        # Last resort, try to find the main.py file recursively
        for root, dirs, files in os.walk(package_path):
            if "main.py" in files and "main_gui" in root:
                return str(Path(root) / "main.py")

        raise FileNotFoundError("Could not find main.py script in main_gui folder")
    except Exception as e:
        print(f"Error finding main script: {e}")
        sys.exit(1)


def run_streamlit_app():
    """Run the Streamlit app directly."""
    main_script = find_main_script()
    print(f"Launching TMSeegPy Streamlit app from: {main_script}")

    # Use Streamlit's CLI to run the app
    sys.argv = ["streamlit", "run", main_script, "--browser.serverAddress=localhost", "--server.headless=true"]
    stcli.main()


def main():
    """Main entry point for the GUI launcher."""
    # Simply launch the app without any shortcut functionality
    run_streamlit_app()


if __name__ == "__main__":
    main()