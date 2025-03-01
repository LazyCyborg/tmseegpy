#!/usr/bin/env python
# gui_launcher.py

import os
import sys
import subprocess
import platform
import streamlit.web.cli as stcli
from pathlib import Path


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


def get_icon_path():
    """Get the path to the appropriate icon file based on the operating system."""
    try:
        # Determine which icon extension to use based on the platform
        system = platform.system()

        if system == "Darwin":  # macOS
            icon_ext = ".icns"
        elif system == "Windows":
            icon_ext = ".ico"
        else:  # Linux and others
            # Try .ico first, then fallback to .png if available
            icon_ext = ".ico"

            # Try to find the icon in various locations
        import tmseegpy
        package_path = Path(tmseegpy.__file__).parent
        project_root = package_path.parent  # The master tmseegpy directory

        # Base icon name without extension
        icon_base = "tmseegpy"

        # Look in these directories for the icons
        icon_dirs = [
            project_root / "icons",
            package_path / "icons",
            project_root / "tmseegpy" / "icons",
            package_path / "main_gui" / "icons",
            project_root / "assets",
            project_root,
            package_path
        ]

        # First try with the platform-specific extension
        for icon_dir in icon_dirs:
            icon_path = icon_dir / f"{icon_base}{icon_ext}"
            if icon_path.exists():
                return str(icon_path)

        # If not found with the preferred extension, try alternatives
        alt_exts = [".ico", ".icns", ".png"]
        for alt_ext in alt_exts:
            if alt_ext == icon_ext:
                continue  # Skip the one we already tried

            for icon_dir in icon_dirs:
                icon_path = icon_dir / f"{icon_base}{alt_ext}"
                if icon_path.exists():
                    print(f"Using {alt_ext} icon as fallback")
                    return str(icon_path)

        # Icon not found - return None
        print("Warning: No icon file found")
        return None

    except Exception as e:
        print(f"Warning: Could not find icon: {e}")
        return None


def run_streamlit_app():
    """Run the Streamlit app directly."""
    main_script = find_main_script()
    print(f"Launching TMSeegPy Streamlit app from: {main_script}")

    # Use Streamlit's CLI to run the app
    sys.argv = ["streamlit", "run", main_script, "--browser.serverAddress=localhost", "--server.headless=true"]
    stcli.main()


def create_shortcut(desktop=True):
    """Create a platform-specific shortcut to launch the GUI."""
    script_path = os.path.abspath(__file__)
    icon_path = get_icon_path()
    system = platform.system()

    if desktop:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    else:
        desktop_path = os.path.expanduser("~")

    if system == "Windows":
        try:
            import winshell
            from win32com.client import Dispatch

            shortcut_path = os.path.join(desktop_path, "TMSeegPy.lnk")

            with winshell.shortcut(shortcut_path) as shortcut:
                shortcut.path = sys.executable
                shortcut.arguments = f"-m tmseegpy.gui_launcher"
                shortcut.description = "TMSeegPy TMS-EEG Processing GUI"
                shortcut.working_directory = os.path.dirname(script_path)
                if icon_path and icon_path.lower().endswith('.ico'):
                    shortcut.icon_location = (icon_path, 0)

            print(f"Shortcut created at: {shortcut_path}")
            return True

        except ImportError:
            print("Could not create Windows shortcut. The winshell and pywin32 packages may be missing.")
            print("Try installing them with: pip install winshell pywin32")
            return False

    elif system == "Darwin":  # macOS
        launcher_path = os.path.join(desktop_path, "TMSeegPy.command")

        # Create the launcher script
        with open(launcher_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"{sys.executable} -m tmseegpy.gui_launcher\n")

        # Make it executable
        os.chmod(launcher_path, 0o755)

        # For macOS, we can try to create an application bundle for a nicer experience
        try:
            # Only try this if we have an .icns file
            if icon_path and icon_path.lower().endswith('.icns'):
                app_path = os.path.join(desktop_path, "TMSeegPy.app")
                os.makedirs(os.path.join(app_path, "Contents", "MacOS"), exist_ok=True)
                os.makedirs(os.path.join(app_path, "Contents", "Resources"), exist_ok=True)

                # Copy the icon
                icon_dest = os.path.join(app_path, "Contents", "Resources", "tmseegpy.icns")
                import shutil
                shutil.copy2(icon_path, icon_dest)

                # Create Info.plist
                with open(os.path.join(app_path, "Contents", "Info.plist"), "w") as f:
                    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                    f.write(
                        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n')
                    f.write('<plist version="1.0">\n')
                    f.write('<dict>\n')
                    f.write('    <key>CFBundleExecutable</key>\n')
                    f.write('    <string>TMSeegPy</string>\n')
                    f.write('    <key>CFBundleIconFile</key>\n')
                    f.write('    <string>tmseegpy.icns</string>\n')
                    f.write('    <key>CFBundleIdentifier</key>\n')
                    f.write('    <string>com.tmseegpy.app</string>\n')
                    f.write('    <key>CFBundleName</key>\n')
                    f.write('    <string>TMSeegPy</string>\n')
                    f.write('    <key>CFBundlePackageType</key>\n')
                    f.write('    <string>APPL</string>\n')
                    f.write('</dict>\n')
                    f.write('</plist>\n')

                # Create the launcher script
                with open(os.path.join(app_path, "Contents", "MacOS", "TMSeegPy"), "w") as f:
                    f.write("#!/bin/bash\n")
                    f.write(f"{sys.executable} -m tmseegpy.gui_launcher\n")

                # Make it executable
                os.chmod(os.path.join(app_path, "Contents", "MacOS", "TMSeegPy"), 0o755)

                print(f"Created application bundle at: {app_path}")
        except Exception as e:
            print(f"Could not create application bundle: {e}")

        print(f"Launcher script created at: {launcher_path}")
        return True

    elif system == "Linux":
        desktop_file_path = os.path.join(desktop_path, "tmseegpy.desktop")

        with open(desktop_file_path, "w") as f:
            f.write("[Desktop Entry]\n")
            f.write("Type=Application\n")
            f.write("Name=TMSeegPy\n")
            f.write("Comment=TMS-EEG Processing GUI\n")
            f.write(f"Exec={sys.executable} -m tmseegpy.gui_launcher\n")
            f.write("Terminal=false\n")
            if icon_path:
                # Linux can use .ico, .png or other formats
                f.write(f"Icon={icon_path}\n")
            f.write("Categories=Science;MedicalSoftware;\n")

        # Make it executable
        os.chmod(desktop_file_path, 0o755)
        print(f"Desktop entry created at: {desktop_file_path}")
        return True

    return False


def main():
    """Main entry point for the GUI launcher."""
    # Check if we're being asked to create a shortcut
    if len(sys.argv) > 1 and sys.argv[1] == "--create-shortcut":
        desktop = True
        if len(sys.argv) > 2 and sys.argv[2] == "--no-desktop":
            desktop = False
        create_shortcut(desktop=desktop)
        return

    # Otherwise launch the app
    run_streamlit_app()


if __name__ == "__main__":
    main()