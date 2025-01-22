#!/usr/bin/env python3
import argparse
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt6.QtWidgets import QApplication


def main():
    parser = argparse.ArgumentParser(description='Launch TMS-EEG Analysis GUI')
    parser.add_argument('--width', type=int, default=1000,
                        help='Initial window width (default: 1000)')
    parser.add_argument('--height', type=int, default=1000,
                        help='Initial window height (default: 1000)')
    parser.add_argument('--style', type=str, default='dark',
                        choices=['dark', 'light'],
                        help='GUI style theme (default: dark)')

    args = parser.parse_args()

    # Create Qt application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for consistent look across platforms


    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()