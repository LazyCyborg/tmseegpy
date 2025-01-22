# ica_process.py
from tmseegpy.cli_ica_selector import CLIICASelector
import os
import sys


def run_ica_selection(ica_obj, inst, component_scores):
    """Run ICA selection using the existing CLIICASelector"""
    try:
        # Print debugging information
        print("Starting ICA selection...")
        print(f"Current working directory: {os.getcwd()}")
        print(f"PYTHONPATH: {sys.path}")
        print(f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM')}")
        print(f"DISPLAY: {os.environ.get('DISPLAY')}")

        # Try setting platform to cocoa for macOS
        if sys.platform == 'darwin':
            os.environ['QT_QPA_PLATFORM'] = 'cocoa'

        selector = CLIICASelector()

        # Add debug print before selection
        print("Created selector, starting component selection...")

        result = selector.select_components(ica_obj, inst, component_scores)

        # Add debug print after selection
        print("Component selection completed")

        return result
    except Exception as e:
        print(f"Error in ICA selection: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def process_target(conn, ica_obj, inst, scores):
    """Function that runs in separate process"""
    try:
        print("Starting process_target...")
        result = run_ica_selection(ica_obj, inst, scores)
        print(f"Selection result: {result}")
        conn.send(result)
    except Exception as e:
        print(f"Error in process_target: {str(e)}")
        import traceback
        traceback.print_exc()
        conn.send([])
    finally:
        conn.close()