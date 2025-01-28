import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
import threading
import queue
import multiprocessing
from .ica_selector_gui.ica_selector_react import ICAComponentSelector_React, ICAComponentSelectorContinuous_React
import matplotlib

matplotlib.use('Qt5Agg')  # Set this before importing pyplot


class CLIICASelector:
    def __init__(self):
        self.result_queue = queue.Queue()
        self.selection_complete = threading.Event()
        self.qt_app = None
        self._initialize_qt()

    def _initialize_qt(self):
        """Initialize Qt in the main thread"""
        if threading.current_thread() is threading.main_thread():
            self.qt_app = QApplication.instance()
            if self.qt_app is None:
                self.qt_app = QApplication(sys.argv)
                print("Created new Qt application instance in main thread")
        else:
            print("Warning: Attempting to initialize Qt outside main thread")

    def select_components(self, ica_instance, inst, component_scores=None):
        """Run ICA selection ensuring Qt app exists"""
        try:
            # Ensure we're in the main thread for Qt operations
            if threading.current_thread() is not threading.main_thread():
                raise RuntimeError("ICA selection must be run in the main thread")

            # Initialize Qt if needed
            if self.qt_app is None:
                self._initialize_qt()

            import mne
            print("Creating selector...")

            # Create appropriate selector based on data type
            if isinstance(inst, mne.io.Raw):
                selector = ICAComponentSelectorContinuous_React(None)
            else:
                selector = ICAComponentSelector_React(None)

            print("Selector created")

            # Callback for when selection is complete
            def selection_callback(components):
                print(f"Selection callback received components: {components}")
                self.result_queue.put(components)
                self.selection_complete.set()

            # Show selector window
            print("Showing selector window...")
            selector.select_components(
                ica_instance=ica_instance,
                raw=inst if isinstance(inst, mne.io.Raw) else None,
                epochs=inst if isinstance(inst, mne.Epochs) else None,
                title="Select ICA Components",
                callback=selection_callback,
                component_scores=component_scores
            )

            # Create a timer to keep Qt event loop alive
            timer = QTimer()
            timer.timeout.connect(lambda: None)
            timer.start(100)

            # Start Qt event loop if not already running
            print("Starting Qt event loop...")
            if not self.qt_app.activeWindow():
                self.qt_app.exec()

            # Wait for selection to complete
            print("Waiting for selection to complete...")
            self.selection_complete.wait()

            # Get selected components
            try:
                selected_components = self.result_queue.get_nowait()
                print(f"Retrieved selected components: {selected_components}")
                return selected_components
            except queue.Empty:
                print("No components were selected (queue empty)")
                return []

        except Exception as e:
            print(f"Error in component selection: {str(e)}")
            import traceback
            traceback.print_exc()
            return []


def get_cli_ica_callback():
    """Create a callback function for CLI ICA selection that ensures main thread execution"""
    selector = CLIICASelector()

    def callback(ica_instance, inst, component_scores=None):
        if threading.current_thread() is threading.main_thread():
            return selector.select_components(ica_instance, inst, component_scores)
        else:
            # If not in main thread, use a queue to communicate with main thread
            result_queue = queue.Queue()

            def run_in_main_thread():
                try:
                    result = selector.select_components(ica_instance, inst, component_scores)
                    result_queue.put(result)
                except Exception as e:
                    print(f"Error in ICA selection: {str(e)}")
                    result_queue.put([])

            # Create and start main thread task
            import asyncio
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(run_in_main_thread)

            # Wait for result
            return result_queue.get()

    return callback