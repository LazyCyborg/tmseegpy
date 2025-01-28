from PyQt6.QtCore import QObject, pyqtSignal, QTimer, Qt
from PyQt6.QtWidgets import QApplication
import queue
import threading
import matplotlib
from tmseegpy.ica_selector_gui.ica_selector_react import ICAComponentSelector_React, ICAComponentSelectorContinuous_React

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


class ICAWindowManager(QObject):
    """Manager for ICA windows that ensures main thread execution"""
    show_window_signal = pyqtSignal(object, object, object)
    selection_complete = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.result_queue = queue.Queue()
        self.selection_done = threading.Event()
        self.show_window_signal.connect(self._show_window_main_thread)

    def _show_window_main_thread(self, ica_instance, inst, component_scores):
        """Create and show ICA window in main thread"""
        try:
            import mne
            print("Creating selector in main thread...")

            # Create appropriate selector
            if isinstance(inst, mne.io.Raw):
                selector = ICAComponentSelectorContinuous_React(None)
            else:
                selector = ICAComponentSelector_React(None)

            def selection_callback(components):
                print(f"Selection callback received components: {components}")
                self.result_queue.put(components)
                self.selection_done.set()
                self.selection_complete.emit(components)

            # Show selector window
            selector.select_components(
                ica_instance=ica_instance,
                raw=inst if isinstance(inst, mne.io.Raw) else None,
                epochs=inst if isinstance(inst, mne.Epochs) else None,
                title="Select ICA Components",
                callback=selection_callback,
                component_scores=component_scores
            )

        except Exception as e:
            print(f"Error showing ICA window: {str(e)}")
            import traceback
            traceback.print_exc()
            self.result_queue.put([])
            self.selection_done.set()


class CLIICASelector:
    def __init__(self):
        # Get the QApplication instance from the main thread
        self.qt_app = QApplication.instance()
        if self.qt_app is None:
            raise RuntimeError("QApplication must be created in main thread before CLIICASelector")

        # Create window manager
        self.window_manager = ICAWindowManager()

    def select_components(self, ica_instance, inst, component_scores=None):
        """Run ICA selection ensuring main thread execution"""
        try:
            print("Starting ICA component selection...")

            # Emit signal to show window in main thread
            self.window_manager.show_window_signal.emit(ica_instance, inst, component_scores)

            # Keep Qt event loop running
            timer = QTimer()
            timer.timeout.connect(lambda: None)
            timer.start(100)

            # Wait for selection to complete
            print("Waiting for selection to complete...")
            self.window_manager.selection_done.wait()

            # Get results
            try:
                selected_components = self.window_manager.result_queue.get_nowait()
                print(f"Retrieved selected components: {selected_components}")
                return selected_components
            except queue.Empty:
                print("No components were selected (queue empty)")
                return []

        except Exception as e:
            print(f"Error in ICA selection: {str(e)}")
            import traceback
            traceback.print_exc()
            return []


def get_cli_ica_callback():
    """Create a callback function for CLI ICA selection"""
    window_manager = None

    def callback(ica_instance, inst, component_scores=None):
        nonlocal window_manager

        try:
            # Ensure we're in the main thread
            if threading.current_thread() is not threading.main_thread():
                result_queue = queue.Queue()

                def run_in_main():
                    try:
                        selector = CLIICASelector()
                        result = selector.select_components(ica_instance, inst, component_scores)
                        result_queue.put(result)
                    except Exception as e:
                        print(f"Error in main thread ICA selection: {str(e)}")
                        result_queue.put([])

                # Schedule execution in main thread
                import asyncio
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(run_in_main)

                return result_queue.get()
            else:
                # Create selector in main thread
                selector = CLIICASelector()
                return selector.select_components(ica_instance, inst, component_scores)

        except Exception as e:
            print(f"Error in ICA callback: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    return callback