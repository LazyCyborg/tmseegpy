from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
import sys
from .ica_selector_gui.ica_selector import ICAComponentSelector, ICAComponentSelectorContinuous
import threading
import queue


class CLIICASelector:
    def __init__(self):
        self.result_queue = queue.Queue()
        self.selection_complete = threading.Event()
        self.qt_app = None

    def select_components(self, ica_instance, inst, component_scores=None):
        """Run ICA selection ensuring Qt app exists"""
        # Initialize Qt application if needed
        self.qt_app = QApplication.instance()
        print("CLIICASelector: Starting component selection")
        print(f"Qt application instance: {QApplication.instance()}")
        if self.qt_app is None:
            self.qt_app = QApplication(sys.argv)

        # Create appropriate selector based on data type
        import mne

        print("Creating selector...")
        if isinstance(inst, mne.io.Raw):
            selector = ICAComponentSelectorContinuous(None)
        else:
            selector = ICAComponentSelector(None)
        print("Selector created")
        # Callback for when selection is complete
        def selection_callback(components):
            self.result_queue.put(components)
            self.selection_complete.set()

        # Show selector window
        print("Calling selector.select_components...")
        selector.select_components(
            ica_instance=ica_instance,
            raw=inst if isinstance(inst, mne.io.Raw) else None,
            epochs=inst if isinstance(inst, mne.Epochs) else None,
            title="Select ICA Components",
            callback=selection_callback,
            component_scores=component_scores
        )
        print("select_components called")

        # Create a timer to keep Qt event loop alive
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        timer.start(100)

        # Start Qt event loop if not already running
        if not self.qt_app.activeWindow():
            self.qt_app.exec()

        # Wait for selection to complete
        self.selection_complete.wait()

        # Get selected components
        try:
            selected_components = self.result_queue.get_nowait()
        except queue.Empty:
            selected_components = []

        return selected_components


def get_cli_ica_callback():
    """Create a callback function for CLI ICA selection"""
    selector = CLIICASelector()

    def callback(ica_instance, inst, component_scores=None):
        return selector.select_components(ica_instance, inst, component_scores)

    return callback