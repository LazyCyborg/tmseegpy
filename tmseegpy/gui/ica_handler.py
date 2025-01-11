import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import mne
import numpy as np
from typing import List, Optional, Callable
import queue
import threading


class ICAComponentSelector:
    def __init__(self, parent: tk.Tk):
        # Keep existing initialization
        self.parent = parent
        self.selected_components = set()
        self.completion_callback = None
        self._cleanup_called = False
        self.tk_vars = []
        self._window = None
        self._fig = None
        self._canvas = None
        self._toolbar = None
        self._selection_listbox = None

        # Add properties for component info
        self._info_window = None
        self._info_canvas = None
        self._current_component = None
        self._ica_instance = None
        self._epochs = None

    def select_components(self,
                          ica_instance: mne.preprocessing.ICA,
                          epochs: mne.Epochs,
                          title: str = "Select ICA Components",
                          callback: Optional[Callable] = None) -> None:
        """Create and show the ICA selection window."""
        self.completion_callback = callback
        self._ica_instance = ica_instance
        self._epochs = epochs

        # Create main window
        self._window = tk.Toplevel(self.parent)
        self._window.title(title)
        self._window.geometry("1400x800")  # Made wider for additional info

        # Create main frame
        main_frame = ttk.Frame(self._window, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left frame for component overview
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self._fig = plt.figure(figsize=(8, 6))
        self._canvas = FigureCanvasTkAgg(self._fig, master=left_frame)
        self._toolbar = NavigationToolbar2Tk(self._canvas, left_frame)
        self._toolbar.pack(side=tk.TOP, fill=tk.X)

        # Plot components
        self._plot_components(ica_instance, epochs)

        # Create right frame for controls and info
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Component list frame
        list_frame = ttk.LabelFrame(right_frame, text="Selected Components", padding="5")
        list_frame.pack(fill=tk.X, pady=5)

        # Selection listbox with scrollbar
        self._selection_listbox = tk.Listbox(list_frame, height=8, width=25)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical",
                                  command=self._selection_listbox.yview)
        self._selection_listbox.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._selection_listbox.config(yscrollcommand=scrollbar.set)

        # Info frame
        info_frame = ttk.LabelFrame(right_frame, text="Component Information", padding="5")
        info_frame.pack(fill=tk.X, pady=5)

        # Add component property buttons
        ttk.Button(info_frame, text="View Topography",
                   command=self._show_topography).pack(fill=tk.X, pady=2)
        ttk.Button(info_frame, text="View Time Course",
                   command=self._show_time_course).pack(fill=tk.X, pady=2)
        ttk.Button(info_frame, text="View Properties",
                   command=self._show_properties).pack(fill=tk.X, pady=2)

        # Instructions
        instruction_frame = ttk.LabelFrame(right_frame, text="Instructions", padding="5")
        instruction_frame.pack(fill=tk.X, pady=5)
        instructions = (
            "1. Click on components to select/deselect\n"
            "2. Selected components will be removed\n"
            "3. Use buttons to view detailed information\n"
            "4. Click Done when finished"
        )
        ttk.Label(instruction_frame, text=instructions, wraplength=250).pack(pady=5)

        # Control buttons
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Done",
                   command=self._finish_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                   command=self._cancel_selection).pack(side=tk.LEFT, padx=5)

        # Connect events
        self._canvas.mpl_connect('button_press_event', self._on_click)
        self._canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._window.protocol("WM_DELETE_WINDOW", self._cancel_selection)

    def _show_topography(self):
        """Show topography of currently selected component."""
        if self._current_component is not None:
            fig = plt.figure()
            self._ica_instance.plot_components(
                picks=[self._current_component],
                ch_type='eeg',
                inst=self._epochs,
                show=False
            )
            plt.show()

    def _show_time_course(self):
        """Show time course of currently selected component."""
        if self._current_component is not None:
            fig = plt.figure()
            self._ica_instance.plot_sources(
                self._epochs,
                picks=[self._current_component],
                show=False
            )
            plt.show()

    def _show_properties(self):
        """Show detailed properties of currently selected component."""
        if self._current_component is not None:
            self._ica_instance.plot_properties(
                self._epochs,
                picks=[self._current_component],
                psd_args={'fmax': 80},
                show=True
            )

    def _on_click(self, event) -> None:
        """Handle component selection clicks with enhanced feedback."""
        if event.inaxes is None or not hasattr(event.inaxes, 'get_title'):
            return

        title = event.inaxes.get_title()
        if not title.startswith('IC'):
            return

        comp_idx = int(title[2:])
        self._current_component = comp_idx  # Store current component

        if comp_idx in self.selected_components:
            self.selected_components.remove(comp_idx)
            event.inaxes.patch.set_facecolor('white')
        else:
            self.selected_components.add(comp_idx)
            event.inaxes.patch.set_facecolor('lightgreen')

        event.inaxes.patch.set_alpha(0.3)
        self._canvas.draw()

        # Update selection list
        if self._selection_listbox is not None:
            self._selection_listbox.delete(0, tk.END)
            for comp in sorted(self.selected_components):
                self._selection_listbox.insert(tk.END, f"Component {comp}")

    def _cleanup(self, components=None):
        """Clean up all resources safely."""
        if self._cleanup_called:
            return
        self._cleanup_called = True

        def do_cleanup():
            try:
                # Hide window first
                if self._window:
                    self._window.withdraw()

                # Clean up matplotlib resources
                if self._fig:
                    plt.close(self._fig)
                    self._fig = None

                # Clean up Tkinter widgets
                if self._canvas:
                    self._canvas.get_tk_widget().destroy()
                    self._canvas = None

                if self._toolbar:
                    self._toolbar.destroy()
                    self._toolbar = None

                # Clean up variables
                for var in self.tk_vars:
                    try:
                        var.set(None)  # Clear the variable
                    except tk.TclError:
                        pass
                self.tk_vars.clear()

                # Destroy window
                if self._window:
                    self._window.destroy()
                    self._window = None

                # Call callback after cleanup
                if self.completion_callback and components is not None:
                    self.completion_callback(components)

            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
                # Ensure callback is called even if cleanup fails
                if self.completion_callback and components is not None:
                    self.completion_callback(components)

        # Schedule cleanup on main thread
        if self.parent:
            self.parent.after_idle(do_cleanup)

    def _finish_selection(self) -> None:
        """Complete selection with chosen components."""
        components = sorted(list(self.selected_components))
        self._cleanup(components)

    def _cancel_selection(self) -> None:
        """Cancel selection."""
        self._cleanup([])


def select_ica_components(ica_instance: mne.preprocessing.ICA,
                        epochs: mne.Epochs,
                        title: str = "Select ICA Components",
                        root: Optional[tk.Tk] = None,
                        callback: Optional[Callable] = None) -> None:
    """Thread-safe ICA component selection with callback support."""
    if root is None:
        raise RuntimeError("No main Tk root provided")

    if callback is None:
        result_queue = queue.Queue()
        processing_complete = threading.Event()

        def internal_callback(components):
            try:
                result_queue.put(components)
                processing_complete.set()
            except Exception as e:
                print(f"Error in completion handling: {str(e)}")
                result_queue.put([])
                processing_complete.set()

        callback = internal_callback

    def show_selector():
        try:
            # Create and show the selector
            selector = ICAComponentSelector(root)
            selector.select_components(ica_instance, epochs, title, callback=callback)
        except Exception as e:
            print(f"Error creating selector: {str(e)}")
            callback([])

    # Ensure we're running on the main thread
    if threading.current_thread() is threading.main_thread():
        show_selector()
    else:
        root.after(0, show_selector)

    # If no callback was provided, wait for and return the result
    if callback == internal_callback:
        processing_complete.wait(timeout=300)  # 5-minute timeout
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return []