# ica_handler.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib

matplotlib.use('TkAgg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import mne
import numpy as np
from typing import List, Optional, Callable, Dict, Tuple
import queue
import threading


class ICAComponentSelector:
    def __init__(self, parent: tk.Tk):
        """Initialize ICA component selector.

        Args:
            parent: Parent Tk window
        """
        # Initialize basic attributes
        self.parent = parent
        self.selected_components = set()
        self.completion_callback = None
        self._cleanup_called = False
        self.tk_vars = []

        # Initialize GUI elements as None
        self._window = None
        self._fig = None
        self._canvas = None
        self._toolbar = None
        self._tree = None

        # Initialize ICA-related attributes
        self._ica_instance = None
        self._epochs = None
        self._component_scores = None
        self._component_labels = None
        self.sources_frame = None
        self.components_frame = None

        # Initialize sources_canvas and sources_fig as None
        self.sources_canvas = None
        self.sources_fig = None
        self.showing_sources = False

    def select_components(self,
                          ica_instance: mne.preprocessing.ICA,
                          epochs: mne.Epochs,
                          title: str = "Select ICA Components",
                          callback: Optional[Callable] = None,
                          component_scores: Optional[dict] = None,
                          component_labels: Optional[dict] = None) -> None:
        """Create and show the ICA selection window.

        Args:
            ica_instance: Fitted ICA instance
            epochs: MNE Epochs object
            title: Window title
            callback: Callback function for component selection
            component_scores: Dictionary of component scores
            component_labels: Dictionary of component labels
        """
        # Store parameters
        self.completion_callback = callback
        self._ica_instance = ica_instance
        self._epochs = epochs
        self._component_scores = component_scores
        self._component_labels = component_labels

        # Create main window with proper modality
        self._window = tk.Toplevel(self.parent)
        self._window.title(title)
        self._window.geometry("1400x800")
        self._window.transient(self.parent)  # Set window to be on top of parent

        # Create main frame
        main_frame = ttk.Frame(self._window, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left frame for plots
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        plt.ion()  # Turn on interactive mode
        self._fig = plt.figure(figsize=(10, 6))
        self._canvas = FigureCanvasTkAgg(self._fig, master=left_frame)

        # Connect events before creating plot
        self._canvas.mpl_connect('button_press_event', self._on_click)

        # Add toolbar
        self._toolbar = NavigationToolbar2Tk(self._canvas, left_frame)
        self._toolbar.pack(side=tk.TOP, fill=tk.X)

        # Add sources plot button
        self._toggle_button = ttk.Button(left_frame, text="Show Sources Plot",
                                         command=self._toggle_sources_plot)
        self._toggle_button.pack(side=tk.TOP, pady=5)

        # Create and pack the sources_frame **before** plotting
        self.sources_frame = ttk.Frame(left_frame)
        self.sources_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.sources_frame.pack_forget()

        # Plot components
        self._plot_components(ica_instance, epochs)
        self._canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create right frame
        selection_frame = ttk.Frame(main_frame)
        selection_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Component list with headers
        label_var = tk.StringVar(value="Selected Components:")
        self.tk_vars.append(label_var)
        ttk.Label(selection_frame, textvariable=label_var).pack(pady=5)

        # Create treeview for component info
        self._create_component_treeview(selection_frame)

        # Instructions
        instructions = (
            "Instructions:\n"
            "1. Click on components to select/deselect\n"
            "2. Selected components will be removed\n"
            "3. View additional info in the list\n"
            "4. Click 'Show Sources Plot' for detailed view\n"
            "5. Click Done when finished"
        )
        ttk.Label(selection_frame, text=instructions, wraplength=250).pack(pady=10)

        # Button frame
        button_frame = ttk.Frame(selection_frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Done",
                   command=self._finish_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                   command=self._cancel_selection).pack(side=tk.LEFT, padx=5)

        # Set window close protocol
        self._window.protocol("WM_DELETE_WINDOW", self._cancel_selection)

        # Update window
        self._window.update_idletasks()
        self._window.focus_set()

    def _create_component_treeview(self, parent: ttk.Frame) -> None:
        """Create treeview for component information.

        Args:
            parent: Parent frame for treeview
        """
        # Create treeview
        columns = ('component', 'score', 'label')
        self._tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)

        # Define headers
        self._tree.heading('component', text='Component')
        self._tree.heading('score', text='Score')
        self._tree.heading('label', text='Label')

        # Configure columns
        self._tree.column('component', width=80)
        self._tree.column('score', width=80)
        self._tree.column('label', width=120)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)

        # Pack widgets
        self._tree.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate data
        self._update_component_info()

    def _update_component_info(self) -> None:
        """Update component information in treeview."""
        # Clear existing items
        for item in self._tree.get_children():
            self._tree.delete(item)

        # Add components
        n_components = self._ica_instance.n_components_
        for idx in range(n_components):
            score = ""
            label = ""

            # Add score if available
            if self._component_scores is not None:
                if 'muscle_ratios' in self._component_scores:
                    score = f"{self._component_scores['muscle_ratios'][idx]:.2f}"

            # Add label if available
            if self._component_labels is not None:
                if 'labels' in self._component_labels:
                    label = self._component_labels['labels'][idx]

            values = (f"IC{idx}", score, label)

            # Highlight selected components
            tags = ('selected',) if idx in self.selected_components else ()
            self._tree.insert('', tk.END, values=values, tags=tags)

        # Configure tag for selected components
        self._tree.tag_configure('selected', background='lightgreen')


    def _toggle_sources_plot(self) -> None:
        """Toggle between components plot and sources plot."""
        if not self.showing_sources:
            # Switch to sources plot
            self._hide_components_plot()
            self._show_sources_plot()
            self._toggle_button.config(text="Show Components Plot")
            self.showing_sources = True
        else:
            # Switch back to components plot
            self._hide_sources_plot()
            self._show_components_plot()
            self._toggle_button.config(text="Show Sources Plot")
            self.showing_sources = False

    def _hide_components_plot(self):
        """Hide the components plot and its toolbar."""
        if self.components_frame:
            self.components_frame.pack_forget()

    def _show_components_plot(self):
        """Show the components plot and its toolbar."""
        if self.components_frame:
            self.components_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            if self._fig and self._canvas:
                self._canvas.draw_idle()

    def _show_sources_plot(self):
        """Show the sources plot embedded in the sources_frame."""
        try:
            # Repack the sources_frame to make it visible
            self.sources_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Create the plot using MNE's plot_sources with show=False
            fig = self._ica_instance.plot_sources(
                self._epochs,
                show=False,  # Prevent MNE from opening its own window
                block=False
            )

            # If a previous sources plot exists, destroy it to prevent duplicates
            if self.sources_canvas:
                self.sources_canvas.get_tk_widget().destroy()
                plt.close(self.sources_fig)
                if self.sources_toolbar:
                    self.sources_toolbar.destroy()

            # Assign the figure to sources_fig for future reference
            self.sources_fig = fig

            # Embed the Matplotlib figure into the sources_frame
            self.sources_canvas = FigureCanvasTkAgg(fig, master=self.sources_frame)
            self.sources_canvas.draw()
            self.sources_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Add a Matplotlib toolbar for the sources plot
            self.sources_toolbar = NavigationToolbar2Tk(self.sources_canvas, self.sources_frame)
            self.sources_toolbar.update()
            self.sources_toolbar.pack(side=tk.TOP, fill=tk.X)

        except Exception as e:
            print(f"Error showing sources plot: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Plot Error", f"An error occurred while plotting sources:\n{e}")

    def _hide_sources_plot(self):
        """Hide the sources plot and its toolbar."""
        if self.sources_frame:
            self.sources_frame.pack_forget()
            if self.sources_canvas:
                self.sources_canvas.get_tk_widget().destroy()
                self.sources_canvas = None
            if self.sources_toolbar:
                self.sources_toolbar.destroy()
                self.sources_toolbar = None
            if self.sources_fig:
                plt.close(self.sources_fig)
                self.sources_fig = None

    def _plot_components(self, ica_instance: mne.preprocessing.ICA, epochs: mne.Epochs) -> None:
        """Plot ICA components with enhanced information.

        Args:
            ica_instance: Fitted ICA instance
            epochs: MNE Epochs object
        """
        if self._fig is None:
            return

        self._fig.clear()
        data = ica_instance.get_sources(epochs).get_data()
        mean_data = data.mean(axis=0)
        var_data = data.std(axis=0)
        n_components = len(mean_data)

        # Calculate layout
        n_rows = int(np.ceil(np.sqrt(n_components)))
        n_cols = int(np.ceil(n_components / n_rows))

        for idx in range(n_components):
            ax = self._fig.add_subplot(n_rows, n_cols, idx + 1)

            # Plot mean activity with variability
            times = epochs.times
            component_data = mean_data[idx]
            ax.plot(times, component_data, 'b-', linewidth=1)
            ax.fill_between(times,
                            component_data - var_data[idx],
                            component_data + var_data[idx],
                            color='blue', alpha=0.2)

            # Add score/label to title if available
            title = f'IC{idx}'
            if self._component_scores is not None:
                if 'muscle_ratios' in self._component_scores:
                    score = self._component_scores['muscle_ratios'][idx]
                    title += f'\nScore: {score:.2f}'
                    if score > 2.0:  # Threshold for highlighting
                        ax.patch.set_facecolor('mistyrose')
                        ax.patch.set_alpha(0.3)

            if self._component_labels is not None:
                if 'labels' in self._component_labels:
                    title += f'\n{self._component_labels["labels"][idx]}'

            ax.set_title(title, fontsize=8)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.3)

            if idx in self.selected_components:
                ax.patch.set_facecolor('lightgreen')
                ax.patch.set_alpha(0.3)

        self._fig.tight_layout()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _on_click(self, event) -> None:
        """Handle component selection clicks."""
        if event.inaxes is None:
            return

        # Get subplot title
        title = event.inaxes.get_title().split('\n')[0]
        if not title.startswith('IC'):
            return

        try:
            comp_idx = int(title[2:])

            # Toggle selection
            if comp_idx in self.selected_components:
                self.selected_components.remove(comp_idx)
                event.inaxes.patch.set_facecolor('white')
            else:
                self.selected_components.add(comp_idx)
                event.inaxes.patch.set_facecolor('lightgreen')

            event.inaxes.patch.set_alpha(0.3)
            self._canvas.draw_idle()

            # Update info
            self._update_component_info()

        except Exception as e:
            print(f"Error handling click: {str(e)}")

    def _cleanup(self, components=None) -> None:
        """Clean up resources."""
        if self._cleanup_called:
            return

        self._cleanup_called = True

        # Call callback first
        if self.completion_callback and components is not None:
            self.completion_callback(components)

        # Clean up resources
        if self._window:
            self._window.destroy()

        if self._fig:
            plt.close(self._fig)
            self._fig = None

        # Ensure both plots are closed
        self._hide_sources_plot()

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
                          callback: Optional[Callable] = None,
                          component_scores: Optional[dict] = None,
                          component_labels: Optional[dict] = None) -> Optional[List[int]]:
    """Thread-safe ICA component selection.

    Args:
        ica_instance: Fitted ICA instance
        epochs: MNE Epochs object
        title: Window title
        root: Parent Tk window
        callback: Callback function for component selection
        component_scores: Dictionary of component scores
        component_labels: Dictionary of component labels

    Returns:
        List of selected component indices or None if selection was cancelled
    """
    if root is None:
        raise RuntimeError("No main Tk root provided")

    if callback is None:
        result_queue = queue.Queue()
        processing_complete = threading.Event()

        def internal_callback(components):
            try:
                result_queue.put(components)
            finally:
                processing_complete.set()

        callback = internal_callback

    def show_selector():
        try:
            selector = ICAComponentSelector(root)
            selector.select_components(
                ica_instance=ica_instance,
                epochs=epochs,
                title=title,
                callback=callback,
                component_scores=component_scores,
                component_labels=component_labels
            )
        except Exception as e:
            print(f"Error creating selector: {str(e)}")
            import traceback
            traceback.print_exc()
            callback([])

    # Schedule selector creation appropriately based on thread
    if threading.current_thread() is threading.main_thread():
        show_selector()
    else:
        root.after(0, show_selector)

    # If using internal callback, wait for result
    if callback == internal_callback:
        if not processing_complete.wait(timeout=300):  # 5-minute timeout
            print("Component selection timed out")
            return []

        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return []

    return None  # Return None if using external callback