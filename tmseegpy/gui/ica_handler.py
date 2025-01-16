# ica_handler.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import mne
import numpy as np
from typing import List, Optional, Callable, Dict
import queue
import threading


def select_ica_components_continuous(ica_instance: mne.preprocessing.ICA,
                                     raw: mne.io.Raw,
                                     title: str = "Select ICA Components",
                                     root: Optional[tk.Tk] = None,
                                     callback: Optional[Callable] = None) -> Optional[List[int]]:
    """Thread-safe ICA component selection for continuous data.

    Parameters
    ----------
    ica_instance : mne.preprocessing.ICA
        The fitted ICA instance
    raw : mne.io.Raw
        The continuous raw data
    title : str
        Window title
    root : tk.Tk
        Root Tkinter window
    callback : Callable, optional
        Callback function for component selection

    Returns
    -------
    List[int] or None
        Selected component indices
    """
    if root is None:
        raise RuntimeError("No main Tk root provided")

    if callback is None:
        # Create synchronization primitives
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
            selector = ICAComponentSelectorContinuous(root)
            selector.select_components(
                ica_instance=ica_instance,
                raw=raw,
                title=title,
                callback=callback
            )
        except Exception as e:
            print(f"Error creating selector: {str(e)}")
            callback([])

    # Schedule selector on main thread
    if threading.current_thread() is threading.main_thread():
        show_selector()
    else:
        root.after(0, show_selector)

    # Wait for result if using internal callback
    if callback == internal_callback:
        processing_complete.wait()
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return []

    return None



class ICAComponentSelector:
    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.selected_components = set()
        self.completion_callback = None
        self._cleanup_called = False

        # Initialize all GUI elements as None
        self._window = None
        self._fig = None
        self._canvas = None
        self._toolbar = None
        self._ica_instance = None
        self._epochs = None
        self._component_scores = None
        self._component_labels = None

        # Source and component plot states
        self.sources_window = None
        self.components_window = None
        self.showing_sources = False
        self.showing_components = False

        # Component scores based on thresholds
        self.scores_window = None
        self.showing_scores = False

        # Create a queue for thread-safe GUI updates
        self.gui_queue = queue.Queue()

    def _process_gui_queue(self):
        """Process any pending GUI updates"""
        try:
            while True:
                callback, args = self.gui_queue.get_nowait()
                callback(*args)
        except queue.Empty:
            pass
        if not self._cleanup_called:
            self._window.after(100, self._process_gui_queue)

    def _schedule_on_gui(self, callback, *args):
        """Schedule a callback to run on the GUI thread"""
        self.gui_queue.put((callback, args))

    def select_components(self,
                          ica_instance: mne.preprocessing.ICA,
                          epochs: mne.Epochs,
                          title: str = "Select ICA Components",
                          callback: Optional[Callable] = None,
                          component_scores: Optional[dict] = None,
                          component_labels: Optional[dict] = None) -> None:

        # Store parameters
        self.completion_callback = callback
        self._ica_instance = ica_instance
        self._epochs = epochs
        self._component_scores = component_scores
        self._component_labels = component_labels

        # Create main window
        self._window = tk.Toplevel(self.parent)
        self._window.title(title)
        self._window.geometry("1600x1000")
        self._window.transient(self.parent)

        # Create main frame
        main_frame = ttk.Frame(self._window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left frame for plots
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        # Create figure
        self._fig = plt.figure(figsize=(12, 8))
        self._canvas = FigureCanvasTkAgg(self._fig, master=left_frame)
        self._canvas.mpl_connect('button_press_event', self._on_click)

        # Add toolbar
        self._toolbar = NavigationToolbar2Tk(self._canvas, left_frame)
        self._toolbar.pack(side=tk.TOP, fill=tk.X)

        # Create button frame
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(side=tk.TOP, pady=5)


        # Add plot toggle buttons - now all three buttons together
        ttk.Button(button_frame,
                   text="Show Sources Plot",
                   command=lambda: self._schedule_on_gui(self._toggle_sources_plot)
                   ).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame,
                   text="Show Components Plot",
                   command=lambda: self._schedule_on_gui(self._toggle_components_plot)
                   ).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame,
                   text="Show Artifact Scores",
                   command=lambda: self._schedule_on_gui(self._toggle_scores_plot)
                   ).pack(side=tk.LEFT, padx=5)

        # Plot initial components
        self._plot_components()
        self._canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create right frame for controls
        controls_frame = ttk.Frame(main_frame, width=200)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20)
        controls_frame.pack_propagate(False)

        # Add instructions
        instructions = (
            "Instructions:\n"
            "1. Click components to select/deselect\n"
            "2. Selected components will be removed\n"
            "3. Use plots for detailed views\n"
            "4. Click Done when finished"
        )
        ttk.Label(controls_frame, text=instructions, wraplength=180).pack(pady=10)

        # Add control buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(side=tk.BOTTOM, pady=20)

        ttk.Button(button_frame, text="Done",
                   command=self._finish_selection).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel",
                   command=self._cancel_selection).pack(side=tk.LEFT, padx=10)

        # Set window protocols
        self._window.protocol("WM_DELETE_WINDOW", self._cancel_selection)

        # Start GUI queue processing
        self._process_gui_queue()

    def _plot_components(self) -> None:
        """Plot ICA components safely"""
        if self._fig is None:
            return

        try:
            self._fig.clear()
            data = self._ica_instance.get_sources(self._epochs).get_data()
            mean_data = data.mean(axis=0)
            var_data = data.std(axis=0)
            n_components = len(mean_data)

            # Calculate layout
            n_rows = int(np.ceil(np.sqrt(n_components)))
            n_cols = int(np.ceil(n_components / n_rows))

            # Plot components
            for idx in range(n_components):
                ax = self._fig.add_subplot(n_rows, n_cols, idx + 1)
                times = self._epochs.times
                ax.plot(times, mean_data[idx], 'b-', linewidth=1)
                ax.fill_between(times,
                                mean_data[idx] - var_data[idx],
                                mean_data[idx] + var_data[idx],
                                color='blue', alpha=0.2)

                # Set title and styling
                ax.set_title(f'IC{idx}')
                ax.set_yticks([])
                ax.grid(True, alpha=0.3)
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.3)

                # Highlight selected components
                if idx in self.selected_components:
                    ax.patch.set_facecolor('lightgreen')
                    ax.patch.set_alpha(0.3)

            self._fig.tight_layout()
            if self._canvas is not None:
                self._canvas.draw_idle()

        except Exception as e:
            print(f"Error plotting components: {str(e)}")
            messagebox.showerror("Plot Error", f"Error plotting components:\n{str(e)}")

    def _show_mne_plot(self, plot_func, window_title):
        """Safely show an MNE plot in a new window"""
        try:
            # Create new window
            plot_window = tk.Toplevel(self._window)
            plot_window.title(window_title)
            plot_window.geometry("1600x1000")

            # Create frame for plot
            frame = ttk.Frame(plot_window)
            frame.pack(fill=tk.BOTH, expand=True)

            # Create plot with MNE
            with plt.ioff():
                # Force matplotlib backend for thread safety
                with mne.viz.use_browser_backend('matplotlib'):
                    fig = plot_func()

            # Embed plot
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)

            return plot_window

        except Exception as e:
            print(f"Error showing plot: {str(e)}")
            messagebox.showerror("Plot Error", f"Error showing plot:\n{str(e)}")
            return None

    def _toggle_sources_plot(self):
        """Safely toggle sources plot"""
        if not self.showing_sources:
            self.sources_window = self._show_mne_plot(
                lambda: self._ica_instance.plot_sources(self._epochs, show=False),
                "ICA Sources"
            )
            self.showing_sources = bool(self.sources_window)
        else:
            if self.sources_window:
                self.sources_window.destroy()
            self.sources_window = None
            self.showing_sources = False

    def _toggle_components_plot(self):
        """Safely toggle components plot"""
        if not self.showing_components:
            self.components_window = self._show_mne_plot(
                lambda: self._ica_instance.plot_components(picks=None, inst=self._epochs, show=False),
                "ICA Components"
            )
            self.showing_components = bool(self.components_window)
        else:
            if self.components_window:
                self.components_window.destroy()
            self.components_window = None
            self.showing_components = False

    def _toggle_scores_plot(self):
        """Toggle artifact scores plot window"""
        if not self.showing_scores:
            self.scores_window = self._show_scores_window()
            self.showing_scores = bool(self.scores_window)
        else:
            if self.scores_window:
                self.scores_window.destroy()
            self.scores_window = None
            self.showing_scores = False

    def _show_scores_window(self):
        """Create and show the artifact scores window"""
        try:
            # Check if we have component scores
            if not self._component_scores:
                messagebox.showwarning("Warning", "No component scores available")
                return None

            # Create new window
            scores_window = tk.Toplevel(self._window)
            scores_window.title("Artifact Detection Scores")
            scores_window.geometry("1200x800")

            # Create frame for plot
            frame = ttk.Frame(scores_window)
            frame.pack(fill=tk.BOTH, expand=True)

            # Create figure with subplots for different score types
            fig = plt.figure(figsize=(12, 8))
            gs = plt.GridSpec(3, 2, figure=fig)

            # Plot TMS-Muscle ratios
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_scores(ax1,
                              self._component_scores['tms_muscle'],
                              "TMS-Muscle Ratios", 2.0)

            # Plot Blink z-scores
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_scores(ax2,
                              self._component_scores['blink'],
                              "Blink Z-Scores", 2.5)

            # Plot Lateral Eye z-scores
            ax3 = fig.add_subplot(gs[1, 0])
            lateral_eye_scores = [max(scores) if isinstance(scores, list) else scores
                                  for scores in self._component_scores['lat_eye']]
            self._plot_scores(ax3, lateral_eye_scores,
                              "Lateral Eye Z-Scores", 2.0)

            # Plot Muscle frequency power ratios
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_scores(ax4,
                              self._component_scores['muscle'],
                              "Muscle Power Ratios", 0.6)

            # Plot Noise z-scores
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_scores(ax5,
                              self._component_scores['noise'],
                              "Noise Z-Scores", 4.0)

            # Add legend with threshold information
            ax_legend = fig.add_subplot(gs[2, 1])
            ax_legend.axis('off')
            legend_text = ("Threshold Values:\n"
                           "TMS-Muscle: 2.0\n"
                           "Blink: 2.5\n"
                           "Lateral Eye: 2.0\n"
                           "Muscle Power: 0.6\n"
                           "Noise: 4.0")
            ax_legend.text(0.1, 0.5, legend_text,
                           bbox=dict(facecolor='white', alpha=0.8),
                           transform=ax_legend.transAxes)

            fig.tight_layout()

            # Embed plot
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)

            return scores_window

        except Exception as e:
            print(f"Error showing scores plot: {str(e)}")
            messagebox.showerror("Plot Error", f"Error showing scores plot:\n{str(e)}")
            return None

    def _plot_scores(self, ax, scores, title, threshold):
        """Helper method to plot scores with threshold line"""
        x = range(len(scores))
        ax.bar(x, scores)
        ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        ax.set_xlabel('Component')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _cleanup(self, components=None):
        """Clean up all windows and resources"""
        if self._cleanup_called:
            return

        self._cleanup_called = True

        # Call completion callback if provided
        if self.completion_callback and components is not None:
            self.completion_callback(components)

        if self.scores_window:
            self.scores_window.destroy()

        # Clean up all windows
        for window in (self.sources_window, self.components_window, self._window):
            if window:
                window.destroy()

        # Close all figures
        plt.close('all')

    def _on_click(self, event):
        """Handle component selection clicks"""
        if event.inaxes is None:
            return

        title = event.inaxes.get_title()
        if not title.startswith('IC'):
            return

        try:
            comp_idx = int(title[2:])
            if comp_idx in self.selected_components:
                self.selected_components.remove(comp_idx)
            else:
                self.selected_components.add(comp_idx)
            self._plot_components()
        except ValueError:
            pass

    def _finish_selection(self):
        """Complete selection with chosen components"""
        components = sorted(list(self.selected_components))
        self._cleanup(components)

    def _cancel_selection(self):
        """Cancel selection"""
        self._cleanup([])


def select_ica_components(ica_instance: mne.preprocessing.ICA,
                          epochs: mne.Epochs,
                          title: str = "Select ICA Components",
                          root: Optional[tk.Tk] = None,
                          callback: Optional[Callable] = None) -> Optional[List[int]]:
    """Thread-safe ICA component selection"""
    if root is None:
        raise RuntimeError("No main Tk root provided")

    if callback is None:
        # Create synchronization primitives
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
                callback=callback
            )
        except Exception as e:
            print(f"Error creating selector: {str(e)}")
            callback([])

    # Schedule selector on main thread
    if threading.current_thread() is threading.main_thread():
        show_selector()
    else:
        root.after(0, show_selector)

    # Wait for result if using internal callback
    if callback == internal_callback:
        processing_complete.wait()
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return []

    return None


class ICAComponentSelectorContinuous(ICAComponentSelector):
    """Adapted ICA Component Selector for continuous data."""

    def select_components(self,
                          ica_instance: mne.preprocessing.ICA,
                          raw: mne.io.Raw,
                          title: str = "Select ICA Components",
                          callback: Optional[Callable] = None,
                          component_scores: Optional[dict] = None,   # <--- ADDED
                          component_labels: Optional[dict] = None):
        """
        Show component selection GUI for continuous data.

        Parameters
        ----------
        ica_instance : mne.preprocessing.ICA
            The fitted ICA instance
        raw : mne.io.Raw
            The continuous raw data
        title : str
            Window title
        callback : Callable, optional
            Callback function for component selection
        """
        # Store parameters
        self.completion_callback = callback
        self._ica_instance = ica_instance
        self._raw = raw
        self._component_scores = component_scores   # <--- ADDED
        self._component_labels = component_labels

        # By default, set up for the scores window if needed
        self.showing_scores = False
        self.scores_window = None

        # Create GUI as before
        self._window = tk.Toplevel(self.parent)
        self._window.title(title)
        self._window.geometry("1600x1000")
        self._window.transient(self.parent)

        # Create main frame
        main_frame = ttk.Frame(self._window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create left frame for plots
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        # Create figure
        self._fig = plt.figure(figsize=(12, 8))
        self._canvas = FigureCanvasTkAgg(self._fig, master=left_frame)
        self._canvas.mpl_connect('button_press_event', self._on_click)

        # Add toolbar
        self._toolbar = NavigationToolbar2Tk(self._canvas, left_frame)
        self._toolbar.pack(side=tk.TOP, fill=tk.X)

        # Create button frame
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(side=tk.TOP, pady=5)

        # Add plot toggle buttons
        ttk.Button(button_frame,
                   text="Show Sources Plot",
                   command=lambda: self._schedule_on_gui(self._toggle_sources_plot)
                   ).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame,
                   text="Show Components Plot",
                   command=lambda: self._schedule_on_gui(self._toggle_components_plot)
                   ).pack(side=tk.LEFT, padx=5)

        # Add a new button for artifact scores**
        ttk.Button(button_frame,
                   text="Show Artifact Scores",
                   command=lambda: self._schedule_on_gui(self._toggle_scores_plot)
                   ).pack(side=tk.LEFT, padx=5)

        # Plot initial components
        self._plot_components()
        self._canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create right frame for controls
        controls_frame = ttk.Frame(main_frame, width=200)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20)
        controls_frame.pack_propagate(False)

        # Add instructions
        instructions = (
            "Instructions:\n"
            "1. Click components to select/deselect\n"
            "2. Selected components will be removed\n"
            "3. Use plots for detailed views\n"
            "4. Click Done when finished"
        )
        ttk.Label(controls_frame, text=instructions, wraplength=180).pack(pady=10)

        # Add control buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(side=tk.BOTTOM, pady=20)

        ttk.Button(button_frame, text="Done",
                   command=self._finish_selection).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel",
                   command=self._cancel_selection).pack(side=tk.LEFT, padx=10)

        # Set window protocols
        self._window.protocol("WM_DELETE_WINDOW", self._cancel_selection)

        # Start GUI queue processing
        self._process_gui_queue()

    def _show_mne_plot(self, plot_func, window_title):
        try:
            plot_window = tk.Toplevel(self._window)
            plot_window.title(window_title)
            plot_window.geometry("1600x1000")

            frame = ttk.Frame(plot_window)
            frame.pack(fill=tk.BOTH, expand=True)

            with plt.ioff():
                with mne.viz.use_browser_backend('matplotlib'):
                    figs = plot_func()

            # If it's a single Figure, put it in a list
            if not isinstance(figs, list):
                figs = [figs]

            for fig in figs:
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                toolbar = NavigationToolbar2Tk(canvas, frame)
                toolbar.update()
                toolbar.pack(side=tk.BOTTOM, fill=tk.X)

            return plot_window

        except Exception as e:
            print(f"Error showing plot: {str(e)}")
            messagebox.showerror("Plot Error", f"Error showing plot:\n{str(e)}")
            return None

    def _plot_components(self) -> None:
        """Plot ICA components for continuous data"""
        if self._fig is None:
            return

        try:
            self._fig.clear()
            # Get source data
            sources = self._ica_instance.get_sources(self._raw)
            data = sources.get_data()

            # Calculate mean and std for visualization
            window_size = int(5 * self._raw.info['sfreq'])  # 5 second windows
            n_windows = data.shape[1] // window_size

            # Reshape data into windows for statistics
            windowed_data = data[:, :n_windows * window_size].reshape(data.shape[0], n_windows, window_size)
            mean_data = windowed_data.mean(axis=1)
            std_data = windowed_data.std(axis=1)

            n_components = len(mean_data)
            times = np.arange(window_size) / self._raw.info['sfreq']

            # Calculate layout
            n_rows = int(np.ceil(np.sqrt(n_components)))
            n_cols = int(np.ceil(n_components / n_rows))

            # Plot components
            for idx in range(n_components):
                ax = self._fig.add_subplot(n_rows, n_cols, idx + 1)
                ax.plot(times, mean_data[idx], 'b-', linewidth=1)
                ax.fill_between(times,
                                mean_data[idx] - std_data[idx],
                                mean_data[idx] + std_data[idx],
                                color='blue', alpha=0.2)

                # Set title and styling
                ax.set_title(f'IC{idx}')
                ax.set_yticks([])
                ax.grid(True, alpha=0.3)

                # Highlight selected components
                if idx in self.selected_components:
                    ax.patch.set_facecolor('lightgreen')
                    ax.patch.set_alpha(0.3)

            self._fig.tight_layout()
            if self._canvas is not None:
                self._canvas.draw_idle()

        except Exception as e:
            print(f"Error plotting components: {str(e)}")
            messagebox.showerror("Plot Error", f"Error plotting components:\n{str(e)}")

    def _toggle_sources_plot(self):
        """Safely toggle sources plot for continuous data"""
        if not self.showing_sources:
            self.sources_window = self._show_mne_plot(
                lambda: self._ica_instance.plot_sources(self._raw, show=False),
                "ICA Sources"
            )
            self.showing_sources = bool(self.sources_window)
        else:
            if self.sources_window:
                self.sources_window.destroy()
            self.sources_window = None
            self.showing_sources = False

    def _toggle_components_plot(self):
        """Safely toggle components plot for continuous data"""
        if not self.showing_components:
            self.components_window = self._show_mne_plot(
                lambda: self._ica_instance.plot_components(picks=None, inst=self._raw, show=False),
                "ICA Components"
            )
            self.showing_components = bool(self.components_window)
        else:
            if self.components_window:
                self.components_window.destroy()
            self.components_window = None
            self.showing_components = False

    def _toggle_scores_plot(self):
        """Toggle artifact scores plot window for continuous data."""
        if not self.showing_scores:
            self.scores_window = self._show_scores_window()
            self.showing_scores = bool(self.scores_window)
        else:
            if self.scores_window:
                self.scores_window.destroy()
            self.scores_window = None
            self.showing_scores = False

    def _show_scores_window(self):
        """Create and show the artifact scores window for continuous data."""
        try:
            # Check if we have component scores
            if not self._component_scores:
                messagebox.showwarning("Warning", "No component scores available")
                return None

            # Create new window
            scores_window = tk.Toplevel(self._window)
            scores_window.title("Artifact Detection Scores")
            scores_window.geometry("1200x800")

            # Create frame for plot
            frame = ttk.Frame(scores_window)
            frame.pack(fill=tk.BOTH, expand=True)

            # Create figure with subplots for different score types
            fig = plt.figure(figsize=(12, 8))
            gs = plt.GridSpec(3, 2, figure=fig)

            # Plot TMS-Muscle ratios (only if they exist and if relevant for continuous)
            if 'tms_muscle' in self._component_scores:
                ax1 = fig.add_subplot(gs[0, 0])
                self._plot_scores(ax1,
                                  self._component_scores['tms_muscle'],
                                  "TMS-Muscle Ratios", 2.0)
            else:
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.set_title("No TMS-Muscle data for continuous")
                ax1.axis('off')

            # Plot Blink z-scores
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_scores(ax2,
                              self._component_scores.get('blink', []),
                              "Blink Z-Scores", 2.5)

            # Plot Lateral Eye z-scores
            ax3 = fig.add_subplot(gs[1, 0])
            lat_eye = self._component_scores.get('lat_eye', [])
            # If lat_eye is a list of lists, you can flatten or take max
            lat_eye_scores = [max(scores) if isinstance(scores, list) else scores
                              for scores in lat_eye]
            self._plot_scores(ax3, lat_eye_scores,
                              "Lateral Eye Z-Scores", 2.0)

            # Plot Muscle frequency power ratios
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_scores(ax4,
                              self._component_scores.get('muscle', []),
                              "Muscle Power Ratios", 0.6)

            # Plot Noise z-scores
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_scores(ax5,
                              self._component_scores.get('noise', []),
                              "Noise Z-Scores", 4.0)

            # Add legend with threshold information
            ax_legend = fig.add_subplot(gs[2, 1])
            ax_legend.axis('off')
            legend_text = ("Threshold Values:\n"
                           "TMS-Muscle: 2.0\n"
                           "Blink: 2.5\n"
                           "Lateral Eye: 2.0\n"
                           "Muscle Power: 0.6\n"
                           "Noise: 4.0")
            ax_legend.text(0.1, 0.5, legend_text,
                           bbox=dict(facecolor='white', alpha=0.8),
                           transform=ax_legend.transAxes)

            fig.tight_layout()

            # Embed plot
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)

            return scores_window

        except Exception as e:
            print(f"Error showing scores plot: {str(e)}")
            messagebox.showerror("Plot Error", f"Error showing scores plot:\n{str(e)}")
            return None

    def _plot_scores(self, ax, scores, title, threshold):
        """Helper method to plot scores with threshold line."""
        if not scores:
            ax.set_title(f"No data for {title}")
            ax.axis('off')
            return

        x = range(len(scores))
        ax.bar(x, scores)
        ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        ax.set_xlabel('Component')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)