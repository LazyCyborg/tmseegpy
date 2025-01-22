# tmseegpy/ica_selector_gui/widgets/ica_selector.py
"""ICA component selector widget for the TMSEEG GUI with PyQt6"""

import numpy as np
from typing import List, Optional, Callable, Dict
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # Changed from qt5agg
from matplotlib.gridspec import GridSpec
from PyQt6.QtWidgets import (QWidget, QMainWindow, QVBoxLayout, QHBoxLayout,
                          QPushButton, QLabel, QFrame, QSplitter, QDialog,
                          QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal
import mne
import threading
import queue

class ICAComponentSelector:
    """Base class for ICA component selection"""

    def __init__(self, parent):
        self.parent = parent
        self.selected_components = set()
        self.completion_callback = None
        self._cleanup_called = False

        # Initialize GUI elements
        self._window = None
        self._fig = None
        self._canvas = None
        self._toolbar = None
        self._ica_instance = None
        self._epochs = None
        self._component_scores = None
        self._component_labels = None

        # Plot windows
        self.sources_window = None
        self.components_window = None
        self.scores_window = None

        # State tracking
        self.showing_sources = False
        self.showing_components = False
        self.showing_scores = False

        # Thread-safe GUI updates
        self.gui_queue = queue.Queue()

    def select_components(self,
                          ica_instance: mne.preprocessing.ICA,
                          raw: Optional[mne.io.Raw] = None,
                          epochs: Optional[mne.Epochs] = None,
                          title: str = "Select ICA Components",
                          callback: Optional[Callable] = None,
                          component_scores: Optional[dict] = None,
                          component_labels: Optional[dict] = None) -> None:
        """Setup and show the component selection window"""
        self.completion_callback = callback
        self._ica_instance = ica_instance
        self._raw = raw
        self._epochs = epochs
        self._component_scores = component_scores
        self._component_labels = component_labels

        # Create main window using Qt
        self._window = QMainWindow(self.parent)
        self._window.setWindowTitle(title)
        self._window.setMinimumSize(1600, 1000)

        # Create central widget and layout
        central_widget = QWidget()
        self._window.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Create left frame for plots
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)

        # Create matplotlib figure
        self._fig = plt.figure(figsize=(12, 8))
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.mpl_connect('button_press_event', self._on_click)

        # Create toolbar
        self._toolbar = NavigationToolbar2QT(self._canvas, left_frame)
        left_layout.addWidget(self._toolbar)

        # Create button panel
        button_panel = QHBoxLayout()

        # Add plot toggle buttons
        sources_btn = QPushButton("Show Sources Plot")
        sources_btn.clicked.connect(lambda: self._schedule_on_gui(self._toggle_sources_plot))
        button_panel.addWidget(sources_btn)

        components_btn = QPushButton("Show Components Plot")
        components_btn.clicked.connect(lambda: self._schedule_on_gui(self._toggle_components_plot))
        button_panel.addWidget(components_btn)

        scores_btn = QPushButton("Show Artifact Scores")
        scores_btn.clicked.connect(lambda: self._schedule_on_gui(self._toggle_scores_plot))
        button_panel.addWidget(scores_btn)

        left_layout.addLayout(button_panel)
        left_layout.addWidget(self._canvas)

        # Create right frame for controls
        right_frame = QFrame()
        right_frame.setMaximumWidth(250)
        right_layout = QVBoxLayout(right_frame)

        # Add instructions
        instructions = """
            Instructions:
            1. Click components to select/deselect
            2. Selected components will be removed
            3. Use plots for detailed views
            4. Click Done when finished
            """
        instructions_label = QLabel(instructions)
        instructions_label.setWordWrap(True)
        right_layout.addWidget(instructions_label)

        # Add control buttons
        button_layout = QHBoxLayout()

        done_btn = QPushButton("Done")
        done_btn.clicked.connect(self._finish_selection)
        button_layout.addWidget(done_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._cancel_selection)
        button_layout.addWidget(cancel_btn)

        right_layout.addStretch()
        right_layout.addLayout(button_layout)

        # Add frames to main layout
        layout.addWidget(left_frame, stretch=4)
        layout.addWidget(right_frame, stretch=1)

        # Plot initial components
        self._plot_components()

        # Show window
        self._window.show()

        # Setup GUI queue processing
        self._process_gui_queue()

    def _process_gui_queue(self):
        """Process GUI updates from queue"""
        try:
            while True:
                callback, args = self.gui_queue.get_nowait()
                callback(*args)
        except queue.Empty:
            pass
        if not self._cleanup_called and self._window:
            # Use QTimer instead of after()
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, self._process_gui_queue)

    def _schedule_on_gui(self, callback, *args):
        """Schedule callback on GUI thread"""
        self.gui_queue.put((callback, args))

    def _plot_components(self):
        """Plot ICA components"""
        if self._fig is None:
            return

        try:
            self._fig.clear()

            # Get data based on what's available
            if self._epochs is not None:
                data = self._ica_instance.get_sources(self._epochs).get_data()
                mean_data = data.mean(axis=0)
                var_data = data.std(axis=0)
                n_components = len(mean_data)
                times = self._epochs.times
                plot_variance = True
            elif self._raw is not None:
                data = self._ica_instance.get_sources(self._raw).get_data()
                n_components = len(data)
                times = np.arange(len(data[0])) / self._raw.info['sfreq']
                plot_variance = False
            else:
                raise ValueError("Neither epochs nor raw data provided")

            # Calculate layout
            n_rows = int(np.ceil(np.sqrt(n_components)))
            n_cols = int(np.ceil(n_components / n_rows))

            # Plot components
            for idx in range(n_components):
                ax = self._fig.add_subplot(n_rows, n_cols, idx + 1)

                if plot_variance:
                    ax.plot(times, mean_data[idx], 'b-', linewidth=1)
                    ax.fill_between(times,
                                    mean_data[idx] - var_data[idx],
                                    mean_data[idx] + var_data[idx],
                                    color='blue', alpha=0.2)
                    ax.axvline(x=0, color='r', linestyle='--', alpha=0.3)
                else:
                    ax.plot(times, data[idx], 'b-', linewidth=0.5)

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
            QMessageBox.critical(self._window, "Error", f"Error plotting components:\n{str(e)}")

    def _toggle_components_plot(self):
        """Toggle components plot visibility with enhanced visualization"""
        if not self.showing_components:
            # Set matplotlib backend before creating plot
            import matplotlib
            matplotlib.use('QtAgg')

            # Use welch method instead of multitaper for component plots
            psd_args = {
                'method': 'welch',
                'fmin': 1,
                'fmax': 100,
                'n_fft': 2048,
                'n_overlap': 1024,
                'n_per_seg': 2048,
                'average': 'mean'
            }

            try:
                # Create the dialog first
                dialog = QDialog(self._window)
                dialog.setWindowTitle("ICA Components")
                dialog.setMinimumSize(1200, 800)
                layout = QVBoxLayout(dialog)

                # Generate the plot
                fig = self._ica_instance.plot_components(
                    picks=None,
                    ch_type=None,
                    inst=self._raw if self._raw is not None else self._epochs,
                    plot_std=True,
                    reject='auto',
                    sensors=True,
                    show_names=False,
                    contours=6,
                    outlines='head',
                    image_interp='cubic',
                    res=64,
                    size=1.5,
                    cmap='RdBu_r',
                    colorbar=True,
                    cbar_fmt='%3.2f',
                    show=False,
                    psd_args=psd_args
                )

                # Handle both single figure and figure list cases
                if isinstance(fig, list):
                    main_fig = fig[0]
                    # Close any additional figures
                    for extra_fig in fig[1:]:
                        plt.close(extra_fig)
                else:
                    main_fig = fig

                # Create canvas and toolbar
                canvas = FigureCanvasQTAgg(main_fig)
                toolbar = NavigationToolbar2QT(canvas, dialog)

                # Add widgets to layout
                layout.addWidget(toolbar)
                layout.addWidget(canvas)

                # Connect cleanup
                dialog.finished.connect(lambda: self._cleanup_plot_windows())

                # Show dialog
                dialog.show()
                self.components_window = dialog
                self.showing_components = True

            except Exception as e:
                print(f"Error in component plotting: {str(e)}")
                if hasattr(self, 'components_window') and self.components_window:
                    self.components_window.close()
                plt.close('all')

        else:
            if self.components_window:
                plt.close('all')
                self.components_window.close()
            self.components_window = None
            self.showing_components = False

    def _show_mne_plot(self, plot_func, title):
        """Enhanced show MNE plot function with better window management"""
        try:
            dialog = QDialog(self._window)
            dialog.setWindowTitle(title)
            dialog.setMinimumSize(1200, 800)
            layout = QVBoxLayout(dialog)

            fig = plot_func()
            canvas = FigureCanvasQTAgg(fig)
            toolbar = NavigationToolbar2QT(canvas, dialog)

            layout.addWidget(toolbar)
            layout.addWidget(canvas)

            # Connect the dialog's close event to cleanup
            dialog.finished.connect(lambda: self._cleanup_plot_windows())

            dialog.show()
            return dialog

        except Exception as e:
            print(f"Error showing plot: {str(e)}")
            return None

    def _cleanup_plot_windows(self):
        """Cleanup function to ensure all related windows are properly closed"""
        plt.close('all')
        if hasattr(self, 'components_window') and self.components_window:
            self.components_window.close()
            self.components_window = None
        self.showing_components = False

    def _toggle_sources_plot(self):
        """Toggle sources plot visibility with enhanced visualization"""
        if not self.showing_sources:
            try:
                data = self._raw if self._raw is not None else self._epochs

                # Set matplotlib backend before creating plot
                import matplotlib
                import matplotlib.pyplot as plt
                matplotlib.use('QtAgg')

                with plt.style.context('default'):
                    # Different parameters for Raw vs Epochs data
                    if self._raw is not None:
                        # Parameters for Raw data
                        fig = self._ica_instance.plot_sources(
                            data,
                            picks=None,
                            start=0,
                            stop=10,
                            title="ICA Sources",
                            show=False,
                            block=False,
                            show_first_samp=True,
                            show_scrollbars=True,
                            time_format='float'
                        )
                    else:
                        # Parameters for Epochs data
                        fig = self._ica_instance.plot_sources(
                            data,
                            picks=None,
                            title="ICA Sources",
                            show=False,
                            block=False,
                            show_scrollbars=True
                        )

                    # Store the figure
                    self.sources_window = fig
                    # Use plt.show() instead of fig.show()
                    plt.show(block=False)
                    self.showing_sources = True

            except Exception as e:
                print(f"Error showing sources plot: {str(e)}")
                self.showing_sources = False
                if hasattr(self, 'sources_window') and self.sources_window:
                    try:
                        plt.close(self.sources_window)
                    except Exception as close_error:
                        print(f"Error closing sources window: {str(close_error)}")
                    finally:
                        self.sources_window = None
                plt.close('all')

        else:
            try:
                import matplotlib.pyplot as plt
                if self.sources_window:
                    plt.close('all')
                self.sources_window = None
                self.showing_sources = False
            except Exception as e:
                print(f"Error closing sources window: {str(e)}")

        # Force garbage collection after closing figures
        try:
            import gc
            gc.collect()
        except Exception as e:
            print(f"Error during garbage collection: {str(e)}")

    def _toggle_scores_plot(self):
        """Toggle scores plot visibility"""
        if not self.showing_scores:
            self.scores_window = self._show_scores_window()
            self.showing_scores = bool(self.scores_window)
        else:
            if self.scores_window:
                self.scores_window.close()
            self.scores_window = None
            self.showing_scores = False


    def _show_scores_window(self):
        """Create and show the artifact scores window"""
        if not self._component_scores:
            QMessageBox.warning(self._window, "Warning", "No component scores available")
            return None

        try:
            dialog = QDialog(self._window)
            dialog.setWindowTitle("Artifact Detection Scores")
            dialog.setMinimumSize(1200, 800)

            layout = QVBoxLayout(dialog)

            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(3, 2, figure=fig)

            # Plot TMS-Muscle ratios
            if 'tms_muscle' in self._component_scores:
                ax1 = fig.add_subplot(gs[0, 0])
                self._plot_scores(ax1,
                                  self._component_scores['tms_muscle'],
                                  "TMS-Muscle Ratios", 2.0)

            # Plot Blink z-scores
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_scores(ax2,
                              self._component_scores.get('blink', []),
                              "Blink Z-Scores", 2.5)

            # Plot Lateral Eye z-scores
            ax3 = fig.add_subplot(gs[1, 0])
            lat_eye = self._component_scores.get('lat_eye', [])
            lat_eye_scores = [max(scores) if isinstance(scores, list) else scores
                              for scores in lat_eye]
            self._plot_scores(ax3,
                              lat_eye_scores,
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

            # Add legend
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

            canvas = FigureCanvasQTAgg(fig)
            toolbar = NavigationToolbar2QT(canvas, dialog)

            layout.addWidget(toolbar)
            layout.addWidget(canvas)

            dialog.show()
            return dialog

        except Exception as e:
            print(f"Error showing scores plot: {str(e)}")
            QMessageBox.critical(self._window, "Error", f"Error showing scores plot:\n{str(e)}")
            return None

    def _plot_scores(self, ax, scores, title, threshold):
        """Plot component scores with threshold line"""
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

    def _on_click(self, event):
        """Handle component selection clicks for continuous data"""
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

    def _cleanup(self, components=None):
        """Clean up all windows and resources"""
        if self._cleanup_called:
            return

        self._cleanup_called = True

        # Call completion callback if provided
        if self.completion_callback and components is not None:
            self.completion_callback(components)

        # Close all child windows safely
        try:
            # Handle MNE figure windows
            if hasattr(self, 'sources_window') and self.sources_window:
                if hasattr(self.sources_window, 'close'):
                    self.sources_window.close()
                elif isinstance(self.sources_window, (plt.Figure, list)):
                    if isinstance(self.sources_window, list):
                        for fig in self.sources_window:
                            plt.close(fig)
                    else:
                        plt.close(self.sources_window)

            # Close Qt windows
            for window in [self.components_window, self.scores_window]:
                if window and hasattr(window, 'close'):
                    window.close()

            # Close main window
            if self._window:
                self._window.close()

        except Exception as e:
            print(f"Warning during cleanup: {str(e)}")
        finally:
            # Always try to close all matplotlib figures
            try:
                plt.close('all')
            except Exception as e:
                print(f"Warning closing matplotlib figures: {str(e)}")

            # Reset window references
            self.sources_window = None
            self.components_window = None
            self.scores_window = None
            self._window = None

class ICAComponentSelectorContinuous(ICAComponentSelector):
    """Adapted ICA Component Selector for continuous data"""

    def select_components(self,
                         ica_instance: mne.preprocessing.ICA,
                         raw: Optional[mne.io.Raw] = None,
                         epochs: Optional[mne.Epochs] = None,
                         title: str = "Select ICA Components",
                         callback: Optional[Callable] = None,
                         component_scores: Optional[dict] = None,
                         component_labels: Optional[dict] = None):
        """Show component selection interface for continuous data"""
        super().select_components(
            ica_instance=ica_instance,
            raw=raw,
            epochs=epochs,
            title=title,
            callback=callback,
            component_scores=component_scores,
            component_labels=component_labels
        )


    def _toggle_sources_plot(self):
        """Toggle sources plot visibility with enhanced visualization"""
        if not self.showing_sources:
            psd_args = {
                'fmin': 1,
                'fmax': 100,
                'n_fft': 2048,
                'n_overlap': 1024,
                'n_per_seg': 2048,
                'average': 'mean',
                'window': 'hamming'
            }

            data = self._raw if self._raw is not None else self._epochs
            start = 0 if self._raw is not None else None
            stop = 10 if self._raw is not None else None

            # Set matplotlib backend before creating plot
            import matplotlib
            matplotlib.use('QtAgg')

            with plt.style.context('default'):
                self.sources_window = self._show_mne_plot(
                    lambda: self._ica_instance.plot_sources(
                        data,
                        picks=None,
                        start=start,
                        stop=stop,
                        title="ICA Sources with PSD",
                        show=False,
                        block=False,
                        show_first_samp=True,
                        show_scrollbars=True,
                        time_format='float',
                        psd_args=psd_args
                    ),
                    "ICA Sources with PSD Analysis"
                )

            if self.sources_window and isinstance(self.sources_window, QDialog):
                self.showing_sources = True
        else:
            if self.sources_window:
                plt.close('all')  # Close all matplotlib figures
                self.sources_window.close()
            self.sources_window = None
            self.showing_sources = False

    def _toggle_components_plot(self):
        """Toggle components plot visibility with enhanced visualization"""
        if not self.showing_components:
            # Set matplotlib backend before creating plot
            import matplotlib
            matplotlib.use('QtAgg')

            psd_args = {
                'fmin': 1,
                'fmax': 100,
                'n_fft': 2048,
                'n_overlap': 1024,
                'n_per_seg': 2048,
                'average': 'mean'
            }

            image_args = {
                'combine': 'mean',
                'colorbar': True,
                'mask': None,
                'mask_style': None,
                'sigma': 1.0,
            }

            with plt.style.context('default'):
                self.components_window = self._show_mne_plot(
                    lambda: self._ica_instance.plot_components(
                        picks=None,
                        ch_type=None,
                        inst=self._raw if self._raw is not None else self._epochs,
                        plot_std=True,
                        reject='auto',
                        sensors=True,
                        show_names=False,
                        contours=6,
                        outlines='head',
                        image_interp='cubic',
                        res=64,
                        size=1.5,
                        cmap='RdBu_r',
                        colorbar=True,
                        cbar_fmt='%3.2f',
                        show=False,
                        image_args=image_args,
                        psd_args=psd_args,
                        nrows='auto',
                        ncols='auto'
                    ),
                    "ICA Components"
                )
            self.showing_components = bool(self.components_window)
        else:
            if self.components_window:
                plt.close('all')  # Close all matplotlib figures
                self.components_window.close()
            self.components_window = None
            self.showing_components = False
