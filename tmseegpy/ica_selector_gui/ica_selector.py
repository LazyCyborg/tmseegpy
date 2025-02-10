# tmseegpy/ica_selector_gui/ica_selector.py
"""ICA component selector widget for the TMSEEG GUI with PyQt6"""

import numpy as np
from typing import List, Optional, Callable, Dict
import matplotlib
#matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # Changed from qt5agg
from matplotlib.gridspec import GridSpec
from PyQt6.QtWidgets import (QWidget, QMainWindow, QVBoxLayout, QHBoxLayout,
                          QPushButton, QLabel, QFrame, QSplitter, QDialog,
                          QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal
import mne
mne.viz.set_browser_backend('qt')
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
        self._window.setMinimumSize(1600, 800)

        # Create central widget and layout
        central_widget = QWidget()
        self._window.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Create left frame for plots
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)

        # Create matplotlib figure
        self._fig = plt.figure(figsize=(14, 6))
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
           # import matplotlib
            #matplotlib.use('QtAgg')
            mne.viz.set_browser_backend('qt')

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
                # Set matplotlib backend before creating plot
                import matplotlib
                matplotlib.use('QtAgg')

                with plt.style.context('default'):
                    if self._raw is not None:
                        # For Raw data
                        fig = self._ica_instance.plot_sources(
                            self._raw,
                            picks=None,
                            start=0,
                            stop=10,
                            show=True,
                            block=False
                        )
                    else:
                        # For Epochs data
                        fig = self._ica_instance.plot_sources(
                            self._epochs,
                            picks=None,
                            show=True,
                            block=False
                        )

                    # Store the figure reference
                    if hasattr(fig, 'canvas') and hasattr(fig.canvas, 'manager'):
                        self.sources_window = fig.canvas.manager.window
                    else:
                        self.sources_window = fig
                    self.showing_sources = True

            except Exception as e:
                print(f"Error showing plot: {str(e)}")
                if hasattr(self, 'sources_window') and self.sources_window:
                    try:
                        if isinstance(self.sources_window, plt.Figure):
                            plt.close(self.sources_window)
                        else:
                            self.sources_window.close()
                    except:
                        pass
                plt.close('all')

        else:
            if self.sources_window:
                try:
                    if isinstance(self.sources_window, plt.Figure):
                        plt.close(self.sources_window)
                    else:
                        self.sources_window.close()
                except:
                    pass
                plt.close('all')
            self.sources_window = None
            self.showing_sources = False

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
        """Create and show the artifact classification window"""
        try:
            from ..ica_topo_classifier import ICATopographyClassifier
            classifier = ICATopographyClassifier(self._ica_instance,
                                                 self._epochs if self._epochs is not None else self._raw)
            results = classifier.classify_all_components()

            dialog = QDialog(self._window)
            dialog.setWindowTitle("Artifact Classification Analysis")
            dialog.setMinimumSize(1200, 800)
            layout = QVBoxLayout(dialog)

            # Create top frame for plots
            plot_frame = QFrame()
            plot_layout = QVBoxLayout(plot_frame)

            fig = plt.figure(figsize=(12, 4))  # Reduced height for plots
            gs = GridSpec(1, 2, figure=fig)  # Changed to 1 row, 2 columns

            # Plot Z-scores
            ax1 = fig.add_subplot(gs[0, 0])
            z_scores = [res['details']['max_zscore'] for res in results.values()]
            self._plot_classifier_metric(ax1, z_scores,
                                         "Maximum Z-Scores",
                                         classifier.zscore_threshold,
                                         "Z-Score")

            # Plot number of peaks
            ax2 = fig.add_subplot(gs[0, 1])
            n_peaks = [res['details']['n_peaks'] for res in results.values()]
            self._plot_classifier_metric(ax2, n_peaks,
                                         "Number of Peaks",
                                         classifier.peak_count_threshold,
                                         "Count",
                                         threshold_direction='below')

            canvas = FigureCanvasQTAgg(fig)
            toolbar = NavigationToolbar2QT(canvas, dialog)

            plot_layout.addWidget(toolbar)
            plot_layout.addWidget(canvas)

            # Create text area with scroll
            text_scroll = QScrollArea()
            text_scroll.setWidgetResizable(True)
            text_widget = QLabel()
            text_scroll.setWidget(text_widget)
            text_scroll.setMinimumHeight(400)  # Set minimum height for scroll area

            # Generate classification summary text
            summary_text = "Classification Summary:\n\n"
            for idx, res in results.items():
                if res['classification'] == 'artifact':
                    reasons = res['details']['reasons']
                    summary_text += f"Component {idx}: ARTIFACT\n"
                    summary_text += f"    Z-score: {res['details']['max_zscore']:.2f}\n"
                    summary_text += f"    Peaks: {res['details']['n_peaks']}\n"
                    summary_text += f"    Reasons: {', '.join(reasons)}\n\n"

            # Add threshold information
            threshold_text = (
                "\nClassification Thresholds:\n"
                f"Z-score > {classifier.zscore_threshold}\n"
                f"Peak count < {classifier.peak_count_threshold}\n"
                f"At least 2 criteria must be met for artifact classification"
            )

            text_widget.setText(summary_text + threshold_text)
            text_widget.setStyleSheet("font-family: monospace;")

            # Add both frames to main layout
            layout.addWidget(plot_frame)
            layout.addWidget(text_scroll)

            dialog.show()
            return dialog

        except Exception as e:
            print(f"Error showing classification analysis: {str(e)}")
            QMessageBox.critical(self._window, "Error",
                                 f"Error showing classification analysis:\n{str(e)}")
            return None

    def _plot_classifier_metric(self, ax, values, title, threshold, ylabel,
                                threshold_direction='above'):
        """Plot component metric with threshold line"""
        x = range(len(values))
        bars = ax.bar(x, values)

        # Add threshold line
        ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')

        # Color bars based on threshold
        for i, bar in enumerate(bars):
            if threshold_direction == 'above':
                if values[i] > threshold:
                    bar.set_color('salmon')
            else:  # below
                if values[i] < threshold:
                    bar.set_color('salmon')

        ax.set_xlabel('Component')
        ax.set_ylabel(ylabel)
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
        for window in [self.sources_window, self.components_window, self.scores_window]:
            if window:
                try:
                    # Check if the window is a Qt dialog
                    if isinstance(window, QDialog):
                        window.close()
                    # Handle MNE browser windows
                    elif hasattr(window, 'close'):
                        try:
                            window.close()
                        except (RuntimeError, AttributeError):
                            # Ignore if window was already closed
                            pass
                    # For matplotlib figures
                    elif isinstance(window, plt.Figure):
                        plt.close(window)
                except Exception as e:
                    print(f"Warning: Error closing window: {str(e)}")

        # Close main window if it exists
        if self._window:
            try:
                self._window.close()
            except Exception as e:
                print(f"Warning: Error closing main window: {str(e)}")

        # Make sure all matplotlib figures are closed
        plt.close('all')

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
