# tmseegpy_gui/components/data_viewer.py

import streamlit as st
import mne
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple
import numpy as np
import subprocess
import tempfile
import os
import sys
import platform
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QSpinBox, QDoubleSpinBox, QCheckBox)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import threading
from typing import Union
from scipy import signal

class DataViewer:
    """Handles visualization of EEG data."""

    @staticmethod
    def display_data_info(data: Union[mne.io.Raw, mne.Epochs]) -> None:
        """Display basic information about the data."""
        if data is None:
            st.warning("No data available")
            return

        try:
            st.subheader("Dataset Information")

            if isinstance(data, mne.io.Raw):
                info_text = (
                    f"Raw Data:\n"
                    f"- Number of channels: {len(data.ch_names)}\n"
                    f"- Sampling frequency: {data.info['sfreq']} Hz\n"
                    f"- Duration: {data.times[-1]:.2f} seconds\n"
                    f"- Bad channels: {data.info['bads'] if data.info['bads'] else 'None'}\n"
                )
            else:  # Epochs
                info_text = (
                    f"Epoched Data:\n"
                    #f"- Number of epochs: {len(data)}\n"
                    f"- Number of channels: {len(data.ch_names)}\n"
                    f"- Sampling frequency: {data.info['sfreq']} Hz\n"
                   # f"- Epoch duration: {data.times[-1] - data.times[0]:.2f} seconds\n"
                    f"- Bad channels: {data.info['bads'] if data.info['bads'] else 'None'}\n"
                )

            st.text(info_text)

            # Show channel types
            ch_types = set(data.get_channel_types())
            st.text(f"Channel types: {', '.join(ch_types)}")

            # Show events if epochs
            if isinstance(data, mne.Epochs):
                event_ids = data.event_id
                st.text(f"Event IDs: {event_ids}")

        except Exception as e:
            st.error(f"Error displaying data info: {str(e)}")

    @staticmethod
    def view_raw(raw: mne.io.Raw) -> None:
        """Open raw data in MNE's Qt viewer."""
        try:
            # Create a temporary file to save the raw data
            with tempfile.NamedTemporaryFile(suffix='-raw.fif', delete=False) as tmp:
                temp_fname = tmp.name
                raw.save(temp_fname, overwrite=True)

            # Create Python script for viewer
            viewer_script = f"""
import mne
import sys
import os

try:
    raw = mne.io.read_raw_fif("{temp_fname}", preload=True)
    raw.plot(block=True)
finally:
    # Cleanup temp file
    try:
        os.remove("{temp_fname}")
    except:
        pass
"""
            # Save the script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                script_fname = tmp.name
                tmp.write(viewer_script)

            # Get the Python executable path
            python_exec = sys.executable

            # Launch the viewer in a separate process
            st.info("Launching MNE Raw Data Viewer...")

            # Different startup depending on platform
            if platform.system() == 'Windows':
                subprocess.Popen([python_exec, script_fname], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # Linux/Mac
                subprocess.Popen([python_exec, script_fname])

            # Store cleanup function in session state
            def cleanup():
                try:
                    os.remove(script_fname)
                except:
                    pass

            st.session_state['cleanup'] = cleanup

        except Exception as e:
            st.error(f"Error launching raw data viewer: {str(e)}")

    @staticmethod
    def view_epochs(epochs: mne.Epochs) -> None:
        """Open epochs in MNE's Qt viewer."""
        try:
            # Create a temporary file to save the epochs
            with tempfile.NamedTemporaryFile(suffix='-epo.fif', delete=False) as tmp:
                temp_fname = tmp.name
                epochs.save(temp_fname, overwrite=True)

            # Create Python script for viewer
            viewer_script = f"""
import mne
import sys
import os

try:
    epochs = mne.read_epochs("{temp_fname}", preload=True)
    epochs.plot(block=True)
finally:
    # Cleanup temp file
    try:
        os.remove("{temp_fname}")
    except:
        pass
"""
            # Save the script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                script_fname = tmp.name
                tmp.write(viewer_script)

            # Get the Python executable path
            python_exec = sys.executable

            # Launch the viewer in a separate process
            st.info("Launching MNE Epochs Viewer...")

            # Different startup depending on platform
            if platform.system() == 'Windows':
                subprocess.Popen([python_exec, script_fname], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # Linux/Mac
                subprocess.Popen([python_exec, script_fname])

            # Store cleanup function in session state
            def cleanup():
                try:
                    os.remove(script_fname)
                except:
                    pass

            st.session_state['cleanup'] = cleanup

        except Exception as e:
            st.error(f"Error launching epochs viewer: {str(e)}")

    @staticmethod
    def view_ica_components(ica: mne.preprocessing.ICA, epochs: mne.Epochs) -> None:
        """Open ICA components in MNE's Qt viewer."""
        try:
            # Create temporary files to save the ICA and epochs
            with tempfile.NamedTemporaryFile(suffix='-ica.fif', delete=False) as tmp_ica, \
                    tempfile.NamedTemporaryFile(suffix='-epo.fif', delete=False) as tmp_epo:

                temp_ica_fname = tmp_ica.name
                temp_epo_fname = tmp_epo.name

                # Save with overwrite=True to handle existing files
                ica.save(temp_ica_fname, overwrite=True)
                epochs.save(temp_epo_fname, overwrite=True)

            # Create temporary file for the script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                script_fname = tmp.name

            # Create Python script for viewer - with explicit line endings
            script_lines = [
                'import mne',
                'import sys',
                'import os',
                '',
                'try:',
                f'    ica = mne.preprocessing.read_ica("{temp_ica_fname}")',
                f'    epochs = mne.read_epochs("{temp_epo_fname}", preload=True)',
                '    fig = ica.plot_sources(epochs, block=True)',
                'finally:',
                '    # Cleanup temp files',
                '    try:',
                f'        os.remove("{temp_ica_fname}")',
                f'        os.remove("{temp_epo_fname}")',
                '    except:',
                '        pass'
            ]

            viewer_script = '\n'.join(script_lines)

            # Save the script with explicit encoding
            with open(script_fname, 'w', encoding='utf-8', newline='\n') as f:
                f.write(viewer_script)

            # Get the Python executable path
            python_exec = sys.executable

            # Launch the viewer in a separate process
            st.info("Launching MNE ICA Components Viewer...")

            # Different startup depending on platform
            if platform.system() == 'Windows':
                subprocess.Popen([python_exec, script_fname], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # Linux/Mac
                subprocess.Popen([python_exec, script_fname])

            # Store cleanup function in session state
            def cleanup():
                try:
                    os.remove(script_fname)
                except:
                    pass

            st.session_state['cleanup'] = cleanup

        except Exception as e:
            st.error(f"Error launching ICA components viewer: {str(e)}")

    @staticmethod
    def plot_psd(data: Union[mne.io.Raw, mne.Epochs]) -> None:
        """Open PSD viewer in a separate window."""
        try:
            # Create temporary file to save the data
            with tempfile.NamedTemporaryFile(suffix='_raw.fif', delete=False) as tmp:
                temp_fname = tmp.name
                data.save(temp_fname, overwrite=True)

            # Create Python script for viewer
            viewer_script = f"""import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QComboBox, QSpinBox, QCheckBox, QApplication)
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import mne
import traceback

class PSDViewer(QMainWindow):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Interactive PSD Viewer')
        self.setGeometry(100, 100, 1200, 800)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Power', units='dB')
        self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        layout.addWidget(self.plot_widget)

        # Control panel
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)

        # Method selection
        method_label = QLabel('Method:')
        self.method_combo = QComboBox()
        self.method_combo.addItems(['welch', 'multitaper'])
        self.method_combo.currentTextChanged.connect(self.update_plot)

        # Frequency range
        fmin_label = QLabel('Fmin (Hz):')
        self.fmin_spin = QSpinBox()
        self.fmin_spin.setRange(0, 1000)
        self.fmin_spin.setValue(0)
        self.fmin_spin.valueChanged.connect(self.update_plot)

        fmax_label = QLabel('Fmax (Hz):')
        self.fmax_spin = QSpinBox()
        self.fmax_spin.setRange(1, 1000)
        self.fmax_spin.setValue(100)
        self.fmax_spin.valueChanged.connect(self.update_plot)

        # Average checkbox
        self.average_check = QCheckBox('Average')
        self.average_check.setChecked(True)
        self.average_check.stateChanged.connect(self.update_plot)

        # Log scale checkbox
        self.log_check = QCheckBox('Log Scale')
        self.log_check.setChecked(True)
        self.log_check.stateChanged.connect(self.update_plot)

        # Add widgets to control layout
        for widget in [method_label, self.method_combo,
                      fmin_label, self.fmin_spin,
                      fmax_label, self.fmax_spin,
                      self.average_check, self.log_check]:
            control_layout.addWidget(widget)

        control_layout.addStretch()
        layout.addWidget(control_widget)

        # Initial plot
        self.update_plot()

    def compute_psd(self):
        try:
            method = self.method_combo.currentText()
            fmin = self.fmin_spin.value()
            fmax = self.fmax_spin.value()

            print(f"Computing PSD with method={{method}}, fmin={{fmin}}, fmax={{fmax}}")

            # Common keyword arguments for both methods
            common_kwargs = {{
                'fmin': fmin,
                'fmax': fmax,
                'picks': 'data',
                'verbose': True
            }}

            # Method-specific parameters
            if method == 'welch':
                method_kw = {{
                    'n_fft': 2048,
                    'n_overlap': 1024,
                    'window': 'hamming'
                }}
            else:  # multitaper
                method_kw = {{
                    'bandwidth': 4,
                    'adaptive': False,
                    'low_bias': True
                }}

            # Compute the spectrum
            spectrum = self.data.compute_psd(
                method=method,
                **common_kwargs,
                **method_kw
            )

            if spectrum is None:
                print("Spectrum computation returned None")
                return None, None

            # Get the data arrays
            try:
                psd_data = spectrum.get_data()
                freq_data = spectrum.freqs
            except AttributeError as e:
                print(f"Error accessing spectrum data: {{e}}")
                return None, None

            print(f"PSD shape: {{psd_data.shape}}")
            print(f"Freqs shape: {{freq_data.shape}}")

            return psd_data, freq_data

        except Exception as exc:
            print("Error computing PSD:")
            print(traceback.format_exc())
            return None, None

    def update_plot(self):
        try:
            self.plot_widget.clear()

            psd_data, freq_data = self.compute_psd()
            if psd_data is None or freq_data is None:
                return

            if self.log_check.isChecked():
                # Avoid log of zero or negative values
                psd_data = np.maximum(psd_data, np.finfo(float).tiny)
                psd_data = 10 * np.log10(psd_data)

            if self.average_check.isChecked():
                mean_psd = np.mean(psd_data, axis=0)
                std_psd = np.std(psd_data, axis=0) / np.sqrt(psd_data.shape[0])

                # Plot mean
                self.plot_widget.plot(freq_data, mean_psd, pen='b', name='Mean PSD')

                # Plot confidence interval
                upper = mean_psd + 1.96 * std_psd
                lower = mean_psd - 1.96 * std_psd

                # Create fill between curves
                fill = pg.FillBetweenItem(
                    pg.PlotDataItem(freq_data, upper),
                    pg.PlotDataItem(freq_data, lower),
                    brush=(100, 100, 255, 50)
                )
                self.plot_widget.addItem(fill)
            else:
                # Plot individual channels
                for chan_idx in range(psd_data.shape[0]):
                    self.plot_widget.plot(
                        freq_data, 
                        psd_data[chan_idx], 
                        pen=(chan_idx, psd_data.shape[0] * 1.3),
                        name=f'Channel {{chan_idx+1}}'
                    )

            # Update axis labels
            self.plot_widget.setLabel('left', 'Power', 
                                    units='dB' if self.log_check.isChecked() else 'µV²/Hz')
            self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')

        except Exception as exc:
            print(f"Error updating plot: {{str(exc)}}")

if __name__ == '__main__':
    import atexit

    app = QApplication(sys.argv)

    # Load the appropriate data type
    fname = r"{temp_fname}"  # Use raw string to handle Windows paths

    try:
        # Always use preload=True for spectral analysis
        data = mne.io.read_raw_fif(fname, preload=True)

        viewer = PSDViewer(data)
        viewer.show()
        app.exec()
    finally:
        # Cleanup
        try:
            os.remove(fname)
        except:
            pass"""

            # Save the script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                script_fname = tmp.name
                tmp.write(viewer_script)

            # Get the Python executable path
            python_exec = sys.executable

            # Launch the viewer in a separate process
            st.info("Launching PSD Viewer...")

            # Different startup depending on platform
            if platform.system() == 'Windows':
                subprocess.Popen([python_exec, script_fname],
                                 creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # Linux/Mac
                subprocess.Popen([python_exec, script_fname])

            # Store cleanup function in session state
            def cleanup():
                try:
                    os.remove(script_fname)
                    os.remove(temp_fname)
                except:
                    pass

            st.session_state['cleanup'] = cleanup

        except Exception as e:
            st.error(f"Error launching PSD viewer: {str(e)}")
            st.exception(e)




