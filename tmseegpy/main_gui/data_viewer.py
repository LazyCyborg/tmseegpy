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





