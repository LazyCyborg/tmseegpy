# preproc_vis.py
import os
from pathlib import Path
import mne
from typing import Optional, Union, Dict, List, Tuple, Any


def create_step_directory(output_dir: str, session_name: str, step_name: str) -> str:
    """Create a directory for a specific preprocessing step's data files."""
    step_dir = Path(output_dir) / session_name / 'preprocessing_steps' / step_name
    step_dir.mkdir(parents=True, exist_ok=True)
    return str(step_dir)


def save_raw_data(raw: mne.io.Raw, output_dir: str, session_name: str, step_name: str) -> str:
    """
    Save raw data after a preprocessing step.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data to save
    output_dir : str
        Base output directory
    session_name : str
        Name of the session
    step_name : str
        Name of the preprocessing step

    Returns
    -------
    str
        Path to the saved file
    """
    step_dir = create_step_directory(output_dir, session_name, step_name)

    # Create filename
    filename = f"{session_name}_{step_name}_raw.fif"
    file_path = os.path.join(step_dir, filename)

    # Save the raw data
    raw.save(file_path, overwrite=True)

    return file_path


def save_epochs_data(epochs: mne.Epochs, output_dir: str, session_name: str, step_name: str) -> str:
    """
    Save epochs data after a preprocessing step.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to save
    output_dir : str
        Base output directory
    session_name : str
        Name of the session
    step_name : str
        Name of the preprocessing step

    Returns
    -------
    str
        Path to the saved file
    """
    if epochs is None or len(epochs) == 0:
        print("Warning: No epochs data available to save")
        return None

    step_dir = create_step_directory(output_dir, session_name, step_name)

    # Create filename
    filename = f"{session_name}_{step_name}_epo.fif"
    file_path = os.path.join(step_dir, filename)

    # Save the epochs
    epochs.save(file_path, overwrite=True)

    return file_path



