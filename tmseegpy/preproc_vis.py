# preproc_vis.py
import os
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import numpy as np
import datetime
from typing import Union

from typing import Optional, Union, Dict, List, Tuple, Any
import warnings
import os

def create_step_directory(output_dir: str, session_name: str, step_name: str) -> str:
    """Create a directory for a specific preprocessing step's plots."""
    step_dir = Path(output_dir) / session_name / 'preprocessing_steps' / step_name
    step_dir.mkdir(parents=True, exist_ok=True)
    return str(step_dir)

def plot_raw_segments(raw, output_dir: str, session_name: str, step_name: str, duration: float = 5.0, overlap: float = 0.0):
    step_dir = create_step_directory(output_dir, session_name, step_name)
    
    ch_types = list(set(raw.get_channel_types()))
    data_type = 'csd' if 'csd' in ch_types else 'eeg'
    
    total_duration = raw.times[-1]
    step = duration - overlap
    start_times = np.arange(0, total_duration - duration, step)
    
    for i, start in enumerate(start_times):
        with mne.viz.use_browser_backend("matplotlib"):
            fig = raw.plot(duration=duration, start=start, show=False)
            plt.suptitle(f"{step_name} ({data_type.upper()}) - Segment {i+1} ({start:.1f}s - {start+duration:.1f}s)")
            fig.savefig(os.path.join(step_dir, f'{data_type}_segment_{i+1:03d}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)

def plot_epochs_grid(epochs, output_dir: str, session_name: str, step_name: str, epochs_per_plot: int = 10):
    step_dir = create_step_directory(output_dir, session_name, step_name)
    
    ch_types = list(set(epochs.get_channel_types()))
    data_type = 'csd' if 'csd' in ch_types else 'eeg'
    
    n_epochs = len(epochs)
    n_plots = (n_epochs + epochs_per_plot - 1) // epochs_per_plot
    
    for plot_idx in range(n_plots):
        start_idx = plot_idx * epochs_per_plot
        end_idx = min(start_idx + epochs_per_plot, n_epochs)
        
        with mne.viz.use_browser_backend("matplotlib"):
            fig = epochs[start_idx:end_idx].plot(show=False)
            plt.suptitle(f"{step_name} ({data_type.upper()}) - Epochs {start_idx+1}-{end_idx}")
            fig.savefig(os.path.join(step_dir, f'{data_type}_epochs_{plot_idx+1:03d}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)

def plot_evoked_response(epochs: mne.epochs, 
                        session_name: str, picks: Optional[str] = None,
                        xlim: Optional[Tuple[float, float]] = (-0.1, 0.3),
                        show: bool = False,
                        plot_gfp: Union[str, bool] = None
) -> None:
    """
    Plot averaged evoked response with butterfly plot and global field power.
    
    Parameters
    ----------
    epochs : mne.epochs
        The epochs object
    session_name: string
        The name of the session
    show : bool
        Whether to show the plot
    xlim : tuple
        X-axis limits in seconds (start_time, end_time)

    plot_gfp : bool | 'only 
   
    """
    if epochs is None:
        raise ValueError("Must create epochs before plotting evoked response")
 
    # Create evoked from epochs
    evoked = epochs.average()
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 8))

    evoked.plot(picks=picks, xlim=xlim, show=show, gfp=plot_gfp, title=f"Evoked response for {session_name}")
    
    return fig

