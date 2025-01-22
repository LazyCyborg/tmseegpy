# preproc_vis.py
import os
import matplotlib

matplotlib.use('Agg')  # Set backend to Agg for thread safety
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import numpy as np
import warnings
from typing import Optional, Union, Dict, List, Tuple, Any


def create_step_directory(output_dir: str, session_name: str, step_name: str) -> str:
    """Create a directory for a specific preprocessing step's plots."""
    step_dir = Path(output_dir) / session_name / 'preprocessing_steps' / step_name
    step_dir.mkdir(parents=True, exist_ok=True)
    return str(step_dir)


def plot_raw_segments(raw, output_dir: str, session_name: str, step_name: str, duration: float = 5.0,
                      overlap: float = 0.0):
    """Plot segments of raw data."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        step_dir = create_step_directory(output_dir, session_name, step_name)
        ch_types = list(set(raw.get_channel_types()))
        data_type = 'csd' if 'csd' in ch_types else 'eeg'

        n_channels = len(raw.ch_names)
        height_per_chan = 0.5
        fig_height = n_channels * height_per_chan
        fig_width = 15

        total_duration = raw.times[-1]
        step = duration - overlap
        start_times = np.arange(0, total_duration - duration, step)

        scalings = dict(eeg=50e-6, csd=50)

        for i, start in enumerate(start_times):
            fig = plt.figure(figsize=(fig_width, fig_height))

            # Plot each channel
            for ch_idx, ch_name in enumerate(raw.ch_names):
                ax = plt.subplot(n_channels, 1, ch_idx + 1)
                data, times = raw[ch_name, int(start * raw.info['sfreq']):
                                           int((start + duration) * raw.info['sfreq'])]
                ax.plot(times, data.T * scalings.get(data_type, 1))
                ax.set_ylabel(ch_name)
                if ch_idx < n_channels - 1:
                    ax.set_xticks([])

            plt.suptitle(
                f"{step_name} ({data_type.upper()}) - Segment {i + 1} ({start:.1f}s - {start + duration:.1f}s)")
            plt.tight_layout()

            fig.savefig(os.path.join(step_dir, f'{data_type}_segment_{i + 1:03d}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)


def plot_epochs_grid(epochs, output_dir: str, session_name: str, step_name: str, epochs_per_plot: int = 10,
                     channels_per_plot: int = 16):
    """Plot grid of epochs using MNE's built-in plotting."""
    import mne
    import numpy as np

    with mne.viz.use_browser_backend('matplotlib'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            if epochs is None or len(epochs) == 0:
                print("Warning: No epochs data available to plot")
                return

            step_dir = create_step_directory(output_dir, session_name, step_name)

            # Determine data type and scaling
            ch_types = list(set(epochs.get_channel_types()))
            data_type = 'csd' if 'csd' in ch_types else 'eeg'
            scalings = dict(eeg=50e-6, csd=1) if data_type == 'eeg' else dict(csd=1)

            # Split channels into groups of 16
            all_channels = epochs.ch_names
            channel_groups = [all_channels[i:i + channels_per_plot]
                              for i in range(0, len(all_channels), channels_per_plot)]

            # Calculate number of epoch groups
            n_epochs = len(epochs)
            n_epoch_groups = (n_epochs + epochs_per_plot - 1) // epochs_per_plot

            # For each channel group
            for group_idx, channel_group in enumerate(channel_groups):
                # Create a temporary copy of epochs with only the current channel group
                epochs_subset = epochs.copy().pick_channels(channel_group)

                # For each group of epochs
                for epoch_idx in range(n_epoch_groups):
                    start_idx = epoch_idx * epochs_per_plot
                    end_idx = min(start_idx + epochs_per_plot, n_epochs)

                    # Create epoch subset for this group
                    current_epochs = epochs_subset[start_idx:end_idx]

                    # Plot epochs for this channel and epoch group
                    fig = current_epochs.plot(
                        scalings=scalings,
                        n_channels=len(channel_group),
                        n_epochs=end_idx - start_idx,
                        picks='all',
                        show=False,
                        block=False,
                        title=f"{step_name} ({data_type.upper()}) - Channels {group_idx * channels_per_plot + 1}-{min((group_idx + 1) * channels_per_plot, len(all_channels))} - Epochs {start_idx + 1}-{end_idx}"
                    )

                    # Save the figure
                    fig_path = os.path.join(
                        step_dir,
                        f'{data_type}_epochs_ch{group_idx * channels_per_plot + 1:03d}-{min((group_idx + 1) * channels_per_plot, len(all_channels)):03d}_ep{start_idx + 1:03d}-{end_idx:03d}.png'
                    )
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig)

                # Clear memory
                del epochs_subset



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

