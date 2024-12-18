import os
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import numpy as np
import datetime

def create_step_directory(output_dir: str, step_name: str) -> str:
    """Create a directory for a specific preprocessing step's plots."""
    step_dir = Path(output_dir) / 'preprocessing_steps' / step_name
    step_dir.mkdir(parents=True, exist_ok=True)
    return str(step_dir)

def plot_raw_segments(raw, output_dir: str, step_name: str, duration: float = 5.0, overlap: float = 0.0):
    """Plot segments of raw EEG/CSD data and save as separate PNG files."""
    step_dir = create_step_directory(output_dir, step_name)
    
    # Get data type (EEG or CSD)
    ch_types = list(set(raw.get_channel_types()))
    data_type = 'csd' if 'csd' in ch_types else 'eeg'
    
    # Calculate number of segments
    total_duration = raw.times[-1]
    step = duration - overlap
    start_times = np.arange(0, total_duration - duration, step)
    
    # Plot each segment
    for i, start in enumerate(start_times):
        with mne.viz.use_browser_backend("matplotlib"):
            fig = raw.copy().plot(duration=duration, start=start, scalings='auto', show=False)
            
            # Add title with time and data type information
            plt.suptitle(f"{step_name} ({data_type.upper()}) - Segment {i+1} ({start:.1f}s - {start+duration:.1f}s)")
            
            # Save figure
            fig.savefig(os.path.join(step_dir, f'{data_type}_segment_{i+1:03d}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)

def plot_epochs_grid(epochs, output_dir: str, step_name: str, epochs_per_plot: int = 10):
    """Plot epochs in grids and save as separate PNG files."""
    step_dir = create_step_directory(output_dir, step_name)
    
    # Get data type (EEG or CSD)
    ch_types = list(set(epochs.get_channel_types()))
    data_type = 'csd' if 'csd' in ch_types else 'eeg'
    
    # Calculate number of plots needed
    n_epochs = len(epochs)
    n_plots = (n_epochs + epochs_per_plot - 1) // epochs_per_plot
    
    # Plot epochs in groups
    for plot_idx in range(n_plots):
        start_idx = plot_idx * epochs_per_plot
        end_idx = min(start_idx + epochs_per_plot, n_epochs)
        
        with mne.viz.use_browser_backend("matplotlib"):
            fig = epochs[start_idx:end_idx].plot(scalings='auto', show=False)
            
            # Add title with data type information
            plt.suptitle(f"{step_name} ({data_type.upper()}) - Epochs {start_idx+1}-{end_idx}")
            
            # Save figure
            fig.savefig(os.path.join(step_dir, f'{data_type}_epochs_{plot_idx+1:03d}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)

def plot_single_epoch(epochs, output_dir: str, step_name: str, epoch_num: int):
    """Plot a single epoch with proper data type handling."""
    step_dir = create_step_directory(output_dir, step_name)
    
    # Get data type (EEG or CSD)
    ch_types = list(set(epochs.get_channel_types()))
    data_type = 'csd' if 'csd' in ch_types else 'eeg'
    
    with mne.viz.use_browser_backend("matplotlib"):
        fig = epochs[epoch_num].plot(scalings='auto', show=False)
        plt.suptitle(f"{step_name} ({data_type.upper()}) - Epoch {epoch_num+1}")
        fig.savefig(os.path.join(step_dir, f'single_{data_type}_epoch_{epoch_num+1:03d}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

def save_preprocessing_visualizations(processor, output_dir: str, raw_duration: float = 5.0):
    """Save visualizations at each preprocessing step."""
    # Create base directory for visualizations
    base_dir = Path(output_dir) / 'preprocessing_visualizations'
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Define preprocessing steps and their corresponding data
    steps = [
        ('1_raw', processor.raw),
        ('2_tms_artifact_removed', processor.raw),
        ('3_filtered', processor.raw),
        ('4_bad_channels_removed', processor.raw),
        ('5_ica_first', processor.raw),
        ('6_muscle_cleaned', processor.raw),
        ('7_ica_second', processor.raw),
        ('8_final', processor.raw)
    ]
    
    # Save raw data visualizations for each step
    for step_name, raw_data in steps:
        if raw_data is not None:
            plot_raw_segments(raw_data.copy(), str(base_dir), step_name, 
                            duration=raw_duration)
    
    # Save epoch visualizations if available
    if hasattr(processor, 'epochs') and processor.epochs is not None:
        plot_epochs_grid(processor.epochs, str(base_dir), '9_epochs')

def plot_with_type_check(data, output_dir: str, session_name: str, plot_type: str = 'raw'):
    """Generic plotting function that handles both EEG and CSD data types."""
    # Get data type
    ch_types = list(set(data.get_channel_types()))
    data_type = 'csd' if 'csd' in ch_types else 'eeg'
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create filename
    save_path = Path(output_dir) / f"{plot_type}_{data_type}_{session_name}_{timestamp}.png"
    
    with mne.viz.use_browser_backend("matplotlib"):
        if plot_type == 'raw':
            fig = data.copy().plot(scalings='auto', show=False)
        else:  # epochs
            fig = data.copy().plot(scalings='auto', show=False)
            
        for ax in fig.get_axes():
            if hasattr(ax, 'invert_yaxis'):
                ax.invert_yaxis()
                
        fig.savefig(save_path)
        plt.close(fig)