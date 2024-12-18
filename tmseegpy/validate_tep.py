# validate_tep.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import mne
from scipy import signal
from typing import Dict, Tuple, List, Optional

# Define key TEP components we want to identify
TEP_COMPONENTS = {
    'N15-P30': {'time': (15, 40), 'polarity': 'negative'},
    'N45': {'time': (40, 50), 'polarity': 'negative'},
    'P60': {'time': (55, 70), 'polarity': 'positive'},
    'N100': {'time': (85, 140), 'polarity': 'negative'},
    'P180': {'time': (150, 250), 'polarity': 'positive'}
}

def analyze_tep_components(evoked: mne.Evoked, prominence: float = 0.01):
    """
    Analyze TEP components using GFP and return their properties.
    
    Parameters
    ----------
    evoked : mne.Evoked
        The evoked response to analyze
    prominence : float
        Peak detection sensitivity
        
    Returns
    -------
    dict
        Detected components and their properties
    """
    # Get data type (EEG or CSD)
    ch_types = list(set(evoked.get_channel_types()))
    data_type = 'csd' if 'csd' in ch_types else 'eeg'
    
    # Calculate GFP
    times = evoked.times * 1000  # Convert to ms
    data = evoked.get_data()
    gfp = np.std(data, axis=0)
    
    # Find components
    components = {}
    for name, criteria in TEP_COMPONENTS.items():
        t_min, t_max = criteria['time']
        
        # Get data in time window
        mask = (times >= t_min) & (times <= t_max)
        window_gfp = gfp[mask]
        window_times = times[mask]
        
        # Find peak based on polarity
        if criteria['polarity'] == 'negative':
            peak_idx = np.argmin(window_gfp)
        else:
            peak_idx = np.argmax(window_gfp)
            
        # Store component properties
        components[name] = {
            'time': window_times[peak_idx],
            'amplitude': window_gfp[peak_idx],
            'data_type': data_type
        }
        
    return components, gfp, times

def plot_tep_analysis(evoked: mne.Evoked, 
                     output_dir: str, 
                     session_name: str,
                     prominence: float = 0.01):
    """
    Create simplified TEP analysis plot with GFP and topomaps.
    
    Parameters
    ----------
    evoked : mne.Evoked
        The evoked response to analyze
    output_dir : str
        Directory to save output
    session_name : str
        Name of the session
    prominence : float
        Peak detection sensitivity
    """
    components, gfp, times = analyze_tep_components(evoked, prominence)
    
    # Create figure with 2 rows: GFP plot and topomaps
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Plot GFP
    ax_gfp = fig.add_subplot(gs[0])
    ax_gfp.plot(times, gfp, 'b-', label='GFP')
    
    # Plot detected components
    colors = ['r', 'g', 'b', 'm', 'c']
    for (name, comp), color in zip(components.items(), colors):
        ax_gfp.plot(comp['time'], comp['amplitude'], 'o', color=color, 
                   label=f"{name} ({comp['time']:.0f} ms)")
        ax_gfp.axvline(comp['time'], color=color, alpha=0.2)
    
    ax_gfp.set_xlabel('Time (ms)')
    ax_gfp.set_ylabel(f"GFP ({'µV/m²' if comp['data_type']=='csd' else 'µV'})")
    ax_gfp.set_title('Global Field Power with TEP Components')
    ax_gfp.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_gfp.grid(True, alpha=0.3)
    ax_gfp.set_xlim(-100, 300)
    
    # Plot topomaps
    ax_topos = plt.subplot(gs[1])
    n_components = len(components)
    ax_topos.set_axis_off()
    
    for idx, (name, comp) in enumerate(components.items()):
        # Create subplot for each topomap
        ax = plt.subplot(1, n_components, idx + 1)
        try:
            evoked.plot_topomap(times=comp['time']/1000.0,  # Convert back to seconds
                              ch_type=comp['data_type'],
                              axes=ax,
                              show=False,
                              time_format=f'{name}\n{comp["time"]:.0f} ms')
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not plot\n{name}", ha='center', va='center')
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(os.path.join(output_dir, f'{session_name}_tep_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Generate simple report
    report_path = os.path.join(output_dir, f'{session_name}_tep_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"TEP Analysis Report for {session_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Data type: {components['N15-P30']['data_type'].upper()}\n\n")
        
        f.write("Component Latencies:\n")
        f.write("-" * 20 + "\n")
        for name, comp in components.items():
            f.write(f"{name}: {comp['time']:.1f} ms\n")
            
        f.write("\nComponent Amplitudes:\n")
        f.write("-" * 20 + "\n")
        for name, comp in components.items():
            f.write(f"{name}: {comp['amplitude']:.2f} "
                   f"{'µV/m²' if comp['data_type']=='csd' else 'µV'}\n")
    
    return components

def generate_validation_summary(components: Dict, 
                              output_dir: str, 
                              session_name: str):
    """Generate a simple validation summary."""
    expected_times = {
        'N15-P30': (15, 40),
        'N45': (40, 50),
        'P60': (55, 70),
        'N100': (85, 140),
        'P180': (150, 250)
    }
    
    summary_path = os.path.join(output_dir, f'{session_name}_validation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("TEP Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for name, comp in components.items():
            expected_range = expected_times[name]
            is_valid = expected_range[0] <= comp['time'] <= expected_range[1]
            
            f.write(f"{name}:\n")
            f.write(f"  Latency: {comp['time']:.1f} ms ")
            f.write(f"({'valid' if is_valid else 'outside expected range'})\n")
            f.write(f"  Expected range: {expected_range[0]}-{expected_range[1]} ms\n\n")