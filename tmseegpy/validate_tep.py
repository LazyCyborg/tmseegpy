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

def find_peak_in_window(times, gfp, t_min, t_max, polarity='positive', prominence=0.1):
    """Find peak in a specific time window."""
    # Get data in time window
    mask = (times >= t_min) & (times <= t_max)
    window_gfp = gfp[mask]
    window_times = times[mask]
    
    # Invert signal for negative peaks
    if polarity == 'negative':
        window_gfp = -window_gfp
    
    # Find peaks
    peaks, properties = signal.find_peaks(window_gfp, prominence=prominence)
    
    if len(peaks) > 0:
        # Get highest prominence peak
        best_peak = peaks[np.argmax(properties['prominences'])]
        peak_time = window_times[best_peak]
        peak_amplitude = window_gfp[best_peak] * (-1 if polarity == 'negative' else 1)
        return peak_time, peak_amplitude
    return None, None

def analyze_tep_components(evoked: mne.Evoked, prominence: float = 0.1):
    """Analyze TEP components using GFP and find peaks."""
    # Get data type
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
        peak_time, peak_amplitude = find_peak_in_window(
            times, gfp, t_min, t_max, 
            criteria['polarity'], 
            prominence=prominence * np.max(gfp)
        )
        
        if peak_time is not None:
            components[name] = {
                'time': peak_time,
                'amplitude': peak_amplitude,
                'data_type': data_type
            }
    
    return components, gfp, times

def plot_tep_analysis(evoked: mne.Evoked, 
                     output_dir: str, 
                     session_name: str,
                     prominence: float = 0.1):
    """Create comprehensive TEP analysis plot with butterfly plot, GFP and topomaps."""
    components, gfp, times = analyze_tep_components(evoked, prominence)
    
    # Create figure with modified layout
    fig = plt.figure(figsize=(15, 10))
    
    # Create grid with small bottom area for topomaps
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    # Create subplot for main plots (butterfly and GFP)
    gs_main = gs[0].subgridspec(2, 1, height_ratios=[1.5, 1], hspace=0.3)
    
    # Plot butterfly plot with GFP
    ax_butterfly = fig.add_subplot(gs_main[0])
    evoked.plot(gfp=True, axes=ax_butterfly, show=False)
    ax_butterfly.set_title('TEP Butterfly Plot with GFP')
    
    # Plot GFP with detected components
    ax_gfp = fig.add_subplot(gs_main[1])
    ax_gfp.plot(times, gfp, 'b-', label='GFP')
    
    # Plot detected components
    colors = ['r', 'g', 'b', 'm', 'c']
    for (name, comp), color in zip(components.items(), colors):
        ax_gfp.plot(comp['time'], abs(comp['amplitude']), 'o', color=color, 
                   label=f"{name} ({comp['time']:.0f} ms)")
        ax_gfp.axvline(comp['time'], color=color, alpha=0.2)
    
    ax_gfp.set_xlabel('Time (ms)')
    ax_gfp.set_ylabel(f"GFP ({'µV/m²' if comp['data_type']=='csd' else 'µV'})")
    ax_gfp.set_title('Global Field Power with TEP Components')
    ax_gfp.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_gfp.grid(True, alpha=0.3)
    ax_gfp.set_xlim(-100, 400)
    
    # Create subplot for topomaps
    gs_topos = gs[1].subgridspec(1, len(components), wspace=0.3)
    
    for idx, (name, comp) in enumerate(components.items()):
        ax = fig.add_subplot(gs_topos[idx])
        try:
            evoked.plot_topomap(times=comp['time']/1000.0,  # Convert back to seconds
                              ch_type=comp['data_type'],
                              axes=ax,
                              show=False,
                              time_format=f'{name}\n{comp["time"]:.0f} ms',
                              colorbar=False)  # Hide individual colorbars
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not plot\n{name}", ha='center', va='center')
    
    # Add a single colorbar at the right of the topomaps
    cax = fig.add_axes([0.92, 0.11, 0.02, 0.15])  # [left, bottom, width, height]
    plt.colorbar(ax.images[-1], cax=cax)
    
    plt.suptitle(f'TEP Analysis - {session_name}', y=0.95)
    
    # Adjust the layout to prevent overlap
    plt.subplots_adjust(right=0.85)
    
    # Save figure
    fig.savefig(os.path.join(output_dir, f'{session_name}_tep_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return components

def generate_validation_summary(components: Dict, 
                              output_dir: str, 
                              session_name: str):
    """Generate a validation summary."""
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
