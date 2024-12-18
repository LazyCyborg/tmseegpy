import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import mne
from scipy import stats
from scipy import signal
from typing import Dict, Tuple, List, Optional

# Define standard TEP components based on literature
TEP_CRITERIA = {
    'N15-P30': {
        'latency_range': (15, 40),  # ms
        'amplitude_range': (-15, 15),  # μV
        'regions': ['FC1', 'FC2', 'C1', 'Cz', 'C2'],
        'description': 'Early component reflecting direct cortical activation',
        'polarity': 'negative'  # Added polarity information
    },
    'N45': {
        'latency_range': (40, 50),
        'amplitude_range': (-10, 10),
        'regions': ['FC1', 'FCz', 'FC2'],
        'description': 'GABA-A mediated inhibition',
        'polarity': 'negative'
    },
    'P60': {
        'latency_range': (55, 70),
        'amplitude_range': (-8, 8),
        'regions': ['FC1', 'FCz', 'FC2', 'C1', 'Cz', 'C2'],
        'description': 'GABA-B mediated inhibition',
        'polarity': 'positive'
    },
    'N100': {
        'latency_range': (85, 140),
        'amplitude_range': (-10, 10),
        'regions': ['FC1', 'FCz', 'FC2', 'C1', 'Cz', 'C2'],
        'description': 'Long-lasting inhibition',
        'polarity': 'negative'
    },
    'P180': {
        'latency_range': (150, 250),
        'amplitude_range': (-8, 8),
        'regions': ['Central'],  # General region
        'description': 'Network reactivation',
        'polarity': 'positive'
    }
}

def get_channel_indices(evoked: mne.Evoked, channel_names: List[str]) -> List[int]:
    """Get indices of channels that exist in the data."""
    available_channels = []
    for ch in channel_names:
        try:
            idx = evoked.ch_names.index(ch)
            available_channels.append(idx)
        except ValueError:
            continue
    return available_channels

def validate_tep_quality(evoked: mne.Evoked, 
                        baseline_window: Tuple[float, float] = (-100, -2)) -> Dict:
    """Validate TEP quality based on established criteria"""
    quality_checks = {
        'baseline': {
            'criterion': 'Baseline period should be stable (std < 1 μV)',
            'threshold': 1.0,
            'pass': False
        },
        'snr': {
            'criterion': 'SNR should be > 3 for reliable components',
            'threshold': 3.0,
            'pass': False
        },
        'artifact_recovery': {
            'criterion': 'Signal should return to baseline by 8-10ms',
            'threshold': 10,
            'pass': False
        }
    }
    
    data = evoked.get_data() * 1e6  # Convert to μV
    times = evoked.times * 1000     # Convert to ms
    
    # Check baseline stability
    baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    baseline_std = np.std(data[:, baseline_mask])
    quality_checks['baseline']['pass'] = baseline_std < quality_checks['baseline']['threshold']
    quality_checks['baseline']['value'] = baseline_std
    
    # Calculate SNR
    signal = data[:, times > 10]
    noise = data[:, baseline_mask]
    snr = np.std(signal) / np.std(noise)
    quality_checks['snr']['pass'] = snr > quality_checks['snr']['threshold']
    quality_checks['snr']['value'] = snr
    
    # Check artifact recovery
    early_recovery = data[:, (times > 8) & (times < 12)]
    recovery_amplitude = np.max(np.abs(np.mean(early_recovery, axis=0)))
    quality_checks['artifact_recovery']['pass'] = recovery_amplitude < 15
    quality_checks['artifact_recovery']['value'] = recovery_amplitude
    
    return quality_checks

def validate_tep_components(evoked: mne.Evoked, 
                          criteria: Dict = TEP_CRITERIA) -> Dict:
    """
    Validate each TEP component against established criteria, considering specific channels
    """
    validation_results = {}
    times = evoked.times * 1000
    data = evoked.get_data() * 1e6
    
    for component, crit in criteria.items():
        # Get relevant channel indices
        if crit['regions'][0] == 'Central':
            # For 'Central' region, use a broader set of channels
            central_channels = ['FCz', 'Cz', 'FC1', 'FC2', 'C1', 'C2']
            channel_idx = get_channel_indices(evoked, central_channels)
        else:
            channel_idx = get_channel_indices(evoked, crit['regions'])
        
        if not channel_idx:
            # If no specified channels are found, skip this component
            validation_results[component] = {
                'error': 'No matching channels found',
                'channels_searched': crit['regions']
            }
            continue
            
        # Get data for relevant channels and time window
        time_mask = (times >= crit['latency_range'][0]) & (times <= crit['latency_range'][1])
        window_data = data[channel_idx][:, time_mask]
        window_times = times[time_mask]
        
        # Find peak in window based on polarity
        mean_data = np.mean(window_data, axis=0)
        if crit['polarity'] == 'negative':
            peak_idx = np.argmin(mean_data)
            peak_func = np.min
        else:
            peak_idx = np.argmax(mean_data)
            peak_func = np.max
            
        peak_latency = window_times[peak_idx]
        peak_amplitude = mean_data[peak_idx]
        
        # Find which channel shows the strongest response
        channel_peaks = [peak_func(ch_data) for ch_data in window_data]
        max_channel_idx = np.argmax(np.abs(channel_peaks))
        max_channel = evoked.ch_names[channel_idx[max_channel_idx]]
        
        # Calculate SNR for component using baseline
        baseline_mask = times < 0
        baseline_data = data[channel_idx][:, baseline_mask]
        component_snr = np.abs(peak_amplitude) / np.std(baseline_data)
        
        # Spatial consistency check (should be similar across specified channels)
        spatial_consistency = np.std(channel_peaks) / np.mean(np.abs(channel_peaks))
        
        validation_results[component] = {
            'latency': peak_latency,
            'amplitude': peak_amplitude,
            'snr': component_snr,
            'within_latency': (crit['latency_range'][0] <= peak_latency <= crit['latency_range'][1]),
            'within_amplitude': (crit['amplitude_range'][0] <= peak_amplitude <= crit['amplitude_range'][1]),
            'acceptable_snr': component_snr > 3,
            'max_channel': max_channel,
            'spatial_consistency': spatial_consistency,
            'channels_used': [evoked.ch_names[idx] for idx in channel_idx],
            'channel_amplitudes': dict(zip([evoked.ch_names[idx] for idx in channel_idx], channel_peaks))
        }
    
    return validation_results

def plot_tep_validation(evoked: mne.Evoked, 
                    validation_results: Dict,
                    output_dir: str,
                    session_name: str,
                    baseline_window: Tuple[float, float] = (-400, -50),
                    response_window: Tuple[float, float] = (0, 299)):
    """
    Create comprehensive TEP validation plots with channel-specific analysis.
    Automatically handles both EEG and CSD data types.
    
    Parameters
    ----------
    evoked : mne.Evoked
        The evoked data to plot
    validation_results : Dict
        Dictionary containing validation results for each component
    output_dir : str
        Directory to save output
    session_name : str
        Name of the session
    baseline_window : Tuple[float, float]
        Time window for baseline in milliseconds
    response_window : Tuple[float, float]
        Time window for response in milliseconds
    """
    # Determine the channel type (EEG or CSD)
    ch_types = list(set(evoked.get_channel_types()))
    if 'csd' in ch_types:
        plot_ch_type = 'csd'
        amplitude_unit = 'μV/m²'
    else:
        plot_ch_type = 'eeg'
        amplitude_unit = 'μV'
    
    times = evoked.times * 1000
    data = evoked.get_data() * 1e6
    
    # Count number of valid components for grid sizing
    n_components = len([r for r in validation_results.values() 
                    if 'latency' in r and 'error' not in r])
    
    # Calculate required columns (2 columns per topomap - map and colorbar)
    n_cols = max(6, 3 + n_components * 2)  # At least 6 columns, more if needed
    
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(2, n_cols)
    
    # Plot 1: Average TEP with components (full width)
    ax1 = fig.add_subplot(gs[0, :])
    mean_tep = np.mean(data, axis=0)
    plt.plot(times, mean_tep, 'b-', label='Global Average')
    
    # Plot component-specific channels
    for component, results in validation_results.items():
        if 'error' in results:
            continue
            
        color = 'g' if results['within_amplitude'] and results['within_latency'] else 'r'
        if 'channels_used' in results:
            channel_indices = [evoked.ch_names.index(ch) for ch in results['channels_used']]
            component_data = np.mean(data[channel_indices], axis=0)
            plt.plot(times, component_data, '--', alpha=0.5, 
                    label=f"{component} channels")
        
        plt.axvline(results['latency'], color=color, alpha=0.3)
        plt.plot(results['latency'], results['amplitude'], 'o', color=color, 
                label=f"{component} peak")
    
    plt.axvspan(baseline_window[0], baseline_window[1], color='gray', alpha=0.2, label='Baseline')
    plt.axvspan(response_window[0], response_window[1], color='green', alpha=0.2, label='Response')
    plt.title(f'TEP Components ({plot_ch_type.upper()} data)', fontsize=12, pad=20)
    plt.xlabel('Time (ms)')
    plt.ylabel(f'Amplitude ({amplitude_unit})')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    
    # Plot 2: Channel contributions
    ax3 = fig.add_subplot(gs[1, :3])
    component_names = []
    channel_names = []
    amplitudes = []
    
    for component, results in validation_results.items():
        if 'channel_amplitudes' not in results:
            continue
        for channel, amp in results['channel_amplitudes'].items():
            component_names.append(component)
            channel_names.append(channel)
            amplitudes.append(amp)
    
    if amplitudes:
        plt.scatter(component_names, amplitudes, c='b', alpha=0.6)
        for i, ch in enumerate(channel_names):
            plt.annotate(ch, (component_names[i], amplitudes[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize='x-small')
    plt.title('Channel Contributions to Components', fontsize=12, pad=20)
    plt.ylabel(f'Amplitude ({amplitude_unit})')
    plt.xticks(rotation=45)
    
    # Plot 3: Topomaps for each component
    if validation_results:
        peak_times = []
        component_names = []
        for component, results in validation_results.items():
            if 'latency' in results and 'error' not in results:
                peak_times.append(results['latency'] / 1000.0)  # Convert to seconds
                component_names.append(component)
        
        if peak_times:
            for idx, (time, name) in enumerate(zip(peak_times, component_names)):
                # Create two axes for each topomap: one for map and one for colorbar
                ax_topo = fig.add_subplot(gs[1, 3 + idx * 2])  # Topomap
                ax_cbar = fig.add_subplot(gs[1, 3 + idx * 2 + 1])  # Colorbar
                
                try:
                    evoked.plot_topomap(times=time, ch_type=plot_ch_type, 
                                    axes=[ax_topo, ax_cbar], 
                                    show=False,
                                    time_format=f'{name}\n{time*1000:.0f} ms')
                except ValueError as e:
                    print(f"Warning: Could not plot topomap for {name} at {time*1000:.0f}ms: {str(e)}")
                    # Add text explaining the error
                    ax_topo.text(0.5, 0.5, f"Could not plot\n{plot_ch_type} topomap",
                            ha='center', va='center')
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(os.path.join(output_dir, f'{session_name}_tep_validation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_tms_components(evoked, ylim=None, prominence=0.01, save=False, output_dir=None, session_name=None):
    """
    Plot Global Field Power (GFP), detect specific TMS-EEG components, and show their topomaps.
    
    Parameters:
    -----------
    evoked : mne.Evoked
        Evoked TMS-EEG data
    ylim : tuple, optional
        Y-axis limits for the plot (min, max)
    prominence : float, optional
        Minimum prominence for peak detection as fraction of max GFP
    save : bool, optional
        Whether to save the plot
    output_dir : str, optional
        Directory to save output if save=True
    session_name : str, optional
        Name of the session for saving
        
    Returns:
    --------
    gfp_uv : ndarray
        Global Field Power values in microvolts
    components : dict
        Dictionary containing the detected component peaks and their properties
    """
    # Create figure with four sections
    # Increase number of columns to 5 to accommodate all possible components
    fig = plt.figure(figsize=(20, 16))
    gs = plt.GridSpec(4, 5, height_ratios=[3, 3, 4, 1], hspace=0.3)
    
    # Plot average with GFP in top subplot
    ax_avg = fig.add_subplot(gs[0, :])
    evoked.plot(gfp=True, axes=ax_avg, show=False)
    ax_avg.set_xlim(-0.1, 0.4)
    ax_avg.set_title("Average with GFP")
    
    # Determine the channel type
    ch_types = list(set(evoked.get_channel_types()))
    plot_ch_type = 'csd' if 'csd' in ch_types else 'eeg'
    
    # Compute Global Field Power (GFP)
    times = evoked.times * 1e3
    data = evoked.get_data()

    ## Medelvädert av alla kanalerna? 
    #gfp = np.sum(x**2, axis=0)
    gfp_uv = np.std(data, axis=0)
    #gfp_uv = mne.baseline.rescale(gfp, times, baseline=(None, 0))
    
    # Use TEP criteria from the validation code
    components = {}
    for component, criteria in TEP_CRITERIA.items():
        t_min, t_max = criteria['latency_range']
        
        # Convert time window to indices
        idx_min = np.searchsorted(times, t_min)
        idx_max = np.searchsorted(times, t_max)
        
        # Select data within time window
        window_data = gfp_uv[idx_min:idx_max]
        window_times = times[idx_min:idx_max]
        
        # Adjust parameters based on component polarity
        if criteria['polarity'] == 'negative':
            peak_idx, properties = signal.find_peaks(
                -window_data,
                distance=len(window_data),
                prominence=prominence * np.max(gfp_uv)
            )
        else:
            peak_idx, properties = signal.find_peaks(
                window_data,
                distance=len(window_data),
                prominence=prominence * np.max(gfp_uv)
            )
        
        if len(peak_idx) > 0:
            max_prom_idx = np.argmax(properties['prominences'])
            global_idx = peak_idx[max_prom_idx] + idx_min
            components[component] = {
                'time': times[global_idx],
                'amplitude': gfp_uv[global_idx],
                'index': global_idx,
                'description': criteria['description']
            }
    
    # Colors for components
    colors = {
        'N15-P30': 'red',
        'N45': 'green',
        'P60': 'purple',
        'N100': 'orange',
        'P180': 'brown'
    }
    
    # Main GFP plot
    ax = fig.add_subplot(gs[1, :])
    
    if ylim is None:
        ylim = (0, 1.5 * np.max(gfp_uv))
    
    # Plot shaded regions for time windows
    for component, criteria in TEP_CRITERIA.items():
        t_min, t_max = criteria['latency_range']
        ax.axvspan(t_min, t_max, color=colors[component], alpha=0.1)
        
    # Plot GFP
    ax.plot(times, gfp_uv, label='GFP', color='b', linewidth=1)
    
    # Plot detected components
    for component, data in components.items():
        ax.plot(data['time'], data['amplitude'], 'o', 
                color=colors[component], label=component, markersize=8)
    
    # Add annotations for each component
    for component, data in components.items():
        ax.annotate(
            f"{component}",
            xy=(data['time'], data['amplitude']),
            xytext=(10, 10),
            textcoords='offset points',
            color=colors[component],
            fontweight='bold',
            fontsize=8,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )
    
    # Plot settings for main plot
    ax.axvline(x=0, color='r', linestyle='--', linewidth=1, label='TMS Pulse')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('GFP (µV)')
    ax.set_title(f"TMS-EEG Components in Global Field Power")
    ax.set_ylim(ylim)
    ax.set_xlim(-100, 400)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Plot topomaps for detected components
    # Create a new subplot for each component, ensuring we don't exceed grid dimensions
    n_components = len(components)
    topomap_cols = min(5, n_components)  # Use at most 5 columns
    
    for idx, (component, data) in enumerate(components.items()):
        if idx >= 5:  # Skip if we have more than 5 components
            print(f"Warning: Topomap for {component} not shown due to space constraints")
            continue
            
        ax_topo = fig.add_subplot(gs[2, idx])
        try:
            evoked.plot_topomap(times=data['time']/1000.0,
                               ch_type=plot_ch_type,
                               axes=ax_topo,
                               colorbar=False,
                               show=False,
                               time_format=f'{component}\n{data["time"]:.0f} ms')
        except ValueError as e:
            print(f"Warning: Could not plot topomap for {component}: {str(e)}")
            ax_topo.text(0.5, 0.5, f"Could not plot\n{component} topomap",
                        ha='center', va='center')
    
    # Window information subplot
    ax_windows = fig.add_subplot(gs[3, :])
    
    # Create table data
    table_data = []
    for component, criteria in TEP_CRITERIA.items():
        t_min, t_max = criteria['latency_range']
        if component in components:
            peak_time = f"{components[component]['time']:.1f}"
            peak_amplitude = f"{components[component]['amplitude']:.2e}"
            row = [component, f"{t_min}-{t_max} ms", f"{peak_time} ms", 
                   f"{peak_amplitude} µV", criteria['description']]
        else:
            row = [component, f"{t_min}-{t_max} ms", "Not detected", "---", 
                   criteria['description']]
        table_data.append(row)
    
    # Create the table
    table = ax_windows.table(
        cellText=table_data,
        colLabels=['Component', 'Search Window', 'Peak Latency', 'Peak Amplitude', 'Description'],
        loc='center',
        cellLoc='center',
        colWidths=[0.1, 0.1, 0.1, 0.15, 0.25]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Color the table cells
    for idx, component in enumerate(TEP_CRITERIA.keys(), start=1):
        for col in range(5):
            table[idx, col].set_facecolor(colors[component])
            table[idx, col].set_alpha(0.2)
    
    # Hide window subplot axes
    ax_windows.axis('off')
    
    plt.tight_layout()
    
    if save and output_dir and session_name: 
        plt.savefig(os.path.join(output_dir, f'{session_name}_tep_analysis.png'),
                   dpi=300, bbox_inches='tight')
    
    return gfp_uv, components

def generate_validation_report(evoked: mne.Evoked,
                             quality_results: Dict,
                             component_results: Dict,
                             output_dir: str,
                             session_name: str):
    """Generate comprehensive validation report with channel-specific information"""
    with open(f"{output_dir}/tep_validation_{session_name}.txt", 'w') as f:
        f.write("TEP Validation Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Quality metrics
        f.write("1. Signal Quality Metrics\n")
        f.write("-" * 30 + "\n")
        for metric, results in quality_results.items():
            f.write(f"\n{metric.title()}:\n")
            f.write(f"  Criterion: {results['criterion']}\n")
            f.write(f"  Value: {results['value']:.2f}\n")
            f.write(f"  Status: {'PASS' if results['pass'] else 'FAIL'}\n")
        
        # Component validation with channel information
        f.write("\n2. TEP Components Analysis\n")
        f.write("-" * 30 + "\n")
        for component, results in component_results.items():
            f.write(f"\n{component}:\n")
            if 'error' in results:
                f.write(f"  Error: {results['error']}\n")
                f.write(f"  Searched channels: {', '.join(results['channels_searched'])}\n")
                continue
                
            f.write(f"  Latency: {results['latency']:.1f} ms ")
            f.write(f"({'Valid' if results['within_latency'] else 'Invalid'})\n")
            f.write(f"  Amplitude: {results['amplitude']:.2f} μV ")
            f.write(f"({'Valid' if results['within_amplitude'] else 'Invalid'})\n")
            f.write(f"  SNR: {results['snr']:.2f} ")
            f.write(f"({'Acceptable' if results['acceptable_snr'] else 'Poor'})\n")
            f.write(f"  Maximum response channel: {results['max_channel']}\n")
            f.write(f"  Spatial consistency (lower is better): {results['spatial_consistency']:.3f}\n")
            f.write("  Channel amplitudes:\n")
            for ch, amp in results['channel_amplitudes'].items():
                f.write(f"    {ch}: {amp:.2f} μV\n")
        
        # Overall assessment
        f.write("\n3. Overall Assessment\n")
        f.write("-" * 30 + "\n")
        
        quality_score = sum([r['pass'] for r in quality_results.values()]) / len(quality_results)
        
# Calculate component score considering channel-specific criteria
        valid_components = [c for c in component_results.values() if 'error' not in c]
        if valid_components:
            component_scores = []
            for comp in valid_components:
                # Consider multiple validation criteria
                criteria_met = [
                    comp['within_latency'],
                    comp['within_amplitude'],
                    comp['acceptable_snr'],
                    comp['spatial_consistency'] < 0.5  # Add threshold for spatial consistency
                ]
                component_scores.append(sum(criteria_met) / len(criteria_met))
            component_score = np.mean(component_scores)
        else:
            component_score = 0.0
        
        f.write(f"\nSignal Quality Score: {quality_score*100:.1f}%\n")
        f.write(f"Component Quality Score: {component_score*100:.1f}%\n")
        
        # Channel utilization assessment
        available_channels = len(evoked.ch_names)
        used_channels = set()
        for results in component_results.values():
            if 'channels_used' in results:
                used_channels.update(results['channels_used'])
        channel_utilization = len(used_channels) / available_channels
        f.write(f"Channel Utilization: {channel_utilization*100:.1f}%\n")
        
        # Warnings and recommendations
        f.write("\nWarnings and Recommendations:\n")
        if quality_score < 0.7:
            f.write("- Signal quality does not meet minimum criteria\n")
            if quality_results['baseline']['pass'] == False:
                f.write("  * Baseline period is unstable, consider longer baseline\n")
            if quality_results['snr']['pass'] == False:
                f.write("  * Poor SNR, consider increasing number of trials or improving noise reduction\n")
            if quality_results['artifact_recovery']['pass'] == False:
                f.write("  * Poor artifact recovery, review TMS artifact removal settings\n")
        
        if component_score < 0.7:
            f.write("- TEP components do not match expected characteristics\n")
            for component, results in component_results.items():
                if 'error' in results:
                    f.write(f"  * {component}: Required channels not found: {', '.join(results['channels_searched'])}\n")
                elif not results['within_latency']:
                    f.write(f"  * {component}: Latency outside expected range\n")
                elif not results['within_amplitude']:
                    f.write(f"  * {component}: Amplitude outside expected range\n")
                elif not results['acceptable_snr']:
                    f.write(f"  * {component}: Poor SNR\n")
        
        if channel_utilization < 0.5:
            f.write("- Low channel utilization, check electrode placement and contact quality\n")
        
        # Add references and methodology notes
        f.write("\nMethodology Notes:\n")
        f.write("- TEP criteria based on established literature\n")
        f.write("- SNR threshold set at 3:1\n")
        f.write("- Spatial consistency measured as coefficient of variation across channels\n")
        f.write("- Component validation includes latency, amplitude, SNR, and spatial distribution\n")

def validate_teps(evoked: mne.Evoked,
                 output_dir: str,
                 session_name: str,
                 baseline_window: Tuple[float, float] = (-400, -50),
                 response_window: Tuple[float, float] = (0, 299),
                 save_outputs: bool = True) -> Tuple[Dict, Dict]:
    """
    Main function to run TEP validation with comprehensive channel analysis
    
    Parameters
    ----------
    evoked : mne.Evoked
        The evoked response to validate
    output_dir : str
        Directory to save outputs
    session_name : str
        Name of the session
    baseline_window : tuple
        Start and end time of baseline period in ms
    response_window : tuple
        Start and end time of response period in ms
    save_outputs : bool
        Whether to save validation reports and plots
        
    Returns
    -------
    quality_results : dict
        Results of signal quality checks
    component_results : dict
        Results of component-specific validation
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run validations with specified baseline
    quality_results = validate_tep_quality(evoked, baseline_window=baseline_window)
    component_results = validate_tep_components(evoked)
    
    if save_outputs:
        # Generate visualizations and report
        plot_tep_validation(evoked, component_results, output_dir, session_name,
                          baseline_window=baseline_window,
                          response_window=response_window)
        generate_validation_report(evoked, quality_results, component_results,
                                 output_dir, session_name)
    
    return quality_results, component_results