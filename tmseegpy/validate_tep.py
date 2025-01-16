import numpy as np
import matplotlib.pyplot as plt
import mne
from typing import Dict, Tuple, List, Optional, Union
import os

DEFAULT_TEP_COMPONENTS = {
    'N15': {'time': (10, 20), 'polarity': 'negative', 'peak': 15},
    'P30': {'time': (20, 40), 'polarity': 'positive', 'peak': 30},
    'N45': {'time': (40, 55), 'polarity': 'negative', 'peak': 45},
    'P60': {'time': (50, 70), 'polarity': 'positive', 'peak': 60},
    'N100': {'time': (70, 150), 'polarity': 'negative', 'peak': 100},
    'P180': {'time': (150, 240), 'polarity': 'positive', 'peak': 180},
    'N280': {'time': (240, 350), 'polarity': 'negative', 'peak': 280}
}

def find_peaks_tesa_style(data: np.ndarray,
                          samples: int = 5,
                          polarity: str = 'positive') -> List[int]:
    """
    Find peaks using exact TESA logic from tesa_tepextract.

    Parameters
    ----------
    data : np.ndarray
        Signal to analyze
    samples : int
        Number of samples to check on each side
    polarity : str
        'positive' or 'negative'

    Returns
    -------
    List[int]
        Indices of detected peaks
    """
    peak_indices = []

    # Loop through each potential peak point (avoiding edges)
    for b in range(samples, len(data) - samples):
        # Initialize arrays for comparisons exactly as TESA does
        t_plus = np.zeros(samples)
        t_minus = np.zeros(samples)

        # Calculate differences exactly as TESA does
        for c in range(samples):
            t_plus[c] = data[b] - data[b + c + 1]  # Compare with later points
            t_minus[c] = data[b] - data[b - c - 1]  # Compare with earlier points

        # Convert to logical arrays as TESA does
        if polarity == 'positive':
            t_plus_log = t_plus > 0
            t_minus_log = t_minus > 0
        else:  # negative
            t_plus_log = t_plus < 0
            t_minus_log = t_minus < 0

        # Check if all comparisons match peak criteria
        if np.sum(t_plus_log) + np.sum(t_minus_log) == samples * 2:
            peak_indices.append(b)

    return peak_indices


def extract_tep(data: np.ndarray,
                times: np.ndarray,
                time_window: Tuple[float, float],
                polarity: str = 'positive',
                samples: int = 5,
                method: str = 'largest',
                peak_time: Optional[float] = None) -> Dict:
    """
    Extract TEP peaks using exact TESA logic.

    Parameters
    ----------
    data : np.ndarray
        Signal to analyze
    times : np.ndarray
        Time points in milliseconds
    time_window : tuple
        (min_time, max_time) in milliseconds
    polarity : str
        'positive' or 'negative'
    samples : int
        Number of samples for peak detection
    method : str
        Peak selection method ('largest' or 'centre')
    peak_time : float, optional
        Target peak time for centre method

    Returns
    -------
    dict
        Peak information matching TESA format
    """
    # Find time window indices
    t_min, t_max = time_window
    win_mask = (times >= t_min) & (times <= t_max)
    win_times = times[win_mask]
    win_data = data[win_mask]

    # Find peaks using TESA's method
    peak_indices = find_peaks_tesa_style(win_data, samples, polarity)

    # Handle peak selection based on number found
    if len(peak_indices) == 0:
        # No peaks found - return amplitude at target latency (TESA behavior)
        if peak_time is None:
            peak_time = (t_min + t_max) / 2
        target_idx = np.argmin(np.abs(times - peak_time))
        return {
            'found': 'no',
            'lat': float('nan'),
            'time': float('nan'),
            'amp': data[target_idx],
            'amplitude': data[target_idx]
        }

    elif len(peak_indices) == 1:
        # Single peak found
        peak_idx = peak_indices[0]
        global_idx = np.where(win_mask)[0][peak_idx]
        return {
            'found': 'yes',
            'lat': times[global_idx],
            'time': times[global_idx],
            'amp': data[global_idx],
            'amplitude': data[global_idx]
        }

    else:
        # Multiple peaks found - use method to select (TESA logic)
        if method == 'largest':
            if polarity == 'positive':
                best_idx = peak_indices[np.argmax(win_data[peak_indices])]
            else:
                best_idx = peak_indices[np.argmin(win_data[peak_indices])]
        else:  # 'centre' method
            if peak_time is None:
                peak_time = (t_min + t_max) / 2
            diffs = np.abs(win_times[peak_indices] - peak_time)
            best_idx = peak_indices[np.argmin(diffs)]

        global_idx = np.where(win_mask)[0][best_idx]
        return {
            'found': 'yes',
            'lat': times[global_idx],
            'time': times[global_idx],
            'amp': data[global_idx],
            'amplitude': data[global_idx]
        }


# Default component definitions matching TESA
DEFAULT_TEP_COMPONENTS = {
    'N15': {'time': (12, 18), 'polarity': 'negative', 'peak': 15},
    'P30': {'time': (25, 35), 'polarity': 'positive', 'peak': 30},
    'N45': {'time': (36, 57), 'polarity': 'negative', 'peak': 45},
    'P60': {'time': (58, 80), 'polarity': 'positive', 'peak': 60},
    'N100': {'time': (81, 144), 'polarity': 'negative', 'peak': 100},
    'P180': {'time': (145, 250), 'polarity': 'positive', 'peak': 180}
}


def analyze_gmfa(epochs: mne.Epochs,
                 components: Dict[str, Dict] = DEFAULT_TEP_COMPONENTS,
                 samples: int = 5,
                 method: str = 'largest') -> Dict[str, Dict]:
    """
    Analyze GMFA using exact TESA logic.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE epochs object containing TEP data
    components : dict
        Component definitions
    samples : int
        Number of samples for peak detection
    method : str
        Peak selection method

    Returns
    -------
    dict
        Results matching TESA's output structure
    """
    times = epochs.times * 1000  # Convert to ms

    # Calculate GMFA exactly as TESA does
    # First average over trials
    trial_avg = np.nanmean(epochs.get_data(), axis=0)  # (channels, times)
    # Then calculate standard deviation across channels
    gmfa = np.std(trial_avg, axis=0)  # (times,)

    # Calculate confidence intervals
    gmfa_trials = np.std(epochs.get_data(), axis=1)  # GMFA for each trial
    ci = 1.96 * (np.std(gmfa_trials, axis=0) / np.sqrt(gmfa_trials.shape[0]))

    results = {}
    for name, criteria in components.items():
        result = extract_tep(
            gmfa,
            times,
            criteria['time'],
            criteria['polarity'],
            samples,
            method,
            criteria['peak']
        )

        # Add confidence intervals
        result['ci'] = ci
        results[name] = result

        # Print TESA-style message
        if result['found'] == 'yes':
            print(f"GMFA {name} peak found with latency of {result['lat']:.1f} ms "
                  f"and amplitude of {result['amp']:.2f} µV.")
        else:
            print(f"GMFA {name} peak not found. Amplitude at {criteria['peak']} ms returned.")

    return results


def analyze_roi(epochs: mne.Epochs,
                channels: List[str],
                components: Dict[str, Dict] = DEFAULT_TEP_COMPONENTS,
                samples: int = 5,
                method: str = 'largest') -> Dict[str, Dict]:
    """
    Analyze ROI using exact TESA logic.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE epochs object containing TEP data
    channels : list
        Channel names for ROI
    components : dict
        Component definitions
    samples : int
        Number of samples for peak detection
    method : str
        Peak selection method

    Returns
    -------
    dict
        Results matching TESA's output structure
    """
    times = epochs.times * 1000  # Convert to ms

    # Get channel indices exactly as TESA does
    if channels[0].lower() == 'all':
        ch_idx = slice(None)
        missing = []
    else:
        ch_idx = []
        missing = []
        for ch in channels:
            try:
                ch_idx.append(epochs.ch_names.index(ch))
            except ValueError:
                missing.append(ch)
                print(f"Warning: {ch} is not present in the current file. "
                      "Electrode not included in average.")

    if not ch_idx:
        raise ValueError("None of the electrodes selected for the ROI are present in the data.")

    # Get ROI data and calculate averages exactly as TESA does
    roi_data = epochs.get_data()[:, ch_idx, :]  # (trials, channels, times)

    # First average over trials (TESA: nanmean over dimension 1)
    trial_avg = np.nanmean(roi_data, axis=0)  # (channels, times)

    # Then average over channels (TESA: nanmean over dimension 3)
    roi_avg = np.nanmean(trial_avg, axis=0)  # (times,)

    # Calculate confidence intervals matching TESA
    # Get channel averages for each trial
    trial_channel_avg = np.nanmean(roi_data, axis=1)  # Average over channels first
    ci = 1.96 * (np.std(trial_channel_avg, axis=0) / np.sqrt(trial_channel_avg.shape[0]))

    results = {}
    for name, criteria in components.items():
        result = extract_tep(
            roi_avg,
            times,
            criteria['time'],
            criteria['polarity'],
            samples,
            method,
            criteria['peak']
        )

        # Add TESA-specific fields
        result['chans'] = channels
        result['ci'] = ci
        if missing:
            result['missing'] = missing

        results[name] = result

        # Print TESA-style message
        if result['found'] == 'yes':
            print(f"ROI {name} peak found with latency of {result['lat']:.1f} ms "
                  f"and amplitude of {result['amp']:.2f} µV.")
        else:
            print(f"ROI {name} peak not found. Amplitude at {criteria['peak']} ms returned.")

    return results


def plot_tep_analysis(epochs: mne.Epochs,
                      output_dir: str,
                      session_name: str,
                      components: Dict[str, Dict] = DEFAULT_TEP_COMPONENTS,
                      analysis_type: str = 'gmfa',
                      channels: Optional[List[str]] = None,
                      n_samples: int = 5,
                      method: str = 'largest') -> Dict[str, Dict]:
    """
    Create comprehensive TEP analysis plot.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE epochs object containing TEP data
    output_dir : str
        Directory to save output files
    session_name : str
        Name of session for file naming
    components : dict
        Component definitions (default uses TESA components)
    analysis_type : str
        'gmfa' or 'roi'
    channels : list or None
        Channel names for ROI analysis (required if analysis_type='roi')
    n_samples : int
        Number of samples for peak detection
    method : str
        Peak selection method

    Returns
    -------
    dict
        Results dictionary containing peak information
    """
    os.makedirs(output_dir, exist_ok=True)
    times = epochs.times * 1000
    evoked = epochs.average()

    # Perform analysis
    if analysis_type.lower() == 'gmfa':
        results = analyze_gmfa(epochs, components, n_samples, method)
        plot_data = np.std(evoked.get_data(), axis=0)
        ylabel = 'GMFA (µV)'
    elif analysis_type.lower() == 'roi':
        if channels is None:
            raise ValueError("channels must be provided for ROI analysis")
        results = analyze_roi(epochs, channels, components, n_samples, method)
        if channels[0].lower() == 'all':
            ch_idx = slice(None)
        else:
            ch_idx = [evoked.ch_names.index(ch) for ch in channels
                      if ch in evoked.ch_names]
        plot_data = np.mean(evoked.get_data()[ch_idx], axis=0)
        ylabel = 'ROI Average (µV)'
    else:
        raise ValueError("analysis_type must be either 'gmfa' or 'roi'")

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    gs_main = gs[0].subgridspec(2, 1, height_ratios=[1.5, 1], hspace=0.3)

    # Butterfly plot with GMFA/GFP
    ax_butterfly = fig.add_subplot(gs_main[0])
    evoked.plot(gfp=True, xlim=(-0.1, 0.4), axes=ax_butterfly, show=False)
    ax_butterfly.set_title(f'TEP Butterfly Plot with {analysis_type.upper()}')

    # Analysis plot
    ax_analysis = fig.add_subplot(gs_main[1])
    ax_analysis.plot(times, plot_data, 'b-', label=analysis_type.upper())

    # Add confidence intervals if available
    if all('ci' in comp for comp in results.values()):
        ci = next(iter(results.values()))['ci']
        ax_analysis.fill_between(times, plot_data - ci, plot_data + ci,
                                 color='b', alpha=0.2, label='95% CI')

    # Plot detected components
    colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
    for (name, comp), color in zip(results.items(), colors):
        if comp['found'] == 'yes':
            ax_analysis.plot(comp['lat'], comp['amp'], 'o',
                             color=color, label=f"{name} ({comp['lat']:.0f} ms)")
            ax_analysis.axvline(comp['lat'], color=color, alpha=0.2)

            # Add window markers
            t_min, t_max = components[name]['time']
            ax_analysis.axvspan(t_min, t_max, color=color, alpha=0.1)

    # Customize analysis plot
    ax_analysis.set_xlabel('Time (ms)')
    ax_analysis.set_ylabel(ylabel)
    ax_analysis.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_analysis.grid(True, alpha=0.3)
    ax_analysis.set_xlim(-100, 400)
    ax_analysis.set_title(f'{analysis_type.upper()} with TEP Components')

    # Add topomaps
    gs_topos = gs[1].subgridspec(1, len(components), wspace=0.3)

    for idx, (name, comp) in enumerate(results.items()):
        ax = fig.add_subplot(gs_topos[idx])
        if comp['found'] == 'yes':
            try:
                evoked.plot_topomap(times=comp['lat'] / 1000.0,
                                    axes=ax,
                                    show=False,
                                    time_format=f'{name}\n{comp["lat"]:.0f} ms',
                                    colorbar=False)
            except Exception as e:
                ax.text(0.5, 0.5, f"Could not plot\n{name}",
                        ha='center', va='center')

    # Add colorbar
    cax = fig.add_axes([0.92, 0.11, 0.02, 0.15])
    plt.colorbar(ax.images[-1], cax=cax)

    plt.suptitle(f'TEP Analysis - {session_name}', y=0.95)
    plt.subplots_adjust(right=0.85)

    # Save figure
    fig.savefig(os.path.join(output_dir, f'{session_name}_tep_analysis.png'),
                dpi=600, bbox_inches='tight')
    plt.close(fig)

    return results


def generate_validation_summary(components: Dict,
                                output_dir: str,
                                session_name: str):
    """
    Generate a validation summary for TEP components.

    Parameters
    ----------
    components : dict
        Dictionary containing detected TEP components and their properties
    output_dir : str
        Directory to save the summary
    session_name : str
        Name of the session for file naming

    Returns
    -------
    None
        Writes summary to a text file
    """
    summary_path = os.path.join(output_dir, f'{session_name}_validation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("TEP Validation Summary\n")
        f.write("=" * 50 + "\n\n")

        # Add analysis timestamp
        from datetime import datetime
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session: {session_name}\n\n")

        # Summary statistics
        total_components = len(components)
        found_components = sum(1 for comp in components.values() if comp['found'] == 'yes')
        valid_components = 0

        f.write("Component Analysis\n")
        f.write("-" * 30 + "\n\n")

        # Analyze each component
        for name, comp in components.items():
            f.write(f"{name}:\n")
            f.write("-" * len(name) + "\n")

            # Check if peak was found
            if comp['found'] == 'yes':
                latency = comp['lat'] if 'lat' in comp else comp['time']
                amplitude = comp['amp'] if 'amp' in comp else comp['amplitude']

                # Check if within expected window
                expected_range = DEFAULT_TEP_COMPONENTS[name]['time']
                is_valid = expected_range[0] <= latency <= expected_range[1]
                if is_valid:
                    valid_components += 1

                f.write(f"  Status: Peak found\n")
                f.write(f"  Latency: {latency:.1f} ms ")
                f.write(f"({'valid' if is_valid else 'outside expected range'})\n")
                f.write(f"  Amplitude: {amplitude:.2f} µV\n")
                f.write(f"  Expected range: {expected_range[0]}-{expected_range[1]} ms\n")

                # Add confidence intervals if available
                if 'ci' in comp:
                    ci_at_peak = comp['ci'][int(np.argmin(np.abs(latency)))]
                    f.write(f"  95% CI at peak: ±{ci_at_peak:.2f} µV\n")

                # Add warning if outside expected range
                if not is_valid:
                    f.write(f"  WARNING: Peak latency {latency:.1f} ms is outside expected ")
                    f.write(f"window of {expected_range[0]}-{expected_range[1]} ms\n")
            else:
                f.write(f"  Status: No peak found\n")
                f.write(f"  Expected range: {DEFAULT_TEP_COMPONENTS[name]['time'][0]}-"
                        f"{DEFAULT_TEP_COMPONENTS[name]['time'][1]} ms\n")
                if 'amp' in comp:
                    f.write(f"  Amplitude at target latency: {comp['amp']:.2f} µV\n")

            f.write("\n")

        # Write summary statistics
        f.write("\nSummary Statistics\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total components analyzed: {total_components}\n")
        f.write(f"Components with peaks found: {found_components}\n")
        f.write(f"Components within expected windows: {valid_components}\n")
        f.write(f"Detection rate: {(found_components / total_components) * 100:.1f}%\n")
        f.write(f"Validation rate: {(valid_components / total_components) * 100:.1f}%\n")

        # Write additional notes
        f.write("\nNotes\n")
        f.write("-" * 10 + "\n")
        f.write("- Expected time windows are based on standard TEP component definitions\n")
        f.write("- Validation considers both peak detection and latency window criteria\n")
        f.write("- Components without detected peaks are counted as invalid\n")

    print(f"Validation summary saved to: {summary_path}")
    return None