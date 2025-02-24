import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne
from typing import Dict, Tuple, List, Optional, Union
import os
import plotly.graph_objects as go

DEFAULT_TEP_COMPONENTS = {
    'N15': {
        'time': (10, 20),
        'center_time': 15,
        'search_window': 5,  # Half the window size
        'polarity': 'negative',
        'peak': 15,
        'expected_distribution': 'focal',
        'min_amplitude': 0.5
    },
    'P30': {
        'time': (20, 40),
        'center_time': 30,
        'search_window': 10,
        'polarity': 'positive',
        'peak': 30,
        'expected_distribution': 'focal',
        'min_amplitude': 0.5
    },
    'N45': {
        'time': (40, 55),
        'center_time': 45,
        'search_window': 7.5,
        'polarity': 'negative',
        'peak': 45,
        'expected_distribution': 'focal',
        'min_amplitude': 0.7
    },
    'P60': {
        'time': (50, 70),
        'center_time': 60,
        'search_window': 10,
        'polarity': 'positive',
        'peak': 60,
        'expected_distribution': 'focal',
        'min_amplitude': 0.7
    },
    'N100': {
        'time': (70, 150),
        'center_time': 100,
        'search_window': 40,
        'polarity': 'negative',
        'peak': 100,
        'expected_distribution': 'contralateral',
        'min_amplitude': 1.0
    },
    'P180': {
        'time': (150, 240),
        'center_time': 180,
        'search_window': 45,
        'polarity': 'positive',
        'peak': 180,
        'expected_distribution': 'bilateral',
        'min_amplitude': 1.0
    }
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
        # Initialize arrays for comparisons
        t_plus = np.zeros(samples)
        t_minus = np.zeros(samples)

        # Calculate differences
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


def parse_peak_windows(peak_windows_str):
    """Parse peak windows from command line string."""
    if not peak_windows_str:
        return None

    windows = []
    for window in peak_windows_str:
        try:
            start, end = map(float, window.split(','))
            windows.append((start, end))
        except ValueError:
            print(f"Warning: Could not parse window {window}, skipping...")
            continue
    return windows if windows else None


def classify_peaks(peaks: List[Dict],
                   components: Dict = DEFAULT_TEP_COMPONENTS) -> List[Dict]:
    """
    Classify detected peaks according to known TEP components.

    Parameters
    ----------
    peaks : List[Dict]
        List of peaks from get_single_channel_peaks with 'latency' and 'amplitude'
    components : Dict
        Component definitions with center times and search windows

    Returns
    -------
    List[Dict]
        Original peaks with added classification information
    """
    classified_peaks = []
    print("\nPeak Classification Results:")
    print("-" * 30)

    for peak in peaks:
        peak_time = peak['latency']
        peak_amp = peak['amplitude']

        # Check each component
        for name, params in components.items():
            # Get window boundaries using either format
            if 'time' in params:
                min_time, max_time = params['time']
                window_size = (max_time - min_time) / 2
                center = (max_time + min_time) / 2
            else:
                center = params['center_time']
                window_size = params['search_window']
                min_time = center - window_size
                max_time = center + window_size

            # Check if peak falls within window
            if min_time <= peak_time <= max_time:
                # Calculate confidence based on distance from center
                time_diff = abs(peak_time - center)
                confidence = 1 - (time_diff / window_size)

                # Get polarity from either format
                polarity = params.get('polarity', 'unknown')

                peak_with_class = peak.copy()
                peak_with_class.update({
                    'component': name,
                    'confidence': confidence,
                    'polarity': polarity,
                    'expected_distribution': params.get('expected_distribution', 'unknown')
                })
                classified_peaks.append(peak_with_class)

                print(f"Peak at {peak_time:.1f}ms classified as {name}")
                print(f"  Channel: {peak['channel']}")
                print(f"  Amplitude: {peak_amp:.2f}µV")
                print(f"  Confidence: {confidence:.2f}")
                print(f"  Polarity: {polarity}")
                print(f"  Expected Distribution: {params.get('expected_distribution', 'unknown')}")
                break
        else:
            # No matching component found
            peak_with_class = peak.copy()
            peak_with_class.update({
                'component': 'unknown',
                'confidence': 0.0,
                'polarity': 'unknown',
                'expected_distribution': None
            })
            classified_peaks.append(peak_with_class)
            print(f"Unclassified peak found at {peak_time:.1f}ms")
            print(f"  Channel: {peak['channel']}")
            print(f"  Amplitude: {peak_amp:.2f}µV")

    print("\nClassification Summary:")
    component_counts = {}
    for peak in classified_peaks:
        component_counts[peak['component']] = component_counts.get(peak['component'], 0) + 1

    for comp, count in component_counts.items():
        print(f"{comp}: {count} peak(s)")

    return classified_peaks


def get_single_channel_peaks(evoked, windows, mode='neg'):
    """
    Find peaks using MNE's get_peak for specified windows.
    Returns exactly one peak per window from the channel with the maximum peak.
    """
    peaks = []
    for start, end in windows:
        try:
            # Convert from ms to seconds for MNE
            ch_name, lat, amp = evoked.get_peak(
                tmin=start / 1000,
                tmax=end / 1000,
                mode=mode,
                return_amplitude=True
            )
            peaks.append({
                'window': (start, end),
                'channel': ch_name,
                'latency': lat * 1000,  # Convert back to ms
                'amplitude': amp # * 1e6  # Convert to µV?
            })
        except Exception as e:
            print(f"Warning: Could not find peak in window {start}-{end}ms: {str(e)}")
            continue
    return peaks




def plot_tep_analysis(epochs: mne.Epochs,
                      output_dir: str,
                      session_name: str,
                      components: Dict[str, Dict] = DEFAULT_TEP_COMPONENTS,
                      analysis_type: str = 'gmfa',
                      channels: Optional[List[str]] = None,
                      n_samples: int = 5,
                      method: str = 'largest',
                      peak_mode: str = None,
                      show_channel_peaks: bool = False,
                      topomap_average: float = 0.01,
                      peak_windows: Optional[List[str]] = None,
                      override_windows: bool = False) -> Dict[str, Dict]:
    """
    Create TEP analysis plot with separate GFP and GMFA plots.

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
    peak_mode : str
        Mode for MNE peak detection ('pos', 'neg', or 'abs')
    show_channel_peaks : bool
        Whether to show individual channel peaks using MNE get_peaks
    peak_windows : Optional[List[str]]
        Time windows for peak detection in format ["start,end", ...] (in ms)
    override_windows : bool
        Use peak_windows to override default component windows
    topomap_average : float
        Time window (in seconds) to average around each peak for topomaps.
        Default is 0.005 (5ms). Increase for smoother but less temporally
        specific topographies.

    Returns
    -------
    dict
        Results dictionary containing peak information
    """
    os.makedirs(output_dir, exist_ok=True)
    times = epochs.times * 1000
    evoked = epochs.average()

    # Parse peak windows if provided
    if peak_windows:
        windows = []
        for window in peak_windows:
            try:
                start, end = map(float, window.split(','))
                windows.append((start, end))
            except ValueError:
                print(f"Warning: Could not parse window {window}, skipping...")
                continue

        if override_windows and windows:
            # Modify component windows if override requested
            print("Overriding default component windows with user-specified windows...")
            for (start, end), name in zip(windows, components.keys()):
                components[name]['time'] = (start, end)
                print(f"Modified window for {name}: {start}-{end}ms")
    else:
        # Use component definition windows if no custom windows provided
        windows = []
        for name, params in components.items():
            if 'time' in params:
                windows.append(params['time'])
            elif 'center_time' in params and 'search_window' in params:
                windows.append((
                    params['center_time'] - params['search_window'],
                    params['center_time'] + params['search_window']
                ))
            else:
                print(f"Warning: Component {name} has invalid window definition")

    if not windows:
        raise ValueError("No valid windows found in component definitions")

    if peak_mode == None:

        # Find peaks using the windows (either default, custom, or overridden)
        detected_peaks = []
        for window, component_name in zip(windows, components.keys()):
            # Get the polarity from component definition
            polarity = components[component_name].get('polarity', 'positive')
            # Set peak detection mode based on polarity
            detection_mode = 'neg' if polarity == 'negative' else 'pos'

            # Get peaks for this window with appropriate polarity
            window_peaks = get_single_channel_peaks(evoked, [window], detection_mode)
            detected_peaks.extend(window_peaks)

        # Classify the detected peaks
        classified_peaks = classify_peaks(detected_peaks, components)
    else:
        detected_peaks = get_single_channel_peaks(evoked, windows, peak_mode)

    # Perform standard analysis
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

    # Determine number of subplots based on channel peaks display
    n_rows = 5 if show_channel_peaks else 4

    # Create figure with larger dimensions and higher DPI
    plt.rcParams['figure.constrained_layout.use'] = False
    fig = plt.figure(figsize=(24, 7 * n_rows), dpi=800)

    # Create main grid for time series plots with more space between subplots
    height_ratios = [1.5, 1, 1, 1.2]
    if show_channel_peaks:
        height_ratios.append(1.2)
    gs_main = GridSpec(len(height_ratios), 1, figure=fig, height_ratios=height_ratios, hspace=0.4)

    # Butterfly plot (keeping existing code)
    ax_butterfly = fig.add_subplot(gs_main[0])
    if analysis_type.lower() == 'gmfa':
        evoked.plot(gfp=False, xlim=(-0.1, 0.4), axes=ax_butterfly, show=False,
                    selectable=False, picks='all')
        ax_butterfly.set_title('TEP Butterfly Plot', fontsize=10, pad=10)
        ax_butterfly.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    else:
        if channels[0].lower() == 'all':
            ch_picks = slice(None)
        else:
            ch_picks = [i for i, ch in enumerate(evoked.ch_names) if ch in channels]
        evoked_roi = evoked.copy()
        evoked_roi.pick(ch_picks)
        evoked_roi.plot(xlim=(-0.1, 0.4), axes=ax_butterfly, show=False)
        ax_butterfly.set_title(f'TEP Plot - ROI Channels ({", ".join(channels)})',
                               fontsize=10, pad=10)
        ax_butterfly.axvline(x=0, color='k', linestyle='--', alpha=0.5)

    # GFP plot (keeping existing code)
    ax_gfp = fig.add_subplot(gs_main[1])
    ax_gfp.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    data = evoked.get_data()
    ch_types = evoked.get_channel_types()

    if all(ch_type == 'eeg' for ch_type in ch_types):
        gfp_data = np.std(data, axis=0, ddof=0)
        label = 'GFP'
    else:
        gfp_data = np.linalg.norm(data, axis=0) / np.sqrt(len(data))
        label = 'RMS'

    ax_gfp.plot(evoked.times, gfp_data, 'b-', label=label)
    ax_gfp.set_title(f'Global Field Power from MNE-Python ({label})', fontsize=10, pad=20)
    ax_gfp.set_xlim(-0.1, 0.4)
    ax_gfp.set_ylabel(f'{label} (µV)')
    ax_gfp.grid(True, alpha=0.3)
    ax_gfp.legend()

    # GMFA/ROI analysis plot (keeping existing code)
    ax_analysis = fig.add_subplot(gs_main[2])
    ax_analysis.plot(times, plot_data, 'b-', label=analysis_type.upper())
    ax_analysis.axvline(x=0, color='k', linestyle='--', alpha=0.5)

    if all('ci' in comp for comp in results.values()):
        ci = next(iter(results.values()))['ci']
        ax_analysis.fill_between(times, plot_data - ci, plot_data + ci,
                                 color='b', alpha=0.2, label='95% CI')

    colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
    for (name, comp), color in zip(results.items(), colors):
        if comp['found'] == 'yes':
            ax_analysis.plot(comp['lat'], comp['amp'], 'o',
                             color=color, label=f"{name} ({comp['lat']:.0f} ms, {comp['amp']:.1f} µV)")
            ax_analysis.axvline(comp['lat'], color=color, alpha=0.2)
            t_min, t_max = components[name]['time']
            ax_analysis.axvspan(t_min, t_max, color=color, alpha=0.1)

    ax_analysis.set_xlabel('Time (ms)')
    ax_analysis.set_ylabel(ylabel)
    ax_analysis.legend(loc='upper right', fontsize=12, bbox_to_anchor=(0.98, 0.98))
    ax_analysis.grid(True, alpha=0.3)
    ax_analysis.set_xlim(-100, 400)
    ax_analysis.set_title(f'{analysis_type.upper()} with TEP Components',
                          fontsize=10, pad=10)

    # Add MNE get_peaks analysis if requested
    if show_channel_peaks:
        ax_peaks = fig.add_subplot(gs_main[3])

        # Find peaks for each window
        mne_peaks = []
        for window, component_name in zip(windows, components.keys()):
            try:
                # Get the polarity from component definition if no peak_mode specified
                if peak_mode is None:
                    polarity = components[component_name].get('polarity', 'positive')
                    detection_mode = 'neg' if polarity == 'negative' else 'pos'
                else:
                    detection_mode = peak_mode

                # Convert from ms to seconds for MNE
                ch_name, lat, amp = evoked.get_peak(
                    tmin=window[0] / 1000,  # Convert to seconds
                    tmax=window[1] / 1000,  # Convert to seconds
                    mode=detection_mode,
                    return_amplitude=True
                )
                mne_peaks.append({
                    'window': window,
                    'channel': ch_name,
                    'latency': lat * 1000,  # Convert back to ms
                    'amplitude': amp  # Already in µV!!
                })
            except Exception as e:
                print(f"Warning: Could not find peak in window {window[0]}-{window[1]}ms: {str(e)}")
                continue

        # Plot each channel that has a peak
        plotted_channels = set()
        for peak in classified_peaks:  # Use classified_peaks instead of mne_peaks
            ch_name = peak['channel']
            if ch_name not in plotted_channels:
                ch_idx = evoked.ch_names.index(ch_name)
                ch_data = evoked.data[ch_idx]
                ax_peaks.plot(times, ch_data, '-', alpha=0.7,
                              label=f'Channel {ch_name}')
                plotted_channels.add(ch_name)

            # Determine marker color based on component and confidence
            if peak['component'] != 'unknown':
                marker_color = 'green' if peak['confidence'] > 0.7 else 'orange'
                component_label = peak['component']
            else:
                marker_color = 'gray'
                component_label = 'Unknown'

            # Plot the peak with classification information
            ax_peaks.plot(peak['latency'], peak['amplitude'], '*',
                          markersize=10,
                          color=marker_color,
                          label=f"{component_label}: {peak['latency']:.1f}ms, {peak['amplitude']:.1f}µV ({ch_name})")
            ax_peaks.axvline(peak['latency'], color=marker_color, alpha=0.2)

            # Add text annotation with component information
            annotation_text = [f"{peak['amplitude']:.1f}µV"]
            if peak['component'] != 'unknown':
                annotation_text.append(f"{peak['component']}")
                if 'confidence' in peak:
                    annotation_text.append(f"(conf: {peak['confidence']:.2f})")

            ax_peaks.annotate(
                '\n'.join(annotation_text),
                (peak['latency'], peak['amplitude']),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='bottom',
                color=marker_color
            )

        ax_peaks.set_xlabel('Time (ms)')
        ax_peaks.set_ylabel('Amplitude (µV)')
        ax_peaks.legend(loc='upper right', fontsize=12, bbox_to_anchor=(0.98, 0.98))
        ax_peaks.grid(True, alpha=0.3)
        ax_peaks.set_xlim(-100, 400)
        ax_peaks.set_title(f'Single-Channel Peak Detection (Component-Specific Polarity)',
                           fontsize=10, pad=10)
        ax_peaks.axvline(x=0, color='k', linestyle='--', alpha=0.5)

    # Topomaps section
    n_peaks = len(classified_peaks)
    bottom_pos = 0.02 if show_channel_peaks else 0.05
    top_pos = 0.22
    gs_topos = GridSpec(1, n_peaks, bottom=bottom_pos, top=top_pos,
                        left=0.03, right=0.97, wspace=0.1)

    for idx, peak in enumerate(classified_peaks):
        ax = fig.add_subplot(gs_topos[idx])
        try:
            # Plot topomap at the detected peak time
            evoked.plot_topomap(times=peak['latency'] / 1000.0,
                                average=topomap_average,
                                axes=ax,
                                show=False,
                                time_format='',
                                colorbar=False,
                                size=8.0,
                                res=1024,
                                outlines='head',
                                extrapolate='head')

            # Create title with component classification
            title_parts = []
            title_parts.append(f"{peak['latency']:.0f}ms")

            if peak['component'] != 'unknown':
                title_parts.append(f"{peak['component']}")
                confidence_color = 'green' if peak['confidence'] > 0.7 else 'orange'
            else:
                title_parts.append("Unknown")
                confidence_color = 'gray'

            # Add channel info
            title_parts.append(f"Ch: {peak['channel']}")

            title = '\n'.join(title_parts)
            ax.set_title(title, color=confidence_color, pad=10, fontsize=10)

            # Add window information
            window_text = f"Window: {peak['window'][0]:.0f}-{peak['window'][1]:.0f}ms (topomap average: {topomap_average}s)"
            ax.text(0.5, -0.15, window_text,
                    ha='center', va='top', transform=ax.transAxes,
                    fontsize=8)

        except Exception as e:
            print(f"Error plotting topomap: {str(e)}")
            ax.clear()
            ax.set_visible(False)

    # Add colorbar
    if n_peaks > 0:
        try:
            cax = fig.add_axes([0.96, 0.05, 0.01, 0.15])
            plt.colorbar(ax.images[-1], cax=cax)
        except (IndexError, UnboundLocalError):
            pass

    plt.suptitle(f'TEP Analysis - {session_name}', y=0.98, fontsize=12)

    # Adjust main plots area
    top_pos = 0.95 if show_channel_peaks else 0.95
    gs_main.update(top=top_pos, bottom=0.25, left=0.1, right=0.9)

    # Save figure
    fig.savefig(os.path.join(output_dir, f'{session_name}_tep_analysis.png'),
                dpi=400,
                bbox_inches=None,
                pad_inches=0.1)
    plt.close(fig)

    return results


def plotly_evoked(evoked, tlims=(-0.3, 0.6)):
    ixs = (tlims[0] < evoked.times) & (evoked.times < tlims[1])
    data = evoked.data[:, ixs] #* 1e6
    n_chs, n_times = data.shape
    times = evoked.times[ixs] * 1e3

    # Create traces for each channel
    traces = []
    for ch in range(n_chs):
        trace = go.Scatter(
            x=times,
            y=data[ch, :],  # Y-axis (evoked potential data for the channel)
            mode='lines',
            name=evoked.ch_names[ch],
            hovertemplate='Amplitude: %{y:.2f} uV<br>Time: %{x:.2f} ms'
        )
        traces.append(trace)

    # Create layout for the plot
    layout = go.Layout(
        title='',
        xaxis=dict(title='Time (ms)'),
        yaxis=dict(title='Voltage (uV)'),
        hovermode='closest',
        width=1000,  # Set the width in pixels
        height=600
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    # Show the interactive plot
    fig.show()


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