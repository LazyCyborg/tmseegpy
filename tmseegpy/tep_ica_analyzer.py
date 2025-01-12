import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from scipy import signal
from typing import Dict, List, Tuple, Optional
import mne


class TEPAnalyzer:
    """
    A class for analyzing TMS-evoked potentials using ICA decomposition of evoked data.
    This novel approach applies ICA to the evoked response to separate different TEP components.

    Parameters
    ----------
    epochs : mne.Epochs
        The preprocessed epochs containing TMS responses
    n_components : int, optional
        Number of ICA components to extract (default is 15)
    random_state : int, optional
        Random seed for reproducibility (default is 42)
    """

    def __init__(self, epochs: mne.Epochs, n_components: int = 9, random_state: int = 42):
        self.epochs = epochs
        self.evoked = epochs.average()
        self.n_components = n_components
        self.random_state = random_state
        self.ica = None
        self.component_scores = None
        self.tep_components = None

        # Expected TEP peaks and their time windows
        self.tep_peaks = {
            'N15-P30': {'window': (15, 40), 'polarity': 'both'},
            'N45': {'window': (40, 50), 'polarity': 'negative'},
            'P60': {'window': (55, 70), 'polarity': 'positive'},
            'N100': {'window': (85, 140), 'polarity': 'negative'},
            'P180': {'window': (150, 250), 'polarity': 'positive'}
        }

    def decompose_teps(self) -> Dict:
        """
        Decompose evoked response using ICA and identify TEP-related components.

        Returns
        -------
        dict
            Dictionary containing component scores and classifications
        """
        # Reshape evoked data for ICA
        data = self.evoked.data
        n_channels, n_times = data.shape

        # Initialize and fit ICA
        self.ica = ICA(
            n_components=self.n_components,
            random_state=self.random_state,
            method='fastica'
        )

        # Fit ICA directly on the original epochs
        self.ica.fit(self.epochs)

        # Get ICA components
        components = self.ica.get_sources(self.evoked)

        # Analyze each component
        self.component_scores = self._score_components(components)

        # Identify TEP components
        self.tep_components = self._classify_components()

        return self.tep_components

    def _score_components(self, components) -> Dict:
        """
        Score ICA components based on their temporal characteristics.

        Parameters
        ----------
        components : array
            ICA component time courses

        Returns
        -------
        dict
            Component scores for different TEP characteristics
        """
        scores = {
            'temporal_correlation': [],
            'peak_times': [],
            'peak_amplitudes': [],
            'snr': []
        }

        times = self.evoked.times * 1000  # Convert to ms

        for comp_idx in range(self.n_components):
            comp_data = components.data[comp_idx]

            # Find peaks in the component
            peaks, properties = signal.find_peaks(np.abs(comp_data),
                                                  prominence=0.1 * np.max(np.abs(comp_data)))
            peak_times = times[peaks]
            peak_amplitudes = comp_data[peaks]

            # Calculate SNR
            signal_window = slice(np.where(times >= 0)[0][0],
                                  np.where(times <= 300)[0][-1])
            noise_window = slice(np.where(times >= -200)[0][0],
                                 np.where(times <= -10)[0][-1])

            signal_power = np.var(comp_data[signal_window])
            noise_power = np.var(comp_data[noise_window])
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0

            scores['peak_times'].append(peak_times)
            scores['peak_amplitudes'].append(peak_amplitudes)
            scores['snr'].append(snr)

            # Calculate temporal correlation with expected TEP pattern
            temporal_corr = self._calculate_tep_correlation(comp_data, times)
            scores['temporal_correlation'].append(temporal_corr)

        return scores

    def _calculate_tep_correlation(self, component: np.ndarray, times: np.ndarray) -> float:
        """
        Calculate correlation between component and ideal TEP pattern.

        Parameters
        ----------
        component : array
            Component time course
        times : array
            Time points in ms

        Returns
        -------
        float
            Correlation score
        """
        # Create idealized TEP pattern
        ideal_tep = np.zeros_like(times)

        for peak, info in self.tep_peaks.items():
            window = info['window']
            polarity = info['polarity']

            mask = (times >= window[0]) & (times <= window[1])
            if polarity == 'negative':
                ideal_tep[mask] = -1
            elif polarity == 'positive':
                ideal_tep[mask] = 1
            elif polarity == 'both':
                # Create biphasic response
                mid_point = (window[0] + window[1]) / 2
                mask_neg = (times >= window[0]) & (times <= mid_point)
                mask_pos = (times > mid_point) & (times <= window[1])
                ideal_tep[mask_neg] = -1
                ideal_tep[mask_pos] = 1

        # Calculate correlation
        correlation = np.corrcoef(ideal_tep, component)[0, 1]
        return correlation if not np.isnan(correlation) else 0

    def _classify_components(self) -> Dict:
        """
        Classify components as TEP or non-TEP based on scoring.

        Returns
        -------
        dict
            Classification of components and their characteristics
        """
        classifications = {}

        for comp_idx in range(self.n_components):
            # Get component scores
            temporal_corr = self.component_scores['temporal_correlation'][comp_idx]
            snr = self.component_scores['snr'][comp_idx]
            peak_times = self.component_scores['peak_times'][comp_idx]

            # Initialize component info
            comp_info = {
                'temporal_correlation': temporal_corr,
                'snr': snr,
                'peak_times': peak_times,
                'associated_teps': [],
                'is_tep': False
            }

            # Check if peaks align with known TEP windows
            for peak_name, peak_info in self.tep_peaks.items():
                window = peak_info['window']
                peaks_in_window = [t for t in peak_times
                                   if window[0] <= t <= window[1]]

                if peaks_in_window:
                    comp_info['associated_teps'].append(peak_name)

            # Classify as TEP component if:
            # 1. Has good temporal correlation
            # 2. Has good SNR
            # 3. Has peaks in known TEP windows
            if (temporal_corr > 0.3 and
                    snr > 3 and
                    len(comp_info['associated_teps']) > 0):
                comp_info['is_tep'] = True

            classifications[comp_idx] = comp_info

        return classifications

    def plot_tep_components(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot the decomposed TEP components.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plots
        """
        if self.tep_components is None:
            raise ValueError("Must run decompose_teps() first")

        # Count TEP components
        tep_comps = [idx for idx, info in self.tep_components.items()
                     if info['is_tep']]
        n_tep_comps = len(tep_comps)

        if n_tep_comps == 0:
            raise ValueError("No TEP components found")

        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(n_tep_comps + 1, 2, height_ratios=[2] + [1] * n_tep_comps)

        # Plot original evoked response
        ax_evoked = fig.add_subplot(gs[0, :])
        times = self.evoked.times * 1000
        self.evoked.plot(axes=ax_evoked, show=False)
        ax_evoked.set_title('Original Evoked Response')

        # Plot TEP components
        for i, comp_idx in enumerate(tep_comps):
            ax = fig.add_subplot(gs[i + 1, :])

            # Get component data
            comp_data = self.ica.get_sources(self.evoked).data[comp_idx]

            # Plot component
            ax.plot(times, comp_data, 'b-', linewidth=1)

            # Add component information
            info = self.tep_components[comp_idx]
            title = f"Component {comp_idx} - TEPs: {', '.join(info['associated_teps'])}"
            ax.set_title(title)

            # Add peaks
            peak_times = info['peak_times']
            peak_amplitudes = comp_data[np.array([np.abs(times - t).argmin()
                                                  for t in peak_times])]
            ax.plot(peak_times, peak_amplitudes, 'r.', markersize=10)

            # Add annotations
            ax.text(0.98, 0.98,
                    f"SNR: {info['snr']:.1f} dB\nCorr: {info['temporal_correlation']:.2f}",
                    transform=ax.transAxes,
                    horizontalalignment='right',
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8))

            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (ms)')

        plt.tight_layout()
        return fig

    def enhanced_tep_plot(self, time_window=(-100, 300), ylim=None):
        """
        Create an enhanced visualization of TEP components with clear labeling
        and comparison between original and reconstructed signals.

        Parameters
        ----------
        time_window : tuple
            Time window to display in ms (default: (-100, 300))
        ylim : tuple or None
            Y-axis limits (min, max) in µV. If None, automatically determined.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plots
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec

        if self.tep_components is None:
            raise ValueError("Must run decompose_teps() first")

        # Create figure with adjusted size
        fig = plt.figure(figsize=(20, 15))

        # Create main GridSpec with better spacing
        n_topos = len(self.tep_peaks)
        gs = GridSpec(3, n_topos, height_ratios=[1.2, 1.2, 1.5], hspace=0.4)

        # Plot 1: Original vs Reconstructed TEPs (spans all columns)
        ax1 = fig.add_subplot(gs[0, :])

        # Get data and time mask
        times = self.evoked.times * 1000  # Convert to ms
        time_mask = (times >= time_window[0]) & (times <= time_window[1])
        plot_times = times[time_mask]

        orig_data = self.evoked.data.mean(axis=0)[time_mask] * 1e6  # Convert to µV
        reconstructed = self.reconstruct_teps()
        recon_data = reconstructed.data.mean(axis=0)[time_mask] * 1e6

        # Plot with better styling
        ax1.plot(plot_times, orig_data, 'k-', label='Original', linewidth=1.5, alpha=0.6)
        ax1.plot(plot_times, recon_data, 'r-', label='Reconstructed', linewidth=2)

        # Add TEP peak markers with improved visibility and spacing
        y_range = np.ptp(orig_data)
        max_y = ax1.get_ylim()[1]
        y_positions = np.linspace(max_y * 0.1, max_y * 0.3, len(self.tep_peaks))

        for (name, info), y_pos in zip(self.tep_peaks.items(), y_positions):
            window = info['window']
            if window[0] >= time_window[0] and window[1] <= time_window[1]:
                # Draw window
                ax1.axvspan(window[0], window[1], color='gray', alpha=0.1)
                # Add label with better positioning
                mid_time = (window[0] + window[1]) / 2
                ax1.text(mid_time, y_pos, name,
                         ha='center', va='bottom', rotation=0,
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray',
                                   pad=2, boxstyle='round'))

        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='TMS')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Time (ms)', fontsize=12)
        ax1.set_ylabel('Amplitude (µV)', fontsize=12)
        ax1.set_title('Original vs Reconstructed TEPs', fontsize=14, pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1.1))

        # Plot 2: Individual TEP Components (spans all columns)
        ax2 = fig.add_subplot(gs[1, :])

        # Get TEP components
        tep_comps = [idx for idx, info in self.tep_components.items() if info['is_tep']]

        if not tep_comps:
            ax2.text(0.5, 0.5, 'No TEP components found',
                     ha='center', va='center', transform=ax2.transAxes,
                     fontsize=12)
        else:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(tep_comps)))

            for comp_idx, color in zip(tep_comps, colors):
                # Get component data
                comp_data = self.ica.get_sources(self.evoked).data[comp_idx][time_mask]
                # Normalize for visualization
                comp_data = comp_data / np.max(np.abs(comp_data))

                # Plot component with improved styling
                ax2.plot(plot_times, comp_data, color=color,
                         label=f"Component {comp_idx}", alpha=0.8,
                         linewidth=2)

                # Add peak markers
                peak_times = self.tep_components[comp_idx]['peak_times']
                peak_times = peak_times[
                    (peak_times >= time_window[0]) &
                    (peak_times <= time_window[1])
                    ]
                peak_indices = [np.abs(plot_times - t).argmin() for t in peak_times]
                if len(peak_indices) > 0:
                    peak_values = comp_data[peak_indices]
                    ax2.plot(peak_times, peak_values, 'o', color=color,
                             markersize=8, markeredgecolor='white',
                             markeredgewidth=1)

                    # Add component info
                    info = self.tep_components[comp_idx]
                    ax2.text(0.02, 0.98 - (0.1 * tep_comps.index(comp_idx)),
                             f"Comp {comp_idx} - SNR: {info['snr']:.1f}dB, " +
                             f"TEPs: {', '.join(info['associated_teps'])}",
                             transform=ax2.transAxes,
                             color=color,
                             fontsize=10,
                             va='top',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        # Set x-axis ticks to increment by 10ms
        xticks = np.arange(np.floor(time_window[0] / 10) * 10,
                           np.ceil(time_window[1] / 10) * 10, 10)
        ax2.set_xticks(xticks)
        ax2.set_xlabel('Time (ms)', fontsize=12)
        ax2.set_ylabel('Normalized Amplitude', fontsize=12)
        ax2.set_title('Individual TEP Components', fontsize=14, pad=20)

        # Do the same for the first plot
        ax1.set_xticks(xticks)

        # Plot topographies with better spacing
        for i, (name, info) in enumerate(self.tep_peaks.items()):
            ax_topo = fig.add_subplot(gs[2, i])

            # Find best time point in window
            window = info['window']
            window_mask = (times >= window[0]) & (times <= window[1])
            if np.any(window_mask):
                peak_time = times[window_mask][
                    np.argmax(np.abs(reconstructed.data.mean(axis=0)[window_mask]))
                ]

                # Plot topography
                reconstructed.plot_topomap(
                    times=peak_time / 1000.0,  # Convert back to seconds
                    axes=ax_topo,
                    show=False,
                    time_format=f'{name}\n{peak_time:.0f} ms',
                    colorbar=False,
                    extrapolate='head'
                )

        # Add colorbar with better positioning
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax_topo)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(ax_topo.images[-1], cax=cax)
        cbar.set_label('Amplitude (µV)', fontsize=10)

        plt.tight_layout()
        return fig

    def reconstruct_teps(self) -> mne.Evoked:
        """
        Reconstruct evoked response using only TEP-related components.

        Returns
        -------
        mne.Evoked
            Reconstructed evoked response containing only TEP components
        """
        if self.tep_components is None:
            raise ValueError("Must run decompose_teps() first")

        # Get TEP component indices
        tep_comps = [idx for idx, info in self.tep_components.items()
                     if info['is_tep']]

        if not tep_comps:
            raise ValueError("No TEP components found")

        # Create copy of ICA with only TEP components
        ica_tep = self.ica.copy()
        exclude = [idx for idx in range(self.n_components) if idx not in tep_comps]

        # Reconstruct using only TEP components
        evoked_tep = self.evoked.copy()
        ica_tep.apply(evoked_tep, exclude=exclude)

        return evoked_tep