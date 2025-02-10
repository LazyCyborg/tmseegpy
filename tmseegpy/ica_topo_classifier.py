import numpy as np
from scipy.spatial.distance import cdist
import mne


class ICATopographyClassifier:
    """
    Classifier for ICA component topographies to detect artifacts based on empirical patterns.
    """

    def __init__(self, ica_instance, inst):
        """
        Initialize classifier with ICA instance and data.

        Parameters
        ----------
        ica_instance : mne.preprocessing.ICA
            Fitted ICA instance
        inst : mne.io.Raw or mne.Epochs
            Data instance used for ICA
        """
        self.ica = ica_instance
        self.inst = inst
        self.info = inst.info

        # Get channel positions
        self.pos = np.array([ch['loc'][:2] for ch in self.info['chs']
                             if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH])

        # Get ICA component topographies
        self.patterns = self.ica.get_components()

        # Parameters based on empirical analysis
        self.zscore_threshold = 3.5  # Mean artifact z-score was 4.08
        self.peak_count_threshold = 3  # Mean artifact peaks was 2.21
        self.edge_dist_threshold = 0.15  # Based on peak distances (~0.09)
        self.focal_area_threshold = 0.2  # For checking spatial spread

    def _normalize_pattern(self, pattern):
        """Normalize pattern to z-scores."""
        return (pattern - np.mean(pattern)) / np.std(pattern)

    def _get_peaks(self, z_pattern):
        """
        Find peaks in the normalized pattern.

        Parameters
        ----------
        z_pattern : array
            Z-scored pattern

        Returns
        -------
        peaks : array
            Indices of peak locations
        peak_positions : list
            Positions of peaks in 2D space
        peak_distances : list
            Distances of peaks from center
        """
        # Find all peaks exceeding z-score threshold
        peaks = np.where(np.abs(z_pattern) > self.zscore_threshold)[0]

        if len(peaks) == 0:
            return [], [], []

        # Get positions and distances
        peak_positions = [self.pos[p] for p in peaks]
        peak_distances = [np.linalg.norm(pos) for pos in peak_positions]

        return peaks, peak_positions, peak_distances

    def _analyze_spatial_pattern(self, peak_positions):
        """
        Analyze the spatial distribution of peaks.

        Parameters
        ----------
        peak_positions : list
            Positions of peaks in 2D space

        Returns
        -------
        dict
            Spatial pattern characteristics
        """
        if not peak_positions:
            return {'spread': 0, 'is_focal': False, 'is_edge': False}

        # Calculate spread between peaks
        if len(peak_positions) > 1:
            distances = cdist(peak_positions, peak_positions)
            spread = np.mean(distances)
        else:
            spread = 0

        # Check if activity is focal (concentrated in small area)
        is_focal = spread < self.focal_area_threshold

        # Check if peaks are at edge
        edge_distances = [np.linalg.norm(pos) for pos in peak_positions]
        is_edge = any(d > (1 - self.edge_dist_threshold) for d in edge_distances)

        return {
            'spread': spread,
            'is_focal': is_focal,
            'is_edge': is_edge
        }

    def classify_component(self, idx):
        """
        Classify a single ICA component based on empirical criteria.

        Parameters
        ----------
        idx : int
            Component index

        Returns
        -------
        str
            Classification ('artifact' or 'non_artifact')
        dict
            Classification details and metrics
        """
        # Get component pattern
        pattern = self.patterns[:, idx]
        z_pattern = self._normalize_pattern(pattern)

        # Get basic metrics
        max_zscore = np.max(np.abs(z_pattern))
        peaks, peak_positions, peak_distances = self._get_peaks(z_pattern)
        n_peaks = len(peaks)

        # Analyze spatial distribution
        spatial_info = self._analyze_spatial_pattern(peak_positions)

        # Collect reasons for classification
        reasons = []

        if max_zscore > self.zscore_threshold:
            reasons.append(f"high_amplitude (z={max_zscore:.2f})")

        if n_peaks < self.peak_count_threshold:
            reasons.append(f"focal_pattern (peaks={n_peaks})")

        if spatial_info['is_focal']:
            reasons.append("concentrated_activity")

        if spatial_info['is_edge']:
            reasons.append("edge_activity")

        # Build details dictionary
        details = {
            'max_zscore': max_zscore,
            'n_peaks': n_peaks,
            'peak_positions': peak_positions,
            'peak_distances': peak_distances,
            'spatial_spread': spatial_info['spread'],
            'is_focal': spatial_info['is_focal'],
            'is_edge': spatial_info['is_edge'],
            'reasons': reasons
        }

        # Classify as artifact if multiple indicators present
        # Based on statistics showing artifacts have high z-scores and few peaks
        if len(reasons) >= 2:
            return 'artifact', details
        else:
            return 'non_artifact', details

    def classify_all_components(self):
        """
        Classify all ICA components.

        Returns
        -------
        dict
            Classification results for all components
        """
        results = {}
        for idx in range(self.ica.n_components_):
            classification, details = self.classify_component(idx)
            results[idx] = {
                'classification': classification,
                'details': details
            }
        return results