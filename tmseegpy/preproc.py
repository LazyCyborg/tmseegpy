# preproc.py

from typing import List, Optional, Callable, Dict, Union, Tuple, Any, TypeVar
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal

# MNE imports
import mne
from mne.preprocessing import compute_proj_ecg, compute_proj_eog, compute_current_source_density, ICA

from mne.preprocessing import ICA

# Required for FASTER bad channel/epoch detection 
from mne_faster import find_bad_channels, find_bad_epochs, find_bad_channels_in_epochs

# Required for artifact cleaning (if using TMSArtifactCleaner)
from sklearn.preprocessing import StandardScaler
import tensorly as tl
from tensorly.decomposition import parafac, non_negative_parafac, tucker
from tqdm import tqdm


## Custom TMS-artefact removal using PARAFAC decomposition
from .clean import TMSArtifactCleaner


def detect_tms_artifacts(raw, threshold_std=10, min_distance_ms=50, existing_events=None):
    """
    Automatically detect TMS artifacts based on amplitude threshold, considering existing events.

    Parameters
    ----------
    threshold_std : float
        Number of standard deviations above mean for detection
    min_distance_ms : float
        Minimum distance between artifacts in milliseconds
    existing_events : array | None
        Existing event samples to avoid duplicate detection

    Returns
    -------
    additional_events : array
        Array of additional events in MNE format (N x 3)
    """
    if raw is None:
        raise ValueError("Must have raw data to detect artifacts")

    data = raw.get_data()
    sfreq = raw.info['sfreq']
    min_distance_samples = int(min_distance_ms * sfreq / 1000)

    print(
        f"Running automatic artifact detection for {min_distance_samples} ms samples with standard deviation {threshold_std}")

    # Calculate statistics across all channels
    data_flat = data.reshape(-1)
    mean = np.mean(data_flat)
    std = np.std(data_flat)
    threshold = mean + threshold_std * std
    print(f"mean: {mean}, std: {std} for data")

    # Find peaks above threshold
    peaks = []
    for ch in range(data.shape[0]):
        channel_peaks = np.where(np.abs(data[ch, :]) > threshold)[0]
        peaks.extend(channel_peaks.tolist())

    if not peaks:
        print("No peaks detected above threshold")
        return None

    peaks = np.unique(np.array(peaks))

    # If we have existing events, prepare exclusion zones
    excluded_zones = []
    if existing_events is not None and len(existing_events) > 0:
        for event_sample in existing_events[:, 0]:
            excluded_zones.append((event_sample - min_distance_samples,
                                   event_sample + min_distance_samples))

    # Group nearby peaks and enforce minimum distance
    artifact_samples = []
    last_peak = -min_distance_samples

    for peak in sorted(peaks):
        # Check if peak is in any exclusion zone
        should_exclude = False
        for start, end in excluded_zones:
            if start <= peak <= end:
                should_exclude = True
                break

        if should_exclude:
            continue

        if peak - last_peak >= min_distance_samples:
            artifact_samples.append(peak)
            last_peak = peak

    if len(artifact_samples) > 0:
        print(f"Detected {len(artifact_samples)} additional TMS artifacts")

        # Create new events array for the additional artifacts
        additional_events = np.zeros((len(artifact_samples), 3), dtype=int)
        additional_events[:, 0] = artifact_samples  # Sample numbers
        additional_events[:, 1] = 0  # Middle column should be 0
        additional_events[:, 2] = 1  # Event ID/value

        # Sort by time
        additional_events = additional_events[additional_events[:, 0].argsort()]

        return additional_events
    else:
        print("No additional artifacts detected")
        return None


class TMSEEGPreprocessor:
    """
    A class for preprocessing TMS-EEG data.
    
    This class implements a preprocessing pipeline for TMS-EEG data,
    including artifact removal, filtering, and data quality checks.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw TMS-EEG data
    montage : str or mne.channels.montage.DigMontage, optional
        The EEG montage to use (default is 'standard_1020')
    ds_sfreq : float, optional
        The desired sampling frequency for resampling (default is 1000 Hz)
        
    Attributes
    ----------
    raw : mne.io.Raw
        The raw TMS-EEG data
    epochs : mne.Epochs
        The epoched TMS-EEG data
    montage : mne.channels.montage.DigMontage
        The EEG montage
    """

    ### currently using easycap-M10 as the standard montage
    ### used standard_1020 before and not sure if it makes a huge difference

    def __init__(self,
                 raw: mne.io.Raw,
                 montage: Union[str, mne.channels.montage.DigMontage] = 'easycap-M1',
                 final_sfreq: float = 725):

        self.raw = raw.copy()
        self.epochs = None
        self.evoked = None
        self.final_sfreq = final_sfreq

        self.first_ica_manual = False
        self.second_ica_manual = False
        self.selected_first_ica_components = []
        self.selected_second_ica_components = []
        self.ica = None
        self.ica2 = None

        self.processing_stage = {
            'initial_removal': False,
            'first_interpolation': False,
            'artifact_cleaning': False,
            'extended_removal': False,
            'final_interpolation': False
        }
        
        # Remove unused EMG channels if present
        for ch in self.raw.info['ch_names']:
            if ch.startswith('EMG'):
                self.raw.drop_channels(ch)
            elif ch.startswith('31'):
                self.raw.drop_channels(ch)
            elif ch.startswith('32'):
                self.raw.drop_channels(ch)
        
        # Channel name standardization
        ch_names = self.raw.ch_names
        rename_dict = {}
        for ch in ch_names:
            # Common naming variations
            if ch in ['31', '32']:
                continue  # Skip non-EEG channels
            if ch.upper() == 'FP1':
                rename_dict[ch] = 'Fp1'
            elif ch.upper() == 'FP2':
                rename_dict[ch] = 'Fp2'
            elif ch.upper() in ['FPZ', 'FPOZ']:
                rename_dict[ch] = 'Fpz'
            elif ch.upper() == 'POZ':
                rename_dict[ch] = 'POz'
            elif ch.upper() == 'PZ':
                rename_dict[ch] = 'Pz'
            elif ch.upper() == 'FCZ':
                rename_dict[ch] = 'FCz'
            elif ch.upper() == 'CPZ':
                rename_dict[ch] = 'CPz'
            elif ch.upper() == 'FZ':
                rename_dict[ch] = 'Fz'
            elif ch.upper() == 'CZ':
                rename_dict[ch] = 'Cz'
            elif ch.upper() == 'OZ':
                rename_dict[ch] = 'Oz'
        
        if rename_dict:
            print("Renaming channels to match standard nomenclature:")
            for old, new in rename_dict.items():
                print(f"  {old} -> {new}")
            self.raw.rename_channels(rename_dict)

        
        self.events = None
        self.event_id = None

        # Initialize attributes that will be set later
        self.stc = None
        self.forward = None
        self.inverse_operator = None
        self.source_space = None
        self.bem_solution = None
        self.noise_cov = None

        self.preproc_stats = {
            'n_orig_events': 0,
            'n_final_events': 0,
            'bad_channels': [],
            'n_bad_epochs': 0,
            'muscle_components': [],
            'excluded_ica_components': [],
            'original_sfreq': 0,
            'interpolated_times': [],
        }






######################## TMS ARTIFACT AND EPOCHS ######################################

    def fix_tms_artifact(self,
                         window: Tuple[float, float] = (-0.002, 0.005),
                         smooth_window: Tuple[float, float] = (-0.002, 0.002),
                         span: int = 2,
                         events: Optional[np.ndarray] = None,
                         verbose: bool = True) -> None:
        """
        Remove TMS artifacts using reversed data and boundary smoothing.

        Parameters
        ----------
        window : tuple
            Start and end time of cut window in seconds (default: (-0.002, 0.005))
        smooth_window : tuple
            Start and end time of smoothing window in seconds (default: (-0.002, 0.002))
        span : int
            Number of samples to use for smoothing on each side (default: 2)
        events : array, optional
            Custom events array (n_events × 3). If None, tries to find events
        verbose : bool
            Whether to print progress information
        """
        if hasattr(self, 'epochs') and self.epochs is not None:
            raise NotImplementedError("TMS pulse removal not yet implemented for epoched data")

        raw_out = self.raw.copy()
        sfreq = raw_out.info['sfreq']

        # Convert windows to samples
        window = np.array([w * sfreq for w in window])
        window_len = int(window[1] - window[0])
        smooth_window = np.array([int(sw * sfreq) for sw in smooth_window])

        # Get events if not provided
        if events is None:
            try:
                events = mne.find_events(raw_out, stim_channel='STI 014')
                if verbose:
                    print(f"\nFound {len(events)} events from STI 014 channel")
            except Exception as e:
                try:
                    events, _ = mne.events_from_annotations(raw_out)
                    if verbose:
                        print(f"\nFound {len(events)} events from annotations")
                except Exception as e2:
                    raise ValueError("No events found or provided. Cannot remove artifacts.")

        if len(events) == 0:
            raise ValueError("No events to process. Cannot remove artifacts.")

        events_sample = events[:, 0]  # Get event sample points

        if verbose:
            print(f"Processing {len(events_sample)} event time points")
            print(f"Window in samples: {window[0]} to {window[1]}")

        # Define the removal function with added validation
        def tms_pulse_removal(y):
            for onset in events_sample:
                cut0 = int(onset + window[0])
                cut1 = int(onset + window[1])

                # Add validation to ensure window is valid
                if cut0 >= cut1:
                    if verbose:
                        print(f"Warning: Invalid window at onset {onset}: cut0={cut0}, cut1={cut1}")
                    continue

                # Calculate the actual window length
                actual_window_len = cut1 - cut0

                # Check if there's enough data before the event
                if cut0 - actual_window_len < 0:
                    if verbose:
                        print(f"Warning: Not enough data before event at {onset} to substitute")
                    continue

                # Substitute data with the reverse of previous data (to remove artifact)
                y[cut0:cut1] = y[cut0 - actual_window_len:cut1 - actual_window_len][::-1]

                # Smooth first "cut"
                smooth_start = int(cut0 + smooth_window[0])
                smooth_end = int(cut0 + smooth_window[1])
                if smooth_start < smooth_end and smooth_start >= 0 and smooth_end < len(y):
                    y[smooth_start:smooth_end] = np.array(
                        [np.mean(y[max(0, samp - span):min(len(y), samp + span + 1)])
                         for samp in range(smooth_start, smooth_end)]
                    )

                # Smooth second "cut"
                smooth_start = int(cut1 + smooth_window[0])
                smooth_end = int(cut1 + smooth_window[1])
                if smooth_start < smooth_end and smooth_start >= 0 and smooth_end < len(y):
                    y[smooth_start:smooth_end] = np.array(
                        [np.mean(y[max(0, samp - span):min(len(y), samp + span + 1)])
                         for samp in range(smooth_start, smooth_end)]
                    )

            return y

        # Apply function to all channels
        raw_out.apply_function(tms_pulse_removal, picks='all', verbose=False)

        # Store info about the removal
        if not hasattr(self, 'tmscut'):
            self.tmscut = []

        self.tmscut.append({
            'window': window,
            'smooth_window': smooth_window,
            'sfreq': sfreq,
            'interpolated': 'no'
        })

        self.raw = raw_out


    def mne_fix_tms_artifact(self,
                             window: Tuple[float, float] = (-0.002, 0.015),
                             mode: str = 'window') -> None:
        """
        Interpolate the TMS artifact using MNE's fix_stim_artifact function.

        Parameters
        ----------
        window : tuple
            Time window around TMS pulse to interpolate (start, end) in seconds
        mode : str
            Interpolation mode ('linear', 'cubic', or 'hann')
        """
        if self.raw is None:
            raise ValueError("Must create raw before interpolating TMS artifact")

        events, event_id = mne.events_from_annotations(self.raw)

        try:
            self.raw = mne.preprocessing.fix_stim_artifact(
                self.raw,
                events=events,
                event_id=event_id,
                tmin=window[0],
                tmax=window[1],
                mode=mode
            )
            print(f"Applied TMS artifact interpolation with mode '{mode}'")
        except Exception as e:
            print(f"Error in TMS artifact interpolation: {str(e)}")


######################### EPOCHS AND REJECTION ################### (story of my life)

    def create_epochs(self,
                      tmin: float = -0.5,
                      tmax: float = 1,
                      baseline: Optional[Tuple[float, float]] = None,
                      amplitude_threshold: float = None,
                      events: Optional[np.ndarray] = None,
                      event_id: Optional[Dict] = None) -> None:
        """
        Create epochs from the continuous data with amplitude rejection criteria.

        Parameters
        ----------
        tmin : float
            Start time of epoch in seconds
        tmax : float
            End time of epoch in seconds
        baseline : tuple or None
            Baseline period (start, end) in seconds
        amplitude_threshold : float
            Threshold for rejecting epochs based on peak-to-peak amplitude in µV
        events : array, optional
            Events to create epochs from
        event_id : dict, optional
            Event IDs to use
        """

        # Verify we have events
        if events is None or len(events) == 0:
            raise ValueError("No events found in the data. Cannot create epochs.")


        print(f"\nCreating epochs with:")
        print(f"Number of events: {len(events)}")
        print(f"Event IDs: {event_id}")
        print(f"Time window: {tmin} to {tmax} seconds")
        if baseline:
            print(f"Baseline period: {baseline}")

        # Store events and event_id
        self.events = events
        self.event_id = event_id

        # Create epochs
        self.epochs = mne.Epochs(
            self.raw,
            events=self.events,
            event_id=self.event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject_by_annotation=True,
            detrend=0,
            preload=True,
            verbose=True
        )

        print(f"\nCreated {len(self.epochs)} epochs")

        # Store preprocessing stats
        self.preproc_stats['n_orig_events'] = len(events)
        self.preproc_stats['n_final_events'] = len(self.epochs)

    def _get_events(self, raw_eve):
        """Get events from epochs or raw data."""
        if self.epochs is not None:
            return self.epochs.events
        elif hasattr(self, 'raw'):
            return mne.find_events(raw_eve, stim_channel='STI 014')
        return None

    def _get_event_ids(self, raw_eve):
        """Get event IDs from epochs or raw data."""
        if self.epochs is not None:
            return self.epochs.event_id
        elif hasattr(self, 'raw'):
            _, event_id = mne.events_from_annotations(raw_eve, event_id='auto')
            return event_id
        return None

    def remove_bad_channels(self, interpolate: bool = False, threshold: int = 2) -> None:
        """
        Remove and interpolate bad channels using FASTER algorithm.

        Parameters
        ----------
        threshold : float
            Threshold for bad channel detection (default = 2)
        """
        if self.epochs is None:
            raise ValueError("Must create epochs before removing bad channels")

        bad_channels = find_bad_channels(self.epochs, thres=threshold)

        if bad_channels:
            print(f"Detected bad channels: {bad_channels}")
            self.epochs.info['bads'] = list(set(self.epochs.info['bads']).union(set(bad_channels)))

            try:
                # First try normal interpolation
                if interpolate:
                    # Try interpolation again
                    self.epochs.interpolate_bads(reset_bads=False)
                    print("Successfully interpolated bad channels using default montage")
                else:
                    self.epochs.drop_channels(self.epochs.info['bads'])

            except ValueError as e:
                print(f"Warning: Standard interpolation failed: {str(e)}")
                print("Attempting alternative interpolation method...")

                try:
                    # Try setting montage again with default positions
                    temp_montage = mne.channels.make_standard_montage(
                        'easycap-M10')  ## standard_1020 was tried before not sure if it makes u huge difference
                    self.epochs.set_montage(temp_montage, match_case=False, on_missing='warn')

                    if interpolate:
                    # Try interpolation again
                        self.epochs.interpolate_bads(reset_bads=False)
                        print("Successfully interpolated bad channels using default montage")
                    else:
                        self.epochs.drop_channels(self.epochs.info['bads'])

                except Exception as e2:
                    print(f"Warning: Alternative interpolation also failed: {str(e2)}")
                    print("Dropping bad channels instead of interpolating")
                    self.epochs.drop_channels(bad_channels)
                    print(f"Dropped channels: {bad_channels}")

            self.preproc_stats['bad_channels'] = bad_channels
        else:
            print("No bad channels detected")

    def remove_bad_epochs(self, threshold: int = 3) -> None:
        """
        Remove bad epochs using FASTER algorithm.

        Parameters
        ----------
        threshold : float
            Threshold for bad epoch detection
        """
        if self.epochs is None:
            raise ValueError("Must create epochs before removing bad epochs")

        bad_epochs = find_bad_epochs(self.epochs, thres=threshold)

        if bad_epochs:
            print(f"Dropping {len(bad_epochs)} bad epochs")
            self.epochs.drop(bad_epochs)
            self.preproc_stats['n_bad_epochs'] = len(bad_epochs)
        else:
            print("No bad epochs detected")

    ######################## TMS ARTIFACT AND EPOCHS ######################################















######################## ICA AND PARAFAC ############################


    from typing import Optional, List

    def run_ica(self,
                output_dir: str,
                session_name: str,
                n_components: int = None,
                method: str = "fastica",
                select_with_nn: bool = False,
                select_with_topo: bool = False,
                use_nn: bool = True,
                use_topo: bool = True,
                topo_edge_threshold: float = 0.15,
                topo_zscore_threshold: float = 3.5,
                topo_peak_threshold: float = 3,
                topo_focal_threshold: float = 0.2,
                ica_callback: Optional[Callable] = None) -> None:
        """
        Run first ICA decomposition with artifact detection.
        Works with both Raw and Epochs data.

        Parameters
        ----------
        output_dir : str
            Directory to save outputs
        session_name : str
            Name of the current session
        method : str
            ICA method ('fastica' or 'infomax')
        n_components : int
            Number of components to use
        select_with_nn : bool
            Whether to automatically select components using neural network classification
        select_with_topo : bool
            Whether to automatically select components using topography-based classification
        use_nn : bool
            Whether to run neural network classification (defaults to True)
        use_topo : bool
            Whether to run topography-based classification (defaults to True)
        topo_edge_threshold, topo_zscore_threshold, topo_peak_threshold, topo_focal_threshold : float
            Parameters for topography-based component selection
        ica_callback : callable, optional
            Not used, kept for backwards compatibility
        """
        # Store copy of data before ICA
        if hasattr(self, 'epochs') and self.epochs is not None:
            inst = self.epochs
            self.epochs_pre_ica = self.epochs.copy()
            is_epochs = True
        else:
            inst = self.raw
            self.raw_pre_ica = self.raw.copy()
            is_epochs = False

        if n_components is None:
            # Set default number of components
            if is_epochs:
                n_channels = len(self.epochs.ch_names)
                n_epochs = len(self.epochs)
                n_components = min(n_channels - 1, n_epochs - 1)
            else:
                n_channels = len(self.raw.ch_names)
                n_components = n_channels - 1

        # Fit ICA
        print("\nFitting ICA...")
        self.ica = ICA(
            n_components=n_components,
            max_iter="auto",
            method=method,
            random_state=42
        )
        self.ica.fit(inst)
        print("ICA fit complete")

        # Initialize storage for classification results
        self.ica_nn_results = None
        self.ica_topo_results = None
        self.selected_first_ica_components = []

        nn_suggested_exclude = []
        topo_suggested_exclude = []

        # Run neural network classification
        if use_nn:
            from .ica_nn_classifier import ICAComponentClassifier, plot_ica_classification, plot_classification_summary

            print("\nRunning neural network-based component classification...")
            try:
                # Initialize classifier
                classifier = ICAComponentClassifier()

                # Classify components
                self.ica_nn_results = classifier.classify_ica(self.ica, inst)

                # Get component classifications
                nn_classifications = self.ica_nn_results['classifications']
                nn_suggested_exclude = self.ica_nn_results['exclude']

                # Print results
                print(f"\nClassified {len(nn_classifications)} components using neural network:")
                for comp_idx, comp_class in nn_classifications.items():
                    prob = self.ica_nn_results['details'][comp_idx]['probability']
                    print(f"  Component {comp_idx}: {comp_class} ({prob:.2f})")

                # Save visualizations
                fig_nn = plot_ica_classification(self.ica, inst, self.ica_nn_results)
                fig_nn.savefig(os.path.join(output_dir, f"{session_name}_nn_classification.png"))
                plt.close(fig_nn)

                fig_summary = plot_classification_summary(self.ica_nn_results)
                fig_summary.savefig(os.path.join(output_dir, f"{session_name}_nn_summary.png"))
                plt.close(fig_summary)

            except Exception as e:
                print(f"\nWarning: Error in neural network classification: {str(e)}")
                print("Neural network classification will not be available")

        # Run topography-based classification
        if use_topo:
            from .ica_topo_classifier import ICATopographyClassifier

            print("\nRunning topography-based component classification...")
            try:
                classifier = ICATopographyClassifier(self.ica, inst)
                classifier.edge_dist_threshold = topo_edge_threshold
                classifier.zscore_threshold = topo_zscore_threshold
                classifier.peak_count_threshold = topo_peak_threshold
                classifier.focal_area_threshold = topo_focal_threshold

                self.ica_topo_results = classifier.classify_all_components()
                topo_suggested_exclude = [idx for idx, res in self.ica_topo_results.items()
                                          if res['classification'] in ['artifact', 'noise']]

                # Print results
                print(f"\nClassified components using topography analysis:")
                for idx, res in self.ica_topo_results.items():
                    print(f"  Component {idx}: {res['classification']}")

            except Exception as e:
                print(f"\nWarning: Error in topography-based classification: {str(e)}")
                print("Topography classification will not be available")

        # Handle component selection based on parameters
        if select_with_nn and nn_suggested_exclude:
            print(
                f"\nAutomatically excluding {len(nn_suggested_exclude)} components based on neural network classification")
            self.ica.apply(inst, exclude=nn_suggested_exclude)
            self.selected_first_ica_components = nn_suggested_exclude
            self.preproc_stats['muscle_components'] = nn_suggested_exclude

        elif select_with_topo and topo_suggested_exclude:
            print(f"\nAutomatically excluding {len(topo_suggested_exclude)} components based on topography")
            self.ica.apply(inst, exclude=topo_suggested_exclude)
            self.selected_first_ica_components = topo_suggested_exclude
            self.preproc_stats['muscle_components'] = topo_suggested_exclude

        else:
            # No components selected for exclusion if no automatic selection is enabled
            print("\nNo components selected for exclusion")
            self.selected_first_ica_components = []
            self.preproc_stats['muscle_components'] = []

        # Update the appropriate data instance
        if is_epochs:
            self.epochs = inst
        else:
            self.raw = inst

    def run_second_ica(self,
                       method: str = "infomax",
                       n_components: int = None,
                       select_with_iclabel: bool = False,
                       use_icalabel: bool = True,
                       icalabel_exclude_labels: List[str] = None,
                       ica_callback: Optional[Callable] = None) -> None:
        """
        Run second ICA with ICLabel detection.
        Works with both Raw and Epochs data.

        Parameters
        ----------
        method : str
            ICA method ('fastica' or 'infomax')
        n_components : int
            Number of components to use
        select_with_iclabel : bool
            Whether to automatically select components using ICA label classification
        use_icalabel : bool
            Whether to run ICA label classification (defaults to True)
        icalabel_exclude_labels : list of str
            List of ICA labels to exclude (if None, excludes all except "brain" and "other")
        ica_callback : callable, optional
            Not used, kept for backwards compatibility
        """
        # Determine if we're working with epochs or raw data
        if hasattr(self, 'epochs') and self.epochs is not None:
            inst = self.epochs
            is_epochs = True
        else:
            inst = self.raw
            is_epochs = False

        if inst is None:
            raise ValueError("No data available for ICA")

        if n_components is None:
            if is_epochs:
                n_channels = len(self.epochs.ch_names)
                n_epochs = len(self.epochs)
                n_components = min(n_channels - 1, n_epochs - 1)
            else:
                n_channels = len(self.raw.ch_names)
                n_components = n_channels - 1

        print("\nPreparing for second ICA...")
        if is_epochs:
            self.set_average_reference()

        # Initialize and fit ICA
        fit_params = dict(extended=True) if method == "infomax" else None
        self.ica2 = ICA(max_iter="auto", n_components=n_components, method=method, random_state=42,
                        fit_params=fit_params)
        self.ica2.fit(inst)
        print("Second ICA fit complete")

        # Initialize storage for classification results
        self.ica_label_results = None
        self.selected_second_ica_components = []
        exclude_idx = []

        # Run ICA label classification
        if use_icalabel:
            from mne_icalabel import label_components

            print("\nRunning ICA label classification for second ICA...")
            try:
                # Set default exclude labels if not provided
                self.ica_label_results= label_components(inst, self.ica2, method="iclabel")
                for i, l in enumerate(self.ica_label_results["labels"]):
                    print(f"ICA components {i}: {l}")
                # Run ICA label classification

                labels= self.ica_label_results["labels"]
                exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
                print(f"Excluding these ICA components in pre: {exclude_idx}")


            except Exception as e:
                print(f"\nError in ICA label classification: {str(e)}")
                print("ICA label classification will not be available")

        # Handle component selection based on parameters
        if select_with_iclabel and exclude_idx:
            print(
                f"\nAutomatically excluding {len(exclude_idx)} components based on ICA label classification: {exclude_idx}")
            self.ica2.apply(inst, exclude=exclude_idx)
            self.selected_second_ica_components = exclude_idx
            self.preproc_stats['excluded_ica_components'] = exclude_idx
        else:
            # No components selected for exclusion if automatic selection is not enabled
            print("\nNo components selected for exclusion")
            self.selected_second_ica_components = []
            self.preproc_stats['excluded_ica_components'] = []

        # Update the appropriate data instance
        if is_epochs:
            self.epochs = inst
        else:
            self.raw = inst

        print('Second ICA complete')


    def clean_muscle_artifacts(self,
                               muscle_window: Tuple[float, float] = (0.005, 0.05),
                               threshold_factor: float = 5.0,
                               n_components: int = 2,
                               verbose: bool = True) -> None:
        """
        Clean TMS-evoked muscle artifacts using tensor decomposition.

        Parameters
        ----------
        muscle_window : tuple
            Time window for detecting muscle artifacts in seconds [start, end]
        threshold_factor : float
            Threshold for artifact detection
        n_components : int
            Number of components to use in tensor decomposition
        verbose : bool
            Whether to print progress information
        """
        if self.epochs is None:
            raise ValueError("Must create epochs before cleaning muscle artifacts")

        # Create cleaner instance
        cleaner = TMSArtifactCleaner(self.epochs, verbose=verbose)

        # Detect artifacts
        artifact_info = cleaner.detect_muscle_artifacts(
            muscle_window=muscle_window,
            threshold_factor=threshold_factor,
            verbose=verbose
        )

        if verbose:
            print("\nArtifact detection results:")
            print(f"Found {artifact_info['muscle']['stats']['n_detected']} artifacts")
            print(f"Detection rate: {artifact_info['muscle']['stats']['detection_rate'] * 100:.1f}%")

        # Clean artifacts
        cleaned_epochs = cleaner.clean_muscle_artifacts(
            n_components=n_components,
            verbose=verbose
        )

        # Update epochs with cleaned data
        self.epochs = cleaned_epochs

        # Apply baseline correction again
        # self.apply_baseline_correction()

        if verbose:
            print("\nMuscle artifact cleaning complete")

    ######################## ICA AND PARAFAC ############################









    ######################## FILTERS ############################

    def filter_raw(self, l_freq=0.1, h_freq=250):
        """
        Filter raw data using a zero-phase Butterworth filter with improved stability.

        Parameters
        ----------
        l_freq : float or None
            Lower frequency cutoff for bandpass filter (default: 0.1 Hz)
        h_freq : float
            Upper frequency cutoff for bandpass filter (default: 45 Hz)
        """
        print("\nFiltering raw data using a zero-phase Butterworth filter")
        try:
            self.raw.filter(l_freq=l_freq,
                  h_freq=h_freq,
                  method = 'iir',
                  iir_params = dict(order=3,
                                    ftype='butter',
                                    phase='zero-double',
                                    btype='bandpass'),
                  verbose=True)

        except Exception as e:
            print(f"Error during filtering: {str(e)}")
            raise

        print("Filtering complete")


    def mne_filter_epochs(self, l_freq=0.1, h_freq=45, notch_freq=50, notch_width=2):
        """
        Filter epoched data using MNE's built-in filtering plus custom notch.

        Parameters
        ----------
        l_freq : float
            Lower frequency bound for bandpass filter
        h_freq : float
            Upper frequency bound for bandpass filter
        notch_freq : float
            Frequency to notch filter (usually power line frequency)
        notch_width : float
            Width of the notch filter

        Returns
        -------
        None
            Updates self.epochs in place
        """
        from scipy.signal import iirnotch, filtfilt
        import numpy as np
        from mne.time_frequency import psd_array_welch

        if self.epochs is None:
            raise ValueError("Must create epochs before filtering")

        # Store original epochs for potential recovery
        original_epochs = self.epochs
        try:
            # Create a deep copy to work with
            filtered_epochs = self.epochs.copy()

            # Get data and sampling frequency
            data = filtered_epochs.get_data()
            sfreq = filtered_epochs.info['sfreq']
            nyquist = sfreq / 2.0

            # Diagnostic before filtering
            psds, freqs = psd_array_welch(data.reshape(-1, data.shape[-1]),
                                          sfreq=sfreq,
                                          fmin=0,
                                          fmax=200,
                                          n_per_seg=256,
                                          n_overlap=128)

            print(f"\nBefore filtering:")
            print(f"Peak frequency: {freqs[np.argmax(psds.mean(0))]} Hz")
            print(f"Frequency range with significant power: {freqs[psds.mean(0) > psds.mean(0).max() * 0.1][0]:.1f} - "
                  f"{freqs[psds.mean(0) > psds.mean(0).max() * 0.1][-1]:.1f} Hz")

            # Apply filters in sequence
            print("\nApplying low-pass filter...")
            filtered_epochs.filter(
                l_freq=None,
                h_freq=h_freq,
                picks='eeg',
                filter_length='auto',
                h_trans_bandwidth=10,
                method='fir',
                fir_window='hamming',
                fir_design='firwin',
                phase='zero',
                verbose=True
            )

            print("\nApplying high-pass filter...")
            filtered_epochs.filter(
                l_freq=l_freq,
                h_freq=None,
                picks='eeg',
                filter_length='auto',
                l_trans_bandwidth=l_freq / 2,
                method='fir',
                fir_window='hamming',
                fir_design='firwin',
                phase='zero',
                verbose=True
            )

            # Get the filtered data for notch filtering
            data = filtered_epochs.get_data()

            print("\nApplying notch filters...")
            for freq in [notch_freq, notch_freq * 2]:
                print(f"Processing {freq} Hz notch...")
                Q = 30.0  # Quality factor
                w0 = freq / nyquist
                b, a = iirnotch(w0, Q)

                # Apply to each epoch and channel
                for epoch_idx in range(data.shape[0]):
                    for ch_idx in range(data.shape[1]):
                        data[epoch_idx, ch_idx, :] = filtfilt(b, a, data[epoch_idx, ch_idx, :])

            # Update the filtered epochs with notch-filtered data
            filtered_epochs._data = data

            # Diagnostic after filtering
            data_filtered = filtered_epochs.get_data()
            psds, freqs = psd_array_welch(data_filtered.reshape(-1, data_filtered.shape[-1]),
                                          sfreq=sfreq,
                                          fmin=0,
                                          fmax=200,
                                          n_per_seg=256,
                                          n_overlap=128)

            print(f"\nAfter filtering:")
            print(f"Peak frequency: {freqs[np.argmax(psds.mean(0))]} Hz")
            print(f"Frequency range with significant power: {freqs[psds.mean(0) > psds.mean(0).max() * 0.1][0]:.1f} - "
                  f"{freqs[psds.mean(0) > psds.mean(0).max() * 0.1][-1]:.1f} Hz")

            # Verify the filtered data
            if np.any(np.isnan(filtered_epochs._data)):
                raise ValueError("Filtering produced NaN values")

            if np.any(np.isinf(filtered_epochs._data)):
                raise ValueError("Filtering produced infinite values")

            # Update the instance's epochs with the filtered version
            self.epochs = filtered_epochs
            print("\nFiltering completed successfully")

        except Exception as e:
            print(f"Error during filtering: {str(e)}")
            print("Reverting to original epochs")
            self.epochs = original_epochs
            raise


    def scipy_filter_epochs(self, l_freq=None, h_freq=45, notch_freq=None, notch_width=2):
        """
        Filter epoched data using a zero-phase Butterworth filter with improved stability.

        Parameters
        ----------
        l_freq : float
            Lower frequency cutoff for bandpass filter (default: 0.1 Hz)
        h_freq : float
            Upper frequency cutoff for bandpass filter (default: 45 Hz)
        notch_freq : float
            Frequency for notch filter (default: 50 Hz)
        notch_width : float
            Width of notch filter (default: 2 Hz)
        """
        from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch
        import numpy as np

        if self.epochs is None:
            raise ValueError("Must create epochs before filtering")

        # Create a copy of the epochs object
        filtered_epochs = self.epochs.copy()

        # Get data and scale it up for better numerical precision
        data = filtered_epochs.get_data()
       # scale_factor = 1e6  # Convert to microvolts
       # data = data * scale_factor

        print(f"Data shape: {data.shape}")
        print(f"Scaled data range: [{np.min(data)}, {np.max(data)}] µV")

        # Ensure data is float64
        data = data.astype(np.float64)

        sfreq = filtered_epochs.info['sfreq']
        nyquist = sfreq / 2

        try:
            # High-pass filter
            sos_high = butter(3, l_freq / nyquist, btype='high', output='sos')
            data = sosfiltfilt(sos_high, data, axis=-1)
            print(f"After high-pass - Data range: [{np.min(data)}, {np.max(data)}] µV")

            # Low-pass filter
            sos_low = butter(5, h_freq / nyquist, btype='low', output='sos')
            data = sosfiltfilt(sos_low, data, axis=-1)
            print(f"After low-pass - Data range: [{np.min(data)}, {np.max(data)}] µV")
            if notch_freq is not None:
                # Multiple notch filters for harmonics
                for freq in [notch_freq, notch_freq * 2]:  # 50 Hz and 100 Hz
                    # Using iirnotch for sharper notch characteristics
                    b, a = iirnotch(freq / nyquist, 35)  # Q=35 for very narrow notch
                    data = filtfilt(b, a, data, axis=-1)
                print(f"After notch - Data range: [{np.min(data)}, {np.max(data)}] µV")

            # Scale back
            #data = data / scale_factor
            filtered_epochs._data = data

        except Exception as e:
            print(f"Error during filtering: {str(e)}")
            raise

        print("Filtering complete")
        self.epochs = filtered_epochs

    ######################## FILTERS ##############################








    ################# SOME FINAL STEPS #####################

    def set_average_reference(self):
        '''
        - Rereference EEG and apply projections
        '''
        self.epochs.set_eeg_reference('average', projection=True)
        print("Rereferenced epochs to 'average'")

    def apply_baseline_correction(self, baseline: Tuple[float, float] = (-0.1, -0.002)) -> None:
        """
        Apply baseline correction to epochs.
        
        Parameters
        ----------
        baseline : tuple
            Start and end time of baseline period in seconds (start, end)
        """
        if self.epochs is None:
            raise ValueError("Must create epochs before applying baseline")
            
        self.epochs.apply_baseline(baseline=baseline)
        print(f"Applied baseline correction using window {baseline} seconds")

    def downsample(self):
        '''
        - Downsample epochs to desired sfreq if current sfreq > desired sfreq (default 1000 Hz)
        '''

        current_sfreq = self.epochs.info['sfreq']
        if current_sfreq > self.ds_sfreq:
            self.epochs = self.epochs.resample(self.ds_sfreq)
            print(f"Downsampled data to {self.ds_sfreq} Hz")
        else:
            print("Current sfreq < target sfreq")
            pass

    def get_preproc_stats(self):
        """Return current preprocessing statistics"""
        return {
            'Original Events': self.preproc_stats['n_orig_events'],
            'Final Events': self.preproc_stats['n_final_events'],
            'Event Retention Rate': f"{(self.preproc_stats['n_final_events']/self.preproc_stats['n_orig_events'])*100:.1f}%",
            'Bad Channels': ', '.join(self.preproc_stats['bad_channels']) if self.preproc_stats['bad_channels'] else 'None',
            'Bad Epochs Removed': self.preproc_stats['n_bad_epochs'],
            'ICA1 Muscle Components': len(self.preproc_stats['muscle_components']),
            'ICA2 Excluded Components': len(self.preproc_stats['excluded_ica_components']),
            'TMS Interpolation Windows': len(self.preproc_stats['interpolated_times'])
    }


    def apply_ssp(self, n_eeg=2):

        
        projs_epochs = mne.compute_proj_epochs(self.epochs, n_eeg=n_eeg, n_jobs=-1, verbose=True)
        self.epochs.add_proj(projs_epochs)
        self.epochs.apply_proj()
        


    def apply_csd(self, lambda2=1e-5, stiffness=4, n_legendre_terms=50, verbose=True):
        """
        Apply Current Source Density transformation maintaining CSD channel type.
        
        Parameters
        ----------
        lambda2 : float
            Regularization parameter
        stiffness : int
            Stiffness of the spline
        n_legendre_terms : int
            Number of Legendre terms
        verbose : bool
            Print progress information
        """
        if verbose:
            print("Applying Current Source Density transformation...")
        
        # Apply CSD transformation
        self.epochs = compute_current_source_density(
            self.epochs,
            lambda2=lambda2,
            stiffness=stiffness,
            n_legendre_terms=n_legendre_terms,
            copy=True
        )
        
        # The channels are now CSD type, so we leave them as is
        if verbose:
            print("CSD transformation complete")
        
        # Store the fact that we've applied CSD
        self.csd_applied = True
        
        return self.epochs



    def final_downsample(self):
        """
        Perform final downsampling of epochs to final_sfreq (default 725 Hz).
        """
        if self.epochs is None:
            raise ValueError("Must create epochs before final downsampling")

        current_sfreq = self.epochs.info['sfreq']
        if current_sfreq > self.final_sfreq:
            self.epochs = self.epochs.resample(self.final_sfreq)
            print(f"Final downsample to {self.final_sfreq} Hz")
        else:
            print(f"Current sfreq ({current_sfreq} Hz) <= final target sfreq ({self.final_sfreq} Hz); "
                  "no final downsampling performed")
    
    def save_epochs(self, fpath: str = None):
        """
        Save preprocessed epochs
        """
        self.epochs.save(fpath, verbose=True, overwrite=True)

        print(f"Epochs saved at {fpath}")
