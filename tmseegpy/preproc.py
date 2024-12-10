# Standard scientific libraries
import numpy as np
from scipy import stats, signal
from scipy.interpolate import interp1d
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.spatial.distance import pdist, squareform
from scipy import optimize
import mne
from mne.preprocessing import compute_current_source_density

from typing import Optional, Union, Dict, List, Tuple, Any
import warnings
import os

# MNE imports
import mne
from mne.minimum_norm import (make_inverse_operator, 
                            apply_inverse,
                            write_inverse_operator)
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne import (compute_raw_covariance,
                read_source_spaces,
                setup_source_space,
                make_bem_model,
                make_bem_solution,
                make_forward_solution,
                read_trans,
                read_bem_solution)
from mne.viz import plot_alignment, plot_bem

# Required for ICA and component labeling
from mne_icalabel import label_components
from mne.preprocessing import ICA

# Required for FASTER bad channel/epoch detection 
from mne_faster import find_bad_channels, find_bad_epochs, find_bad_channels_in_epochs

# Required for artifact cleaning (if using TMSArtifactCleaner)
from sklearn.preprocessing import StandardScaler
import tensorly as tl
from tensorly.decomposition import parafac, non_negative_parafac, tucker
from tqdm import tqdm


## Custom TMS-artefact cleaner
from clean import TMSArtifactCleaner


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
    
    def __init__(self, 
                 raw: mne.io.Raw,
                 montage: Union[str, mne.channels.montage.DigMontage] = 'standard_1020',
                 ds_sfreq: float = 1000):
        
        self.raw = raw.copy()
        self.epochs = None
        self.evoked = None
        self.ds_sfreq = ds_sfreq

        self.processing_stage = {
        'initial_removal': False,
        'first_interpolation': False,
        'artifact_cleaning': False,
        'extended_removal': False,
        'final_interpolation': False
    }
        
        # Set montage
        if isinstance(montage, str):
            self.montage = mne.channels.make_standard_montage(montage)
        else:
            self.montage = montage 
            
        # Remove unused EMG channels if present
        for ch in self.raw.info['ch_names']:
            if ch.startswith('EMG'):
                self.raw.drop_channels(ch)
                
        # Set montage
        self.raw.set_montage(self.montage)
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
        
        
        
    def create_epochs(self, 
                    tmin: float = -0.5, 
                    tmax: float = 1,
                    baseline: Optional[Tuple[float, float]] = None,
                    reject: Optional[Dict[str, float]] = None) -> None:
        """
        Create epochs from the continuous data with optional rejection criteria.
        
        Parameters
        ----------
        tmin : float
            Start time of epoch in seconds
        tmax : float
            End time of epoch in seconds
        baseline : tuple or None
            Baseline period (start, end) in seconds. None for no baseline correction
        reject : dict or None
            Rejection parameters based on peak-to-peak amplitude.
            For example: dict(eeg=200e-6) to reject epochs with peak-to-peak
            amplitude > 200 µV in any EEG channel
        """
        # Default rejection criteria if none provided
        #if reject is None:
           # reject = dict(eeg=200e-6)  # 200 µV for EEG
            
        self.events, self.event_id = mne.events_from_annotations(self.raw)
        
        self.epochs = mne.Epochs(self.raw, 
                            self.events, 
                            event_id=self.event_id,
                            tmin=tmin, 
                            tmax=tmax, 
                            baseline=baseline,
                            reject=reject,  # Add rejection criteria
                            reject_by_annotation=True,
                            detrend=0,  # Remove mean
                            preload=True,
                            verbose=True)
        
        print(f"Created {len(self.epochs)} epochs")
        if len(self.events) > len(self.epochs):
            n_rejected = len(self.events) - len(self.epochs)
            print(f"Rejected {n_rejected} epochs based on amplitude criteria")
        self.preproc_stats['n_orig_events'] = len(self.events)
        self.preproc_stats['n_final_events'] = len(self.epochs)


    def _get_events(self):
        """Get events from epochs or raw data."""
        if self.epochs is not None:
            return self.epochs.events
        elif hasattr(self, 'raw'):
            return mne.find_events(self.raw, stim_channel='STI 014')
        return None

    def _get_event_ids(self):
        """Get event IDs from epochs or raw data."""
        if self.epochs is not None:
            return self.epochs.event_id
        elif hasattr(self, 'raw'):
            _, event_id = mne.events_from_annotations(self.raw, event_id='auto')
            return event_id
        return None
    
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
            print(f"Detection rate: {artifact_info['muscle']['stats']['detection_rate']*100:.1f}%")
        
        # Clean artifacts
        cleaned_epochs = cleaner.clean_muscle_artifacts(
            n_components=n_components,
            verbose=verbose
        )
        
        # Update epochs with cleaned data
        self.epochs = cleaned_epochs
        
        # Apply baseline correction again
        #self.apply_baseline_correction()
        
        if verbose:
            print("\nMuscle artifact cleaning complete")

    def remove_bad_channels(self, threshold: int = 2) -> None:
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
                self.epochs.interpolate_bads(reset_bads=True)
                print("Interpolated bad channels")
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

    def remove_tms_artifact(self, 
                        cut_times_tms: Tuple[float, float] = (-2, 10), 
                        replace_times: Optional[Tuple[float, float]] = None,
                        verbose: bool = True) -> None:
        """
        Remove TMS artifacts exactly following TESA implementation.
        
        Parameters
        ----------
        cut_times_tms : tuple
            Time window to cut around TMS pulse in ms [start, end]
            Default is [-2, 10] following TESA
        replace_times : tuple, optional
            Time window for calculating average to replace removed data in ms [start, end]
            If None (default), data will be replaced with 0s
        """
        raw_out = self.raw.copy()
        data = raw_out.get_data()
        sfreq = raw_out.info['sfreq']
        
        # Store original info about cut (like TESA's EEG.tmscut)
        if not hasattr(self, 'tmscut'):
            self.tmscut = []
        
        tmscut_info = {
            'cut_times_tms': cut_times_tms,
            'replace_times': replace_times,
            'sfreq': sfreq,
            'interpolated': 'no'
        }
        
        cut_samples = np.round(np.array(cut_times_tms) * sfreq / 1000).astype(int)
        
        # Get TMS annotations
        tms_annotations = [ann for ann in raw_out.annotations 
                        if ann['description'] == 'Stimulation']
        
        print(f"\nFound {len(tms_annotations)} TMS events to process")
        print(f"Removing artifact in window {cut_times_tms} ms")

        processed_count = 0
        skipped_count = 0
        
        for ann in tms_annotations:
            event_sample = int(ann['onset'] * sfreq)
            start = event_sample + cut_samples[0]
            end = event_sample + cut_samples[1]
            
            if start < 0 or end >= data.shape[1]:
                skipped_count += 1
                continue
                
            if replace_times is None:
                data[:, start:end] = 0
            else:
                # Calculate average from replace_times window
                replace_samples = np.round(np.array(replace_times) * sfreq / 1000).astype(int)
                baseline_start = event_sample + replace_samples[0]
                baseline_end = event_sample + replace_samples[1]
                if baseline_start >= 0 and baseline_end < data.shape[1]:
                    baseline_mean = np.mean(data[:, baseline_start:baseline_end], axis=1)
                    data[:, start:end] = baseline_mean[:, np.newaxis]
            processed_count += 1
        
        print(f"Successfully removed artifacts from {processed_count} events")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} events due to window constraints")
        
        raw_out._data = data
        raw_out.set_annotations(raw_out.annotations)
        self.raw = raw_out
        self.tmscut.append(tmscut_info)

    def interpolate_tms_artifact(self, 
                            method: str = 'cubic',
                            interp_window: float = 1.0,
                            cut_times_tms: Tuple[float, float] = (-2, 10),  # Add this back
                            verbose: bool = True) -> None:
        """
        Interpolate TMS artifacts exactly following TESA implementation.
        Uses polynomial interpolation rather than spline interpolation.
        
        Parameters
        ----------
        method : str
            Interpolation method: must be 'cubic' for TESA compatibility
        interp_window : float
            Time window (in ms) before and after artifact for fitting cubic function
            Default is 1.0 ms following TESA
        cut_times_tms : tuple
            Time window where TMS artifact was removed in ms [start, end]
            Default is (-2, 10) following TESA
        verbose : bool
            Whether to print progress information
        """
        if not hasattr(self, 'tmscut') or not self.tmscut:
            raise ValueError("Must run remove_tms_artifact first")
        
        print(f"\nStarting interpolation with {method} method")
        print(f"Using interpolation window of {interp_window} ms")
        print(f"Processing cut window {cut_times_tms} ms")

        interpolated_count = 0
        warning_count = 0
            
        raw_out = self.raw.copy()
        data = raw_out.get_data()
        sfreq = raw_out.info['sfreq']
        
        cut_samples = np.round(np.array(cut_times_tms) * sfreq / 1000).astype(int)
        interp_samples = int(round(interp_window * sfreq / 1000))
            
        for tmscut in self.tmscut:
            if tmscut['interpolated'] == 'no':
                cut_times = tmscut['cut_times_tms']
                cut_samples = np.round(np.array(cut_times) * sfreq / 1000).astype(int)
                interp_samples = int(round(interp_window * sfreq / 1000))
                
                # Process annotations
                tms_annotations = [ann for ann in raw_out.annotations 
                                if ann['description'] == 'Stimulation']
                
                for ann in tms_annotations:
                    event_sample = int(ann['onset'] * sfreq)
                    start = event_sample + cut_samples[0]
                    end = event_sample + cut_samples[1]
                    
                    # Calculate fitting windows
                    window_start = start - interp_samples
                    window_end = end + interp_samples
                    
                    if window_start < 0 or window_end >= data.shape[1]:
                        warning_count += 1
                        continue
                    
                    # Get time points for fitting
                    x = np.arange(window_end - window_start + 1)
                    x_fit = np.concatenate([
                        x[:interp_samples],
                        x[-interp_samples:]
                    ])
                    
                    # Center x values at 0 to avoid badly conditioned warnings (TESA approach)
                    x_fit = x_fit - x_fit[0]
                    if len(x) <= 2 * interp_samples:
                        print(f"Warning: Window too small for interpolation at sample {event_sample}")
                        warning_count += 1
                        continue

                    x_interp = x[interp_samples:-interp_samples] - x_fit[0]

                    # Interpolate each channel using polynomial fit
                    for ch in range(data.shape[0]):
                        y_full = data[ch, window_start:window_end+1]
                        y_fit = np.concatenate([
                            y_full[:interp_samples],
                            y_full[-interp_samples:]
                        ])
                        
                        # Use polynomial fit (like TESA) instead of spline
                        p = np.polyfit(x_fit, y_fit, 3)
                        data[ch, start:end+1] = np.polyval(p, x_interp)
                    
                    interpolated_count += 1


                
                tmscut['interpolated'] = 'yes'
        
        print(f"\nSuccessfully interpolated {interpolated_count} events")
        if warning_count > 0:
            print(f"Encountered {warning_count} warnings during interpolation")
        print("TMS artifact interpolation complete")
        raw_out._data = data
        raw_out.set_annotations(raw_out.annotations)
        self.raw = raw_out

    '''def _apply_interpolation(self, data, epoch_idx, start_idx, end_idx, method, sfreq, window_samples):
        if method == 'linear':
            x_fit = np.array([max(0, start_idx-1), min(data.shape[2 if epoch_idx is not None else 1]-1, end_idx+1)])
            x_interp = np.arange(start_idx, end_idx+1)
            
            for ch in range(data.shape[1]):
                y_fit = data[epoch_idx, ch, x_fit.astype(int)] if epoch_idx is not None else data[ch, x_fit.astype(int)]
                f = interp1d(x_fit, y_fit, kind='linear')
                if epoch_idx is not None:
                    data[epoch_idx, ch, start_idx:end_idx+1] = f(x_interp)
                else:
                    data[ch, start_idx:end_idx+1] = f(x_interp)
        else:  # cubic
            window_start = max(0, start_idx - window_samples[0])
            window_end = min(data.shape[2 if epoch_idx is not None else 1]-1, end_idx + window_samples[1])
            
            x_fit = np.concatenate([
                np.arange(window_start, start_idx),
                np.arange(end_idx+1, window_end+1)
            ])
            x_interp = np.arange(start_idx, end_idx+1)
            
            for ch in range(data.shape[1]):
                y_fit = data[epoch_idx, ch, x_fit.astype(int)] if epoch_idx is not None else data[ch, x_fit.astype(int)]
                x_norm = x_fit - x_fit[0]
                x_interp_norm = x_interp - x_fit[0]
                f = interp1d(x_norm, y_fit, kind='cubic')
                if epoch_idx is not None:
                    data[epoch_idx, ch, start_idx:end_idx+1] = f(x_interp_norm)
                else:
                    data[ch, start_idx:end_idx+1] = f(x_interp_norm)'''
    
    def staged_artifact_removal(self, verbose: bool = True) -> None:
        """
        Perform staged TMS artifact removal following TESA pipeline.
        """
        # Stage 1: Initial removal (-2 to 10 ms)
        if verbose:
            print("Stage 1: Initial TMS artifact removal (-2 to 10 ms)")
        self.remove_tms_artifact(cut_times_tms=(-2, 10), verbose=verbose)
        self.interpolate_tms_artifact(method='cubic', 
                                    interp_window=1.0,
                                    cut_times_tms=(-2, 10),
                                    verbose=verbose)
        
        self.run_ica()
        
        # Stage 2: Extended removal (-2 to 15 ms)
        if verbose:
            print("Stage 2: Extended TMS artifact removal (-2 to 15 ms)")
        self.remove_tms_artifact(cut_times_tms=(-2, 15), verbose=verbose)
        self.interpolate_tms_artifact(method='cubic',
                                    interp_window=5.0,
                                    cut_times_tms=(-2, 15),
                                    verbose=verbose)

        if verbose:
            print("Staged artifact removal complete")

    def run_ica(self, method: str = "fastica", tms_muscle_thresh: float = 1.0, plot_components: bool = True) -> None:
        """
        Run first ICA decomposition focusing on TMS-evoked muscle artifacts.
        
        Parameters
        ----------
        method : str
            ICA method to use ('infomax' or 'fastica')
        tms_muscle_thresh : float
            Threshold for muscle component detection
            Default is 1.0 based on empirical testing
        """
        if self.epochs is None:
            raise ValueError("Must create epochs before running ICA")
        # Store pre-ICA epochs
        self.epochs_pre_ica = self.epochs.copy()
        # Set up ICA
        self.ica = ICA(
            max_iter="auto",
            method=method,
            random_state=42
        )
        # Fit ICA
        self.ica.fit(self.epochs)
        # Detect muscle components
        muscle_comps, scores = self.detect_tms_muscle_components(
            tms_muscle_thresh=tms_muscle_thresh,
            verbose=True
        )
        self.preproc_stats['muscle_components'] = muscle_comps
        if plot_components:
            # Plot results
            self.plot_muscle_components(muscle_comps, scores)
        # Apply ICA
        if len(muscle_comps) > 0:
            print(f"Excluding {len(muscle_comps)} muscle components: {muscle_comps}")
            self.ica.apply(self.epochs, exclude=muscle_comps)
        else:
            print("No muscle components detected to exclude")
        # Apply baseline correction maybe not required if we run second ICA
        #self.epochs.apply_baseline(baseline=(-0.1, -0.002))
        #print('Applied baseline corrections post first ICA')

    def filter_raw(self, l_freq=1, h_freq=90, notch_freq=50, notch_width=2, plot_psd=True, fmax=200):
        """
        Apply bandpass and notch filters to raw data.
        
        Parameters
        ----------
        l_freq : float
            Lower cutoff frequency for the bandpass filter
        h_freq : float
            Upper cutoff frequency for the bandpass filter 
        notch_freq : float
            Frequency for the notch filter
        notch_width : float
            Width of the notch filter
        plot_psd : bool
            Whether to plot power spectral density before and after filtering
        fmax: int
            Maximum frequency to display
        """
        if plot_psd:
            # Store copy for before/after comparison
            raw_orig = self.raw.copy()
            
            # Plot PSD before filtering
            fig = raw_orig.compute_psd(fmax=fmax).plot(show=False)
            fig.suptitle('PSD Before Filtering')
        
        # Apply bandpass filter
        self.raw.filter(
            l_freq=l_freq, 
            h_freq=h_freq,
            picks='eeg',
            filter_length='auto',
            l_trans_bandwidth=0.1,
            h_trans_bandwidth=0.5,
            method='fir',
            fir_design='firwin',
            phase='zero',
            verbose=True
        )
        
        # Apply notch filter
        self.raw.notch_filter(
            freqs=notch_freq,
            picks='eeg',
            notch_widths=notch_width,
            filter_length='auto',
            method='fir',
            phase='zero',
            verbose=True
        )
        
        if plot_psd:
            # Plot PSD after filtering
            fig = self.raw.compute_psd(fmax=fmax).plot(show=False)
            fig.suptitle('PSD After Filtering')
            plt.show()
            
        print("Raw data filtering complete")

    def filter_epochs(self, l_freq=1, h_freq=90, notch_freq=50, notch_width=2, notch=True):
        """
        Filter epochs and plot PSD before and after filtering.

        Parameters
        ----------
        l_freq : float
            Lower cutoff frequency for the bandpass filter
        h_freq : float
            Upper cutoff frequency for the bandpass filter 
        notch_freq : float
            Frequency for the notch filter
        notch_width : float
            Width of the notch filter
        notch: bool
            Set to true if you want to apply a notch filter
        """
        if self.epochs is None:
            raise ValueError("Must create epochs before plotting PSD")

        # Before filtering (store copy)
        epochs_orig = self.epochs.copy()

        # Plot PSD before filtering
        fig = epochs_orig.compute_psd(method="multitaper", fmin=0, fmax=100).plot(average=True, show=False)
        fig.suptitle('PSD Before Filtering')

        # Apply bandpass filter
        self.epochs.filter(l_freq=l_freq, 
                        h_freq=h_freq,
                        picks='eeg',
                        filter_length='auto',  
                        l_trans_bandwidth=0.5,  # Specify transition bandwidth for low cutoff
                        h_trans_bandwidth=0.5,  # Specify transition bandwidth for high cutoff
                        method='fir',
                        fir_design='firwin',
                        phase='zero',
                        verbose=True)
        if notch:
            # Apply notch filter
            self.epochs._data = mne.filter.notch_filter(
                self.epochs._data, 
                Fs=self.epochs.info['sfreq'], 
                freqs=notch_freq,
                notch_widths=notch_width, 
                method='fir',
                filter_length='auto',
                phase='zero',
                verbose=True
            )

        # Plot PSD after filtering
        fig = self.epochs.compute_psd(method="multitaper", fmin=0, fmax=100).plot(average=True, show=False)
        fig.suptitle('PSD After Filtering')

        plt.show()

    def run_second_ica(self, method: str = "infomax", 
                    exclude_labels: List[str] = ["eye blink", "muscle artifact", "heart beat", "channel noise"]) -> None:
        """
        Run a second ICA specifically for remaining physiological artifacts using ICLabel.
        
        Parameters
        ----------
        method : str
            ICA method to use ('infomax' or 'fastica')
        exclude_labels : list
            List of ICLabel categories to exclude
        """
        if not hasattr(self, 'epochs'):
            raise ValueError("Must have epochs data to run ICA")
        
        self.set_average_reference()
        if method == "infomax":
            fit_params=dict(extended=True)
        else:
            fit_params=None
        # Set up ICA
        self.ica2 = ICA(
            max_iter="auto",
            method=method,
            random_state=42,
            fit_params=fit_params,
        )
        
        # Fit ICA
        self.ica2.fit(self.epochs)
        
        # Label components with ICLabel
        ic_labels = label_components(self.epochs, self.ica2, method="iclabel")
        labels = ic_labels["labels"]
        
        # Print all component labels
        for n, label in enumerate(labels):
            print(f"Component {n}: {label}")
        
        # Find components to exclude based on ICLabel
        exclude_idx = [
            idx for idx, label in enumerate(labels) if label in exclude_labels
        ]
        
        if len(exclude_idx) > 0:
            print(f"Excluding {len(exclude_idx)} components in second ICA")
            print(f"Excluded components: {exclude_idx}")
            self.ica2.apply(self.epochs, exclude=exclude_idx)
            self.preproc_stats['excluded_ica_components'] = exclude_idx

        else:
            print("No components excluded in second ICA")

        self.epochs.apply_baseline(baseline=(-0.1, -0.002))
        print('Applied baseline corrections post second ICA')

    def detect_tms_muscle_components(self, 
                                tms_muscle_window: Tuple[float, float] = (11, 30), 
                                tms_muscle_thresh: float = 2,  
                                plot_window: Tuple[float, float] = (-200, 500),  
                                verbose: bool = True) -> Tuple[List[int], Dict]:
        """
        Detect TMS-evoked muscle artifacts in ICA components.
        Python implementation of TESA's tesa_compselect muscle detection.
        
        Parameters
        ----------
        tms_muscle_window : tuple
            Time window (in ms) for detecting TMS-evoked muscle activity [start, end]
            Default is (11, 30) ms post-TMS
        tms_muscle_thresh : float
            Threshold for detecting muscle components
            Ratio of mean absolute z-score in muscle window vs. entire time course
        plot_window : tuple
            Time window for plotting in ms [start, end]
        verbose : bool
            Whether to print verbose output
            
        Returns
        -------
        muscle_components : list
            Indices of components classified as muscle artifacts
        component_scores : dict
            Dictionary containing scores and metrics for each component
        """
        if not hasattr(self, 'ica'):
            raise ValueError("Must run ICA before detecting muscle components")
            
        if verbose:
            print(f"Analyzing {self.ica.n_components_} components")
            print(f"Using muscle window: {tms_muscle_window}ms")
            print(f"Using threshold: {tms_muscle_thresh}")
        
        # Get ICA components
        components = self.ica.get_sources(self.epochs)
        n_components = components.get_data().shape[1]
        
        # Get sampling rate and convert windows to samples
        sfreq = self.epochs.info['sfreq']
        tms_muscle_window_samples = np.array([np.abs(self.epochs.times - w/1000).argmin() 
                                            for w in tms_muscle_window])
        
        # Initialize outputs
        muscle_components = []
        component_scores = {
            'muscle_ratios': np.zeros(n_components),
            'window_scores': np.zeros(n_components),
            'total_scores': np.zeros(n_components),
        }
        
        # Process each component
        for comp_idx in range(n_components):
            if verbose and comp_idx % 5 == 0:
                print(f"\nAnalyzing component {comp_idx}")
                
            # Get component time course
            comp_data = components.get_data()[:, comp_idx, :].reshape(-1)
            
            # Z-score the data
            comp_z = zscore(comp_data)
            
            # Calculate scores
            muscle_score = np.abs(comp_z)
            window_score = np.mean(muscle_score[tms_muscle_window_samples[0]:tms_muscle_window_samples[1]])
            total_score = np.mean(muscle_score)
            muscle_ratio = window_score / total_score
            
            if verbose:
                print(f"Comp. {comp_idx} TMS-evoked muscle ratio is {muscle_ratio:.2f}.")
                
            # Store metrics
            component_scores['muscle_ratios'][comp_idx] = muscle_ratio
            component_scores['window_scores'][comp_idx] = window_score
            component_scores['total_scores'][comp_idx] = total_score
            
            # Classify component
            if muscle_ratio >= tms_muscle_thresh:
                muscle_components.append(comp_idx)
        
        return muscle_components, component_scores

    def plot_muscle_components(self, muscle_components: List[int], 
                            component_scores: Dict,
                            plot_window: Tuple[float, float] = (-100, 250)) -> None:
        """
        Plot detected muscle components using MNE's plotting functions.
        
        Parameters
        ----------
        muscle_components : list
            Indices of components classified as muscle artifacts
        component_scores : dict
            Dictionary containing component metrics 
        plot_window : tuple
            Time window for plotting in ms [start, end]
        """
        if not hasattr(self, 'ica'):
            raise ValueError("Must run ICA before plotting components")
            
        if len(muscle_components) == 0:
            print("No muscle components detected to plot")
            
            # Plot muscle ratios
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(component_scores['muscle_ratios'])), 
                    component_scores['muscle_ratios'])
            plt.axhline(y=2, color='r', linestyle='--', label='Threshold')
            plt.xlabel('Component')
            plt.ylabel('Muscle Ratio')
            plt.title('TMS-Evoked Muscle Activity')
            plt.legend()
            plt.show()
            return
        with mne.viz.use_browser_backend("matplotlib"):
            self.ica.plot_sources(self.epochs)
        # Plot detected muscle components
        self.ica.plot_components(picks=muscle_components)
        
        
        '''
        # Plot component properties 
        for comp in muscle_components:
            self.ica.plot_properties(self.epochs, picks=comp, 
                                psd_args=dict(fmax=70))
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot muscle ratios
        bars = ax.bar(range(len(component_scores['muscle_ratios'])), 
                    component_scores['muscle_ratios'])
        ax.axhline(y=2, color='r', linestyle='--', label='Threshold')
        
        # Color the muscle components
        for comp in muscle_components:
            bars[comp].set_color('red')
        
        ax.set_xlabel('Component')
        ax.set_ylabel('Muscle Ratio')
        ax.set_title('TMS-Evoked Muscle Activity')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        '''
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


    def plot_evoked_response(self, picks: Optional[str] = None,
                            ylim: Dict = {'eeg': [-7, 7]},
                            xlim: Optional[Tuple[float, float]] = (-0.1, 0.3),
                            title: str = 'Evoked Response',
                            show: bool = True) -> None:
        """
        Plot averaged evoked response with butterfly plot and global field power.
        
        Parameters
        ----------
        picks : str or list
            Channels to include in plot
        ylim : dict
            Y-axis limits for different channel types
        xlim : tuple
            X-axis limits in seconds (start_time, end_time)
        title : str
            Title for the plot
        show : bool
            Whether to show the plot immediately
        """
        if self.epochs is None:
            raise ValueError("Must create epochs before plotting evoked response")
            
        # Create evoked from epochs
        evoked = self.epochs.average()
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(12, 8))
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
        
        # Butterfly plot
        ax1 = fig.add_subplot(gs[0])
        evoked.plot(picks=picks, axes=ax1, ylim=ylim, xlim=xlim, show=False)
        ax1.set_title(f'{title} - Butterfly Plot')
        
        # GFP plot
        ax2 = fig.add_subplot(gs[1])
        gfp = np.std(evoked.data, axis=0) * 1e6  # Convert to μV
        times = evoked.times
        ax2.plot(times, gfp, 'b-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('GFP (μV)')
        ax2.set_title('Global Field Power')
        ax2.grid(True)
        if xlim is not None:
            ax2.set_xlim(xlim)
        
        # Add vertical line at t=0
        for ax in [ax1, ax2]:
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig


    def apply_ssp(self, n_eeg=2):

        
        projs_epochs = mne.compute_proj_epochs(self.epochs, n_eeg=n_eeg, n_jobs=-1, verbose=True)
        self.epochs.add_proj(projs_epochs)
        self.epochs.apply_proj()
        
        
    def plot_epochs(self, ylim: Optional[Dict] = None) -> None:
        """
        Plot epochs with inverted y-axis.
        
        Parameters
        ----------
        ylim : dict or None
            Y-axis limits for plotting
        """
        if self.epochs is None:
            raise ValueError("No epochs available to plot")
            
        with mne.viz.use_browser_backend("matplotlib"):
            fig = self.epochs.copy().plot()
            for ax in fig.get_axes():
                if hasattr(ax, 'invert_yaxis'):
                    ax.invert_yaxis()
            fig.canvas.draw()
            
    def plot_ica_components(self) -> None:
        """
        Plot ICA components if ICA has been run.
        """
        if not hasattr(self, 'ica'):
            raise ValueError("Must run ICA before plotting components")
            
        with mne.viz.use_browser_backend("matplotlib"):    
            self.ica.plot_sources(self.epochs, show_scrollbars=True)



    def apply_csd(self, lambda2=1e-5, stiffness=4, n_legendre_terms=50, verbose=True):
        """
        Apply Current Source Density transformation.
        
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
        
        # Convert raw data to CSD
        self.epochs = compute_current_source_density(
            self.epochs,
            lambda2=lambda2,
            stiffness=stiffness,
            n_legendre_terms=n_legendre_terms,
            copy=True
        )
        
        if verbose:
            print("CSD transformation complete")
            

    def fix_tms_artifact(self, 
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


    def downsample(self, target_sfreq: float = None):
        """
        Downsample epochs to the desired sampling frequency if the current sampling frequency is higher.

        Parameters
        ----------
        target_sfreq : float or None
            The target sampling frequency. If None, uses self.ds_sfreq.
        """
        if target_sfreq is None:
            target_sfreq = self.ds_sfreq
        current_sfreq = self.epochs.info['sfreq']
        if current_sfreq > target_sfreq:
            self.epochs = self.epochs.resample(target_sfreq)
            print(f"Downsampled data to {target_sfreq} Hz")
        else:
            print("Current sfreq <= target sfreq; no downsampling performed")
    
    def save_epochs(self, fpath: str = None):
        """
        Save preprocessed epochs
        """
        self.epochs.save(fpath, verbose=True, overwrite=True)

        print(f"Epochs saved at {fpath}")
