# preproc.py
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
from mne.io.constants import FIFF
from matplotlib.widgets import CheckButtons, Button
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
import threading
import queue
import mne
from mne.preprocessing import ICA
from typing import List

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
#from .mne_icalabel import label_components
# A working version of mne_icalabel is found on the backup branch of tmseegpy.
# I currently disabled ica_label functionality since it is not used but the references to it are only commented out
from mne.preprocessing import ICA

# Required for FASTER bad channel/epoch detection 
from mne_faster import find_bad_channels, find_bad_epochs, find_bad_channels_in_epochs

# Required for artifact cleaning (if using TMSArtifactCleaner)
from sklearn.preprocessing import StandardScaler
import tensorly as tl
from tensorly.decomposition import parafac, non_negative_parafac, tucker
from tqdm import tqdm


## Custom TMS-artefact cleaner
from .clean import TMSArtifactCleaner


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
                 initial_sfreq: float = 1000,
                 final_sfreq: float = 725):

        self.raw = raw.copy()
        self.epochs = None
        self.evoked = None
        self.initial_sfreq = initial_sfreq
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
        
        # Set montage with error handling
        if isinstance(montage, str):
            try:
                self.montage = mne.channels.make_standard_montage(montage)
            except ValueError as e:
                print(f"Warning: Could not create montage '{montage}': {str(e)}")
                print("Falling back to standard_1020 montage")
                self.montage = mne.channels.make_standard_montage('standard_1020')
        else:
            self.montage = montage
        
        try:
            # First try to set montage normally
            self.raw.set_montage(self.montage)
        except ValueError as e:
            print(f"\nWarning: Could not set montage directly: {str(e)}")
            
            # Get the channel types
            ch_types = {ch: self.raw.get_channel_types(picks=ch)[0] for ch in self.raw.ch_names}
            
            # Identify non-EEG channels
            non_eeg = [ch for ch, type_ in ch_types.items() if type_ not in ['eeg', 'unknown']]
            if non_eeg:
                print(f"\nFound non-EEG channels: {non_eeg}")
                print("Setting their types explicitly...")
                for ch in non_eeg:
                    self.raw.set_channel_types({ch: 'misc'})
            
            # Try setting montage again with on_missing='warn'
            try:
                self.raw.set_montage(self.montage, on_missing='warn')
                print("\nMontage set successfully with warnings for missing channels")
            except Exception as e2:
                print(f"\nWarning: Could not set montage even with warnings: {str(e2)}")
                print("Continuing without montage. Some functionality may be limited.")
        
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
                    amplitude_threshold: float = 300.0) -> None:
        """
        Create epochs from the continuous data with amplitude rejection criteria.
        
        Parameters
        ----------
        tmin : float
            Start time of epoch in seconds
        tmax : float
            End time of epoch in seconds
        baseline : tuple or None
            Baseline period (start, end) in seconds. None for no baseline correction
       # amplitude_threshold : float
       #     Threshold for rejecting epochs based on peak-to-peak amplitude in µV.
        #    Default is 300 µV.
        """
        # Convert µV to V for MNE
        #reject = dict(eeg=amplitude_threshold * 1e-6)
                
        self.events, self.event_id = mne.events_from_annotations(self.raw)
        
        self.epochs = mne.Epochs(self.raw, 
                            self.events, 
                            event_id=self.event_id,
                            tmin=tmin, 
                            tmax=tmax, 
                            baseline=baseline,
                            #reject=reject,
                            reject_by_annotation=True,
                            detrend=0,
                            preload=True,
                            verbose=True)
        
        print(f"Created {len(self.epochs)} epochs")
       # if len(self.events) > len(self.epochs):
        #    n_rejected = len(self.events) - len(self.epochs)
         #   print(f"Rejected {n_rejected} epochs based on {amplitude_threshold}µV amplitude threshold")
       # self.preproc_stats['n_orig_events'] = len(self.events)
       # self.preproc_stats['n_final_events'] = len(self.epochs)


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
            
            try:
                # First try normal interpolation
                self.epochs.interpolate_bads(reset_bads=True)
                print("Interpolated bad channels")
                
            except ValueError as e:
                print(f"Warning: Standard interpolation failed: {str(e)}")
                print("Attempting alternative interpolation method...")
                
                try:
                    # Try setting montage again with default positions
                    temp_montage = mne.channels.make_standard_montage('standard_1020')
                    self.epochs.set_montage(temp_montage, match_case=False, on_missing='warn')
                    
                    # Try interpolation again
                    self.epochs.interpolate_bads(reset_bads=True)
                    print("Successfully interpolated bad channels using default montage")
                    
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

    def remove_tms_artifact(self, 
                        cut_times_tms: Tuple[float, float] = (-2, 10), 
                        replace_times: Optional[Tuple[float, float]] = None,
                        verbose: bool = True) -> None:
        """
        Remove TMS artifacts following TESA implementation.
        
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
        Interpolate TMS artifacts following TESA implementation.
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


    from typing import Optional, List
    import threading
    import tkinter as tk

    def run_ica(self,
                output_dir: str,
                session_name: str,
                method: str = "fastica",
                tms_muscle_thresh: float = 2.0,
                blink_thresh: float = 2.5,
                lat_eye_thresh: float = 2.0,
                muscle_thresh: float = 0.6,
                noise_thresh: float = 4.0,
                plot_components: bool = False,
                manual_mode: bool = False) -> None:
        """
        Run first ICA decomposition with TESA artifact detection.
        Works with both Raw and Epochs data.

        Parameters
        ----------
        output_dir : str
            Directory to save outputs
        session_name : str
            Name of the current session
        method : str
            ICA method ('fastica' or 'infomax')
        tms_muscle_thresh : float
            Threshold for TMS-muscle artifact detection
        blink_thresh : float
            Threshold for blink detection
        lat_eye_thresh : float
            Threshold for lateral eye movement detection
        muscle_thresh : float
            Threshold for muscle artifact detection
        noise_thresh : float
            Threshold for noise detection
        plot_components : bool
            Whether to plot ICA components
        manual_mode : bool
            Whether to use manual component selection
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

        # Fit ICA
        print("\nFitting ICA...")
        self.ica = ICA(
            max_iter="auto",
            method=method,
            random_state=42
        )
        self.ica.fit(inst)
        print("ICA fit complete")

        if manual_mode:
            # Get the main Tk root window
            import builtins
            root = getattr(builtins, 'GUI_MAIN_ROOT', None)
            if root is None:
                raise RuntimeError("No main Tk root found")

            self.first_ica_manual = True
            print("\nStarting manual component selection...")
            print("A new window will open for component selection.")

            from .gui.ica_handler import ICAComponentSelector, ICAComponentSelectorContinuous

            if is_epochs:
                selector_class = ICAComponentSelector
            else:
                selector_class = ICAComponentSelectorContinuous

            try:
                # Run all TESA artifact detection methods
                print("\nRunning TESA artifact detection...")
                artifact_results = self.detect_all_artifacts(
                    tms_muscle_thresh=tms_muscle_thresh,
                    blink_thresh=blink_thresh,
                    lat_eye_thresh=lat_eye_thresh,
                    muscle_freq_thresh=muscle_thresh,
                    noise_thresh=noise_thresh,
                    verbose=True
                )

                # Calculate component scores for GUI
                component_scores = {
                    'blink': artifact_results['blink']['scores']['z_scores'],
                    'lat_eye': artifact_results['lateral_eye']['scores']['z_scores'],
                    'muscle': artifact_results['muscle']['scores']['power_ratios'],
                    'noise': artifact_results['noise']['scores']['max_z_scores']
                }

                # Add TMS-muscle scores if using epoched data
                if is_epochs:
                    component_scores['tms_muscle'] = artifact_results['tms_muscle']['scores']['ratios']

            except Exception as e:
                print(f"\nWarning: Error in component analysis: {str(e)}")
                print("Continuing with manual selection without automatic scores")
                component_scores = None

            # Create synchronization primitives
            selection_complete = threading.Event()
            result_queue = queue.Queue()

            def handle_selection(components):
                try:
                    result_queue.put(components)
                finally:
                    selection_complete.set()

            def create_selector():
                try:
                    selector = selector_class(root)
                    if is_epochs:
                        # Epoched data -> pass epochs
                        selector.select_components(
                            ica_instance=self.ica,
                            epochs=inst,
                            title="First ICA - Select Components to Remove",
                            callback=handle_selection,
                            component_scores=component_scores
                        )
                    else:
                        # Continuous data -> pass raw
                        selector.select_components(
                            ica_instance=self.ica,
                            raw=inst,
                            title="First ICA - Select Components to Remove",
                            callback=handle_selection,
                            component_scores=component_scores
                        )
                except Exception as e:
                    print(f"Error creating selector: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    selection_complete.set()
                    result_queue.put([])

            # Schedule selector creation
            if threading.current_thread() is threading.main_thread():
                create_selector()
            else:
                root.after(0, create_selector)

            if not selection_complete.wait(timeout=300):
                print("\nComponent selection timed out")
                return

            try:
                selected_components = result_queue.get_nowait()
            except queue.Empty:
                print("\nNo components selected")
                return

            if selected_components:
                print(f"\nExcluding {len(selected_components)} manually selected components: {selected_components}")
                self.ica.apply(inst, exclude=selected_components)
                self.selected_first_ica_components = selected_components
                self.preproc_stats['muscle_components'] = selected_components
            else:
                print("\nNo components selected for exclusion")
                self.preproc_stats['muscle_components'] = []

        else:
            # Automatic detection using TESA methods
            artifact_results = self.detect_all_artifacts(
                tms_muscle_thresh=tms_muscle_thresh,
                blink_thresh=blink_thresh,
                lat_eye_thresh=lat_eye_thresh,
                muscle_freq_thresh=muscle_thresh,
                noise_thresh=noise_thresh,
                verbose=True
            )

            # Combine all detected components
            exclude_components = []
            for key in artifact_results:
                if key == 'tms_muscle' and not is_epochs:
                    continue  # Skip TMS-muscle components for raw data
                exclude_components.extend(artifact_results[key]['components'])
            exclude_components = list(set(exclude_components))  # Remove duplicates

            if exclude_components:
                print(f"\nExcluding {len(exclude_components)} components: {exclude_components}")
                self.ica.apply(inst, exclude=exclude_components)
                self.preproc_stats['muscle_components'] = exclude_components
            else:
                print("\nNo components detected to exclude")
                self.preproc_stats['muscle_components'] = []

        # Update the appropriate data instance
        if is_epochs:
            self.epochs = inst
        else:
            self.raw = inst

    def run_second_ica(self,
                       method: str = "infomax",
                       exclude_labels: List[str] = ["eye blink", "heart beat", "muscle artifact", "channel noise",
                                                    "line noise"],
                       blink_thresh: float = 2.5,
                       lat_eye_thresh: float = 2.0,
                       muscle_thresh: float = 0.6,
                       noise_thresh: float = 4.0,
                       manual_mode: bool = False) -> None:
        """
        Run second ICA with both TESA and ICLabel detection methods.
        Works with both Raw and Epochs data.

        Parameters
        ----------
        method : str
            ICA method ('fastica' or 'infomax')
        exclude_labels : list of str
            Labels of components to exclude if using ICLabel
        blink_thresh : float
            Threshold for blink detection
        lat_eye_thresh : float
            Threshold for lateral eye movement detection
        muscle_thresh : float
            Threshold for muscle artifact detection
        noise_thresh : float
            Threshold for noise detection
        manual_mode : bool
            Whether to use manual component selection
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

        print("\nPreparing for second ICA...")
        if is_epochs:
            self.set_average_reference()

        # Initialize and fit ICA
        fit_params = dict(extended=True) if method == "infomax" else None
        self.ica2 = ICA(max_iter="auto", method=method, random_state=42, fit_params=fit_params)
        self.ica2.fit(inst)
        print("Second ICA fit complete")

        if manual_mode:
            import builtins
            root = getattr(builtins, 'GUI_MAIN_ROOT', None)
            if root is None:
                raise RuntimeError("No main Tk root found")

            from .gui.ica_handler import ICAComponentSelector, ICAComponentSelectorContinuous

            if is_epochs:
                selector_class = ICAComponentSelector
            else:
                selector_class = ICAComponentSelectorContinuous

            self.second_ica_manual = True
            print("\nStarting manual component selection for second ICA...")

            try:
                # Run TESA artifact detection (excluding TMS-muscle for continuous data)
                print("\nRunning TESA artifact detection...")
                artifact_results = self.detect_all_artifacts(
                    blink_thresh=blink_thresh,
                    lat_eye_thresh=lat_eye_thresh,
                    muscle_freq_thresh=muscle_thresh,
                    noise_thresh=noise_thresh,
                    verbose=True
                )

                # Calculate component scores (exclude TMS-muscle for continuous data)
                component_scores = {
                    'blink': artifact_results['blink']['scores']['z_scores'],
                    'lat_eye': artifact_results['lateral_eye']['scores']['z_scores'],
                    'muscle': artifact_results['muscle']['scores']['power_ratios'],
                    'noise': artifact_results['noise']['scores']['max_z_scores']
                }

                # Add TMS-muscle scores if using epoched data
                if is_epochs:
                    component_scores['tms_muscle'] = artifact_results['tms_muscle']['scores']['ratios']

                # Combine suggestions
                suggested_exclude = []
                for key in artifact_results:
                    if key == 'tms_muscle' and not is_epochs:
                        continue  # Skip TMS-muscle components for raw data
                    suggested_exclude.extend(artifact_results[key]['components'])
                suggested_exclude = list(set(suggested_exclude))

                if suggested_exclude:
                    print(f"\nSuggested components for removal: {suggested_exclude}")
                    print("(Based on TESA artifact detection)")

            except Exception as e:
                print(f"\nWarning: Error in component analysis: {str(e)}")
                component_scores = None

            # Create synchronization primitives
            selection_complete = threading.Event()
            result_queue = queue.Queue()

            def handle_selection(components):
                try:
                    result_queue.put(components)
                finally:
                    selection_complete.set()

            def create_selector():
                try:
                    selector = selector_class(root)
                    if is_epochs:
                        # Epoched data -> pass epochs
                        selector.select_components(
                            ica_instance=self.ica2,
                            epochs=inst,
                            title="Second - Select Components to Remove",
                            callback=handle_selection,
                            component_scores=component_scores
                        )
                    else:
                        # Continuous data -> pass raw
                        selector.select_components(
                            ica_instance=self.ica2,
                            raw=inst,
                            title="Second ICA - Select Components to Remove",
                            callback=handle_selection,
                            component_scores=component_scores
                        )
                except Exception as e:
                    print(f"Error creating selector: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    selection_complete.set()
                    result_queue.put([])

            if threading.current_thread() is threading.main_thread():
                create_selector()
            else:
                root.after(0, create_selector)

            if not selection_complete.wait(timeout=300):
                print("\nComponent selection timed out")
                return

            try:
                selected_components = result_queue.get_nowait()
            except queue.Empty:
                print("\nNo components selected")
                return

            if selected_components:
                print(f"\nExcluding {len(selected_components)} manually selected components: {selected_components}")
                self.ica2.apply(inst, exclude=selected_components)
                self.selected_second_ica_components = selected_components
                self.preproc_stats['excluded_ica_components'] = selected_components
            else:
                print("\nNo components selected for exclusion")
                self.preproc_stats['excluded_ica_components'] = []

        else:
            # Automatic detection using TESA methods
            try:
                artifact_results = self.detect_all_artifacts(
                    blink_thresh=blink_thresh,
                    lat_eye_thresh=lat_eye_thresh,
                    muscle_freq_thresh=muscle_thresh,
                    noise_thresh=noise_thresh,
                    verbose=True
                )

                # Combine detected components
                exclude_idx = []
                for key in artifact_results:
                    if key == 'tms_muscle' and not is_epochs:
                        continue  # Skip TMS-muscle components for raw data
                    exclude_idx.extend(artifact_results[key]['components'])
                exclude_idx = list(set(exclude_idx))

                if exclude_idx:
                    print(f"\nExcluding {len(exclude_idx)} components: {exclude_idx}")
                    self.ica2.apply(inst, exclude=exclude_idx)
                    self.preproc_stats['excluded_ica_components'] = exclude_idx
                else:
                    print("\nNo components excluded")
                    self.preproc_stats['excluded_ica_components'] = []

            except Exception as e:
                print(f"Warning: Error in automatic component detection: {str(e)}")
                print("No components will be automatically excluded")
                self.preproc_stats['excluded_ica_components'] = []

        # Update the appropriate data instance
        if is_epochs:
            self.epochs = inst
        else:
            self.raw = inst

        print('Second ICA complete')

    def filter_raw(self, l_freq=0.1, h_freq=45, notch_freq=50, notch_width=2):
        """
        Filter raw data using a zero-phase Butterworth filter with improved stability.

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

        print(f"Applying SciPy filters to raw data with frequency {l_freq}Hz and frequency {h_freq}Hz")

        # Create a copy of the raw data
        filtered_raw = self.raw.copy()

        # Get data and scale it up for better numerical precision
        data = filtered_raw.get_data()
        scale_factor = 1e6  # Convert to microvolts
        data = data * scale_factor

        print(f"Data shape: {data.shape}")
        print(f"Scaled data range: [{np.min(data)}, {np.max(data)}] µV")

        # Ensure data is float64
        data = data.astype(np.float64)

        sfreq = filtered_raw.info['sfreq']
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

            # Multiple notch filters for harmonics
            for freq in [notch_freq, notch_freq * 2]:  # 50 Hz and 100 Hz
                # Using iirnotch for sharper notch characteristics
                b, a = iirnotch(freq / nyquist, 35)  # Q=35 for very narrow notch
                data = filtfilt(b, a, data, axis=-1)
            print(f"After notch - Data range: [{np.min(data)}, {np.max(data)}] µV")

            # Scale back
            data = data / scale_factor
            filtered_raw._data = data

        except Exception as e:
            print(f"Error during filtering: {str(e)}")
            raise

        print("Filtering complete")
        self.raw = filtered_raw

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


    def scipy_filter_epochs(self, l_freq=0.1, h_freq=45, notch_freq=50, notch_width=2):
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
        scale_factor = 1e6  # Convert to microvolts
        data = data * scale_factor

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

            # Multiple notch filters for harmonics
            for freq in [notch_freq, notch_freq * 2]:  # 50 Hz and 100 Hz
                # Using iirnotch for sharper notch characteristics
                b, a = iirnotch(freq / nyquist, 35)  # Q=35 for very narrow notch
                data = filtfilt(b, a, data, axis=-1)
            print(f"After notch - Data range: [{np.min(data)}, {np.max(data)}] µV")

            # Scale back
            data = data / scale_factor
            filtered_epochs._data = data

        except Exception as e:
            print(f"Error during filtering: {str(e)}")
            raise

        print("Filtering complete")
        self.epochs = filtered_epochs

    def detect_all_artifacts(self,
                             tms_muscle_window=(11, 30),
                             tms_muscle_thresh=2,
                             blink_thresh=2.5,
                             lat_eye_thresh=2.0,
                             muscle_freq_window=(30, 100),
                             muscle_freq_thresh=0.6,
                             noise_thresh=4.0,
                             verbose=True) -> Dict:
        """
        Detect all artifact types following TESA's implementation.
        Works with both Raw and Epochs data.

        Parameters
        ----------
        tms_muscle_window : tuple
            Time window (ms) for detecting TMS-evoked muscle activity
        tms_muscle_thresh : float
            Threshold for TMS-evoked muscle components
        blink_thresh : float
            Threshold for blink components
        lat_eye_thresh : float
            Threshold for lateral eye movement components
        muscle_freq_window : tuple
            Frequency window (Hz) for detecting persistent muscle activity
        muscle_freq_thresh : float
            Threshold for persistent muscle components
        noise_thresh : float
            Threshold for electrode noise components
        verbose : bool
            Whether to print verbose output

        Returns
        -------
        dict
            Dictionary containing detected components and their scores
        """
        if not hasattr(self, 'ica'):
            raise ValueError("Must run ICA before detecting components")

        # Initialize results dictionary
        results = {
            'tms_muscle': {'components': [], 'scores': {}},
            'blink': {'components': [], 'scores': {}},
            'lateral_eye': {'components': [], 'scores': {}},
            'muscle': {'components': [], 'scores': {}},
            'noise': {'components': [], 'scores': {}}
        }

        # Get ICA weights
        weights = self.ica.get_components()
        n_components = self.ica.n_components_

        # Get ICA components (sources)
        if hasattr(self, 'epochs') and self.epochs is not None:
            inst = self.epochs
            components = self.ica.get_sources(inst)
            is_epochs = True
        else:
            inst = self.raw
            components = self.ica.get_sources(inst)
            is_epochs = False

        # 1. Detect TMS-evoked muscle artifacts (if using epoched data)
        if is_epochs:
            muscle_comps, muscle_scores = self._detect_tms_muscle(
                components, tms_muscle_window, tms_muscle_thresh)
            results['tms_muscle']['components'] = muscle_comps
            results['tms_muscle']['scores'] = muscle_scores

            if verbose:
                print(f"\nFound {len(muscle_comps)} TMS-muscle components")

        # 2. Detect eye blink artifacts
        blink_comps, blink_scores = self._detect_blinks(
            weights, inst, blink_thresh)
        results['blink']['components'] = blink_comps
        results['blink']['scores'] = blink_scores

        if verbose:
            print(f"Found {len(blink_comps)} blink components")

        # 3. Detect lateral eye movement artifacts
        lat_eye_comps, lat_eye_scores = self._detect_lateral_eye(
            weights, inst, lat_eye_thresh)
        results['lateral_eye']['components'] = lat_eye_comps
        results['lateral_eye']['scores'] = lat_eye_scores

        if verbose:
            print(f"Found {len(lat_eye_comps)} lateral eye movement components")

        # 4. Detect persistent muscle artifacts
        muscle_comps, muscle_scores = self._detect_muscle_frequency(
            components, inst.info['sfreq'], muscle_freq_window, muscle_freq_thresh)
        results['muscle']['components'] = muscle_comps
        results['muscle']['scores'] = muscle_scores

        if verbose:
            print(f"Found {len(muscle_comps)} persistent muscle components")

        # 5. Detect electrode noise
        noise_comps, noise_scores = self._detect_electrode_noise(
            weights, noise_thresh)
        results['noise']['components'] = noise_comps
        results['noise']['scores'] = noise_scores

        if verbose:
            print(f"Found {len(noise_comps)} noisy electrode components")

        return results

    def _detect_tms_muscle(self, components, window=(11, 30), thresh=2.0):
        """
        Detect TMS-evoked muscle artifacts following Equation 3.
        Only works with epoched data.
        """
        if not hasattr(self, 'epochs'):
            return [], {'ratios': [], 'window_means': [], 'total_means': []}

        # Get time window indices
        sfreq = self.epochs.info['sfreq']
        window_samples = np.array([np.abs(self.epochs.times - w / 1000).argmin()
                                   for w in window])

        # Initialize outputs
        muscle_components = []
        scores = {'ratios': [], 'window_means': [], 'total_means': []}

        # Process each component
        for comp_idx in range(self.ica.n_components_):
            # Get component time course averaged across trials
            comp_data = np.mean(components.get_data()[:, comp_idx, :], axis=0)

            # Take absolute values
            comp_abs = np.abs(comp_data)

            # Calculate means following TESA formula
            window_length = window_samples[1] - window_samples[0]
            window_mean = (1 / window_length) * np.sum(
                comp_abs[window_samples[0]:window_samples[1]])
            total_mean = (1 / len(comp_abs)) * np.sum(comp_abs)

            # Calculate ratio
            muscle_ratio = window_mean / total_mean

            # Store scores
            scores['ratios'].append(muscle_ratio)
            scores['window_means'].append(window_mean)
            scores['total_means'].append(total_mean)

            # Classify component
            if muscle_ratio >= thresh:
                muscle_components.append(comp_idx)

        return muscle_components, scores

    def _detect_blinks(self, weights, inst, thresh=2.5):
        """
        Detect eye blink artifacts following Equation 4.
        Works with both Raw and Epochs data.

        Parameters
        ----------
        weights : array
            ICA weight matrix
        inst : Raw or Epochs
            MNE Raw or Epochs instance
        thresh : float
            Z-score threshold for blink detection
        """
        # Get electrode indices for Fp1 and Fp2
        fp_channels = ['Fp1', 'Fp2']
        fp_idx = [inst.ch_names.index(ch) for ch in fp_channels
                  if ch in inst.ch_names]

        if not fp_idx:
            print("Warning: Could not find Fp1/Fp2 channels for blink detection")
            return [], {'z_scores': []}

        # Initialize outputs
        blink_components = []
        scores = {'z_scores': []}

        # Calculate z-scores for weights
        w_mean = np.mean(weights, axis=0)
        w_std = np.std(weights, axis=0)

        for comp_idx in range(weights.shape[1]):
            # Get average z-score for Fp1/Fp2
            fp_z_scores = [(weights[fp, comp_idx] - w_mean[comp_idx]) / w_std[comp_idx]
                           for fp in fp_idx]
            mean_z = np.abs(np.mean(fp_z_scores))

            scores['z_scores'].append(mean_z)

            # Classify component
            if mean_z > thresh:
                blink_components.append(comp_idx)

        return blink_components, scores

    def _detect_lateral_eye(self, weights, inst, thresh=2.0):
        """
        Detect lateral eye movement artifacts following Equations 5 & 6.
        Works with both Raw and Epochs data.

        Parameters
        ----------
        weights : array
            ICA weight matrix
        inst : Raw or Epochs
            MNE Raw or Epochs instance
        thresh : float
            Z-score threshold for lateral eye movement detection
        """
        # Get electrode indices for F7 and F8
        lat_channels = ['F7', 'F8']
        lat_idx = [inst.ch_names.index(ch) for ch in lat_channels
                   if ch in inst.ch_names]

        if len(lat_idx) < 2:
            print("Warning: Could not find F7/F8 channels for lateral eye detection")
            return [], {'z_scores': []}

        # Initialize outputs
        lat_eye_components = []
        scores = {'z_scores': []}

        # Calculate z-scores for weights
        w_mean = np.mean(weights, axis=0)
        w_std = np.std(weights, axis=0)

        for comp_idx in range(weights.shape[1]):
            # Get z-scores for F7/F8
            z_scores = [(weights[ch, comp_idx] - w_mean[comp_idx]) / w_std[comp_idx]
                        for ch in lat_idx]

            scores['z_scores'].append(z_scores)

            # Check for opposite polarity exceeding threshold
            if ((z_scores[0] > thresh and z_scores[1] < -thresh) or
                    (z_scores[0] < -thresh and z_scores[1] > thresh)):
                lat_eye_components.append(comp_idx)

        return lat_eye_components, scores

    def _detect_muscle_frequency(self, components, sfreq, freq_window=(30, 100), thresh=0.6):
        """
        Detect persistent muscle artifacts following Equation 7.
        Works with both Raw and Epochs data.

        Parameters
        ----------
        components : array
            ICA component data
        sfreq : float
            Sampling frequency
        freq_window : tuple
            Frequency window (Hz) for muscle activity detection
        thresh : float
            Threshold for muscle component detection
        """
        from scipy.signal import welch

        # Initialize outputs
        muscle_components = []
        scores = {'power_ratios': []}

        # Get component data
        if isinstance(components, mne.BaseEpochs):
            comp_data = components.get_data()
        else:  # Raw data
            comp_data = components.get_data()
            # Reshape to match epochs format [n_epochs=1, n_components, n_times]
            comp_data = comp_data.reshape(1, *comp_data.shape)

        # Calculate frequency representation for each component
        for comp_idx in range(self.ica.n_components_):
            # Calculate power spectrum
            freqs, psd = welch(comp_data[:, comp_idx, :], fs=sfreq)

            # Get indices for frequency window
            freq_idx = np.where((freqs >= freq_window[0]) &
                                (freqs <= freq_window[1]))[0]

            # Calculate power ratio
            window_power = np.mean(psd[:, freq_idx])
            total_power = np.mean(psd)
            power_ratio = window_power / total_power

            scores['power_ratios'].append(power_ratio)

            # Classify component
            if power_ratio > thresh:
                muscle_components.append(comp_idx)

        return muscle_components, scores

    def _detect_electrode_noise(self, weights, thresh=4.0):
        """
        Detect electrode noise following Equation 8.
        Works with both Raw and Epochs data as it only uses ICA weights.

        Parameters
        ----------
        weights : array
            ICA weight matrix
        thresh : float
            Z-score threshold for noise detection
        """
        # Initialize outputs
        noise_components = []
        scores = {'max_z_scores': []}

        # Calculate z-scores for weights
        w_mean = np.mean(weights, axis=0)
        w_std = np.std(weights, axis=0)

        for comp_idx in range(weights.shape[1]):
            # Calculate z-scores for all electrodes
            z_scores = (weights[:, comp_idx] - w_mean[comp_idx]) / w_std[comp_idx]
            max_abs_z = np.max(np.abs(z_scores))

            scores['max_z_scores'].append(max_abs_z)

            # Classify component
            if max_abs_z > thresh:
                noise_components.append(comp_idx)

        return noise_components, scores


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
                            ylim: Optional[Dict] = None,
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

    def initial_downsample(self):
        """
        Perform initial downsampling of raw data to initial_sfreq (default 1000 Hz).
        """
        current_sfreq = self.raw.info['sfreq']
        if current_sfreq > self.initial_sfreq:
            self.raw = self.raw.resample(self.initial_sfreq)
            print(f"Initially downsampled raw data to {self.initial_sfreq} Hz")
        else:
            print(f"Current sfreq ({current_sfreq} Hz) <= initial target sfreq ({self.initial_sfreq} Hz); "
                  "no initial downsampling performed")

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
