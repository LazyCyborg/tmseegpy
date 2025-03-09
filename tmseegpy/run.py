# run.py
import os
from pathlib import Path
import sys


# Automatically set up Qt plugin path
def setup_qt_plugin_path():
    try:
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            possible_plugin_paths = [
                # macOS conda paths
                Path(conda_prefix) / "lib" / "python3.11" / "site-packages" / "PyQt6" / "Qt6" / "plugins" / "platforms",
                Path(conda_prefix) / "lib" / "python3.11" / "site-packages" / "PyQt6" / "Qt6" / "plugins",
                # Additional macOS-specific paths
                Path(conda_prefix) / "lib" / "python3.11" / "site-packages" / "PyQt6-Qt6" / "plugins" / "platforms",
                Path(conda_prefix) / "lib" / "python3.11" / "site-packages" / "PyQt6-Qt6" / "Qt6" / "plugins" / "platforms",
                # Windows path
                Path(conda_prefix) / "Library" / "plugins" / "platforms",
            ]

            for path in possible_plugin_paths:
                if path.exists():
                    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(path)
                    print(f"Set QT_QPA_PLATFORM_PLUGIN_PATH to: {path}")
                    return  # Exit after first valid path

            # Fallback to PyQt6 direct path
            import PyQt6
            qt_path = Path(PyQt6.__file__).parent / "Qt6" / "plugins" / "platforms"
            if qt_path.exists():
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(qt_path)
                print(f"Set QT_QPA_PLATFORM_PLUGIN_PATH to: {qt_path}")
    except Exception as e:
        print(f"Warning: Could not automatically set Qt plugin path: {e}")




def _is_in_jupyter():
    """Check if we're running in a Jupyter notebook"""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
        return False
    except ImportError:
        return False

# Only call setup_qt_plugin_path if we're not in a notebook
if not _is_in_jupyter():
    setup_qt_plugin_path()

import numpy as np
import matplotlib.pyplot as plt
from tmseegpy.preproc import TMSEEGPreprocessor  # absolute import
from tmseegpy.dataloader import TMSEEGLoader

from tmseegpy.pcist import PCIst
from tmseegpy.preproc_vis import save_raw_data, save_epochs_data
from tmseegpy.validate_tep import (
    analyze_gmfa,
    analyze_roi,
    plot_tep_analysis,
    generate_validation_summary,
    DEFAULT_TEP_COMPONENTS
)
import mne
import time
from .neurone_loader_fix import Recording
import argparse
import queue

mne.viz.use_browser_backend("matplotlib")
plt.rcParams['figure.figsize'] = [8, 6]




def generate_research_stats(pcist_values, pcist_objects, details_list, session_names, output_dir):
    """
    Generate detailed research statistics for PCIst measurements, including both individual
    session statistics and pooled analysis across all sessions.
    
    Args:
        pcist_values: List of PCIst values for all sessions
        pcist_objects: List of PCIst objects containing the analyses
        details_list: List of dictionaries containing PCIst calculation details for each session
        session_names: List of session names
        output_dir: Directory to save the output file
    """
    import numpy as np
    from pathlib import Path
    from scipy import stats
    import datetime
    
    output_file = Path(output_dir) / f"pcist_research_stats_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(output_file, 'w') as f:
        # Header
        f.write("PCIst Research Statistics Report\n")
        f.write("=" * 50 + "\n\n")
        
        # 1. Pooled Statistics
        f.write("1. POOLED STATISTICS ACROSS ALL SESSIONS\n")
        f.write("-" * 40 + "\n")
        
        # Basic descriptive statistics
        pcist_array = np.array(pcist_values)
        f.write(f"Number of sessions: {len(pcist_values)}\n")
        f.write(f"Mean PCIst: {np.mean(pcist_array):.4f}\n")
        f.write(f"Median PCIst: {np.median(pcist_array):.4f}\n")
        f.write(f"Standard deviation: {np.std(pcist_array):.4f}\n")
        f.write(f"Coefficient of variation: {(np.std(pcist_array)/np.mean(pcist_array))*100:.2f}%\n")
        f.write(f"Range: {np.min(pcist_array):.4f} - {np.max(pcist_array):.4f}\n")
        
        # Distribution statistics
        f.write("\nDistribution Statistics:\n")
        f.write(f"Skewness: {stats.skew(pcist_array):.4f}\n")
        f.write(f"Kurtosis: {stats.kurtosis(pcist_array):.4f}\n")
        shapiro_stat, shapiro_p = stats.shapiro(pcist_array)
        f.write(f"Shapiro-Wilk test (normality): W={shapiro_stat:.4f}, p={shapiro_p:.4f}\n")
        
        # Quartile statistics
        q1, q2, q3 = np.percentile(pcist_array, [25, 50, 75])
        iqr = q3 - q1
        f.write(f"\nQuartile Statistics:\n")
        f.write(f"Q1 (25th percentile): {q1:.4f}\n")
        f.write(f"Q2 (median): {q2:.4f}\n")
        f.write(f"Q3 (75th percentile): {q3:.4f}\n")
        f.write(f"IQR: {iqr:.4f}\n")
        
        # Outlier detection
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = pcist_array[(pcist_array < lower_bound) | (pcist_array > upper_bound)]
        outlier_sessions = [session_names[i] for i, val in enumerate(pcist_array) 
                          if val < lower_bound or val > upper_bound]
        
        f.write("\nOutlier Analysis:\n")
        f.write(f"Lower bound: {lower_bound:.4f}\n")
        f.write(f"Upper bound: {upper_bound:.4f}\n")
        f.write(f"Number of outliers: {len(outliers)}\n")
        if outliers.size > 0:
            f.write("Outlier sessions:\n")
            for session, value in zip(outlier_sessions, outliers):
                f.write(f"  {session}: {value:.4f}\n")
        
        # 2. Session-by-Session Analysis
        f.write("\n2. INDIVIDUAL SESSION STATISTICS\n")
        f.write("-" * 40 + "\n")
        
        for i, (pcist_obj, details, session_name) in enumerate(zip(pcist_objects, details_list, session_names)):
            f.write(f"\nSession: {session_name}\n")
            f.write(f"PCIst value: {pcist_values[i]:.4f}\n")
            
            # SVD and Component Analysis
            if 'signal_shape' in details:
                f.write(f"\nSignal Analysis:\n")
                f.write(f"Signal shape: {details['signal_shape']}\n")
            
            if 'singular_values' in details:
                sv = details['singular_values']
                f.write(f"Number of components: {len(sv)}\n")
                f.write(f"Non-zero components: {np.sum(sv > 1e-10)}\n")
                if 'variance_explained' in details:
                    f.write(f"Variance explained: {details['variance_explained']:.2f}%\n")
            
            # SNR Information
            if 'snr_values' in details:
                snr_values = details['snr_values']
                f.write("\nSNR Statistics:\n")
                f.write(f"Mean SNR: {np.mean(snr_values):.4f}\n")
                f.write(f"Max SNR: {np.max(snr_values):.4f}\n")
                if 'parameters' in details and 'min_snr' in details['parameters']:
                    f.write(f"Channels above SNR threshold: {np.sum(snr_values > details['parameters']['min_snr'])}\n")
            
            # NST Analysis
            if 'nst_base_range' in details and 'nst_resp_range' in details:
                f.write("\nNST Analysis:\n")
                f.write(f"NST baseline range: {details['nst_base_range'][0]:.3f} to {details['nst_base_range'][1]:.3f}\n")
                f.write(f"NST response range: {details['nst_resp_range'][0]:.3f} to {details['nst_resp_range'][1]:.3f}\n")
            
            if 'dnst_values' in details:
                f.write(f"Mean dNST: {np.mean(details['dnst_values']):.4f}\n")
                f.write(f"Max dNST: {np.max(details['dnst_values']):.4f}\n")
            
            # Parameters used
            if 'parameters' in details:
                f.write("\nAnalysis Parameters:\n")
                for param, value in details['parameters'].items():
                    f.write(f"{param}: {value}\n")
            
        # 3. Quality Control Summary
        f.write("\n3. QUALITY CONTROL SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        # Overall quality metrics
        mean_snr = np.mean([np.mean(d.get('snr_values', [0])) for d in details_list])
        mean_var_explained = np.mean([d.get('variance_explained', 0) for d in details_list])
        
        f.write(f"Mean SNR across sessions: {mean_snr:.4f}\n")
        f.write(f"Mean variance explained: {mean_var_explained:.2f}%\n")
        
        # Quality assessment
        quality_threshold = 0.5
        quality_metrics = [
            mean_snr / 10,
            mean_var_explained / 100,
            1 - (len(outliers) / len(pcist_values))
        ]
        overall_quality = np.mean(quality_metrics)
        
        f.write(f"\nOverall Quality Assessment: {overall_quality:.4f}\n")
        if overall_quality < quality_threshold:
            f.write("WARNING: Quality metrics below threshold - review data carefully\n")
        
    return output_file

### Do not touch this unless you know what you are doing ###

def process_subjects(args, status_callback=None):

    import builtins
    setattr(builtins, 'STOP_PROCESSING', False)
    def check_stop():
        if getattr(builtins, 'STOP_PROCESSING', False):
            print("\nProcessing stopped by user")
            return True
        return False


    data_dir = Path(args.data_dir)

    # Check if the path already ends with TMSEEG
    if data_dir.name == 'TMSEEG':
        TMS_DATA_PATH = data_dir
    else:
        TMS_DATA_PATH = data_dir / 'TMSEEG'

    # Verify paths exist
    required_paths = {
        'Data Directory': data_dir,
        'TMS Data': TMS_DATA_PATH,
    }
    for name, path in required_paths.items():
        if not path.exists():
            print(f"WARNING: {name} not found at: {path}")
        else:
            print(f"✓ Found {name} at: {path}")
    
    # Store PCI values
    subject_pcist_values = []
    pcist_objects = []
    pcist_details = []
    session_names = []

    # Load the raw data using the new loader
    from .dataloader import TMSEEGLoader
    loader = TMSEEGLoader(
        data_path=TMS_DATA_PATH,
        format=args.data_format,
        substitute_zero_events_with=10,
        verbose=True
    )

    raw_list = loader.load_data()
    session_info = loader.get_session_info()

    np.random.seed(42)

    # Loop through the loaded raw data
    for n, raw in enumerate(raw_list):
        if check_stop():
            return []

        raw_eve = raw.copy()
        session_name = session_info[n]['name']

        if args.save_raw_data:
            save_raw_data(raw, args.output_dir, step_name='raw_0', session_name=session_name)

        print(f"\nProcessing Session {n}: {session_name}")

        if check_stop(): return []

        # Process session...
        events = None

        try:
            # First try user-specified channel
            if args.stim_channel and args.stim_channel in raw.ch_names:
                print(f"Using specified stim channel: {args.stim_channel}")
                original_events = mne.find_events(raw_eve, stim_channel=args.stim_channel)
                original_event_id = {str(ev_id): ev_id for ev_id in np.unique(original_events[:, 2])}
                print(f"Found {len(original_events)} original events")
                print(f"Original event IDs: {original_event_id}")
                events = original_events
            else:
                # Look for events in annotations
                print("Looking for events in annotations...")
                if len(raw.annotations) > 1:
                    # Get unique annotation descriptions
                    try:
                        print("Trying fallback event extraction with Stimulus/A annotation...")
                        events, event_dict = mne.events_from_annotations(raw, event_id={'Stimulus/A': 1})
                        if events is not None and len(events) > 0:
                            print(f"Found {len(events)} events from Stimulus/A annotations")
                    except Exception as e:
                        print(f"Error with Stimulus/A fallback: {str(e)}")

                else:
                    print("No TMS-related annotations found, checking stim channels...")
                    # Try to find stim channels
                    stim_channels = mne.pick_types(raw.info, stim=True, exclude=[])
                    if len(stim_channels) > 0:
                        stim_ch_name = raw.ch_names[stim_channels[0]]
                        print(f"Using detected stim channel: {stim_ch_name}")
                        events = mne.find_events(raw, stim_channel=stim_ch_name)
                    else:
                        # Try common stim channel names
                        common_stim_names = ['STI 014', 'STIM', 'STI101', 'trigger', 'STI 001']
                        for ch_name in common_stim_names:
                            if ch_name in raw.ch_names:
                                print(f"Using stim channel: {ch_name}")
                                events = mne.find_events(raw, stim_channel=ch_name)
                                break

            if events is not None and len(events) > 0:
                print(f"Found {len(events)} trigger-based events")


        except Exception as e:
            print(f"Error during trigger-based event detection: {str(e)}")
            events = None

        # Drop unnecessary channels
        channels_to_drop = []
        if 'EMG1' in raw.ch_names:
            channels_to_drop.append('EMG1')
        if channels_to_drop:
            print(f"Dropping channels: {channels_to_drop}")
            raw.drop_channels(channels_to_drop)
        if args.save_raw_data:
            save_raw_data(raw, args.output_dir, step_name='raw_1', session_name=session_name)

        raw_data = raw.get_data()
        print(f"Initial raw data range: [{np.min(raw_data)}, {np.max(raw_data)}]")

            # Initialize processor here since we need it for artifact detection
        montage = mne.channels.make_standard_montage(args.montage_name)
        raw.set_montage(montage)

        print(f"Set montage: {args.montage_name}")

        from .preproc import detect_tms_artifacts

        if args.auto_detect_artifacts:
            print("\nLooking for additional artifacts...")
            try:
                additional_events = detect_tms_artifacts(raw,
                    threshold_std=args.artifact_threshold_std,
                    min_distance_ms=args.min_artifact_distance_ms,
                    existing_events=events
                )

                if additional_events is not None:
                    if events is None:
                        events = additional_events
                    else:
                        events = np.vstack((events, additional_events))
                        # Sort by time and remove duplicates
                        events = events[events[:, 0].argsort()]
                        _, unique_idx = np.unique(events[:, 0], return_index=True)
                        events = events[unique_idx]
                    print(f"Total events after combining: {len(events)}")
            except Exception as e:
                print(f"Error during automatic artifact detection: {str(e)}")
                import traceback
                traceback.print_exc()

            if events is not None and len(events) > 0:
                annot_array = mne.Annotations(
                    onset=events[:, 0] / raw.info['sfreq'],
                    duration=np.zeros(len(events)),
                    description=['Stimulation'] * len(events)
                )
                raw.set_annotations(annot_array)  # This will overwrite any existing annotations
                print(f"Created annotations for {len(events)} events")
                print(f"First few event times: {events[:5, 0] / raw.info['sfreq']}")
                print(f"First few annotation onsets: {raw.annotations.onset[:5]}")

        processor = TMSEEGPreprocessor(raw)
        if check_stop(): return []
        processor.fix_tms_artifact(events=processor.events)  # Step 8

       # print("\nInterpolating TMS artifact...")
       # processor.interpolate_tms_artifact(method='cubic',
                                       # interp_window=args.initial_interp_window,  # 1ms window for initial interpolation
                                       # cut_times_tms=(args.initial_window_start, args.initial_window_end))  # Step 9

        if args.save_preproc:
            save_raw_data(raw, args.output_dir, step_name='raw_2', session_name=session_name)

        if args.filter_raw:
            print(f"\nFiltering raw eeg data with lowpass {args.raw_h_freq} Hz... and highpass {args.raw_l_freq}")
            if check_stop(): return []
            processor.filter_raw(l_freq=args.raw_l_freq, h_freq=args.raw_h_freq)


        #if args.save_preproc:
          #  save_raw_data(raw, args.output_dir, step_name='raw_f',)

        print("\nCreating epochs...")
        processor.create_epochs(tmin=args.epochs_tmin, tmax=args.epochs_tmax, baseline=None,
                                events=events)
        epochs = processor.epochs

        epochs_data = epochs.get_data()
        print(f"Data range after epoching: [{np.min(epochs_data)}, {np.max(epochs_data)}]")

        if args.save_preproc:
            save_epochs_data(processor.epochs, args.output_dir, session_name=session_name, step_name='0_epochs')

        print("\nRemoving bad channels...")

        processor.remove_bad_channels(threshold=args.bad_channels_threshold)

        if args.save_preproc:
            save_epochs_data(processor.epochs, args.output_dir, session_name=session_name, step_name='1_epochs_no_bad_channels')

        print("\nRemoving bad epochs...")

        processor.remove_bad_epochs(threshold=args.bad_epochs_threshold)
        if args.save_preproc:
            save_epochs_data(processor.epochs, args.output_dir, session_name=session_name, step_name='2_epochs_bad_no_epoch')

        print("\nSetting average reference...")
        processor.set_average_reference()

        if not args.no_first_ica:
            print("\nRunning first ICA...")
            if check_stop(): return []

            if not args.no_select_with_nn:
                select_with_nn = True
            else:
                select_with_nn = False

            # Run ICA with automatic classifications only
            processor.run_ica(
                output_dir=args.output_dir,
                session_name=session_name,
                method=args.ica_method,
                select_with_nn=select_with_nn,
                select_with_topo=args.select_with_topo,
                use_nn=True,  # Always run neural network classification
                use_topo=True,  # Always run topography classification
                topo_edge_threshold=args.topo_edge_threshold,
                topo_zscore_threshold=args.topo_zscore_threshold,
                topo_peak_threshold=args.topo_peak_threshold,
                topo_focal_threshold=args.topo_focal_threshold
            )

            if args.save_preproc:
                save_epochs_data(processor.epochs, args.output_dir, session_name=session_name, step_name='2a_ica1')

        if args.parafac_muscle_artifacts:
            print("\nCleaning muscle artifacts with PARAFAC decomposition...")
            if check_stop(): return []
            processor.clean_muscle_artifacts(
                muscle_window=(args.muscle_window_start, args.muscle_window_end),
                threshold_factor=args.threshold_factor,
                n_components=args.n_components,
                verbose=True
            )



        if check_stop(): return []
        if args.mne_filter_epochs:
            print(f"\nFiltering epoched data with highpass {args.l_freq} Hz... and lowpass {args.h_freq} Hz... ...")
            processor.mne_filter_epochs(
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                notch_freq=args.notch_freq,
                notch_width=args.notch_width,
            )
        elif args.scipy_filter_epochs:
            print(f"\nFiltering epoched data with highpass {args.l_freq} Hz... and lowpass {args.h_freq} Hz... ...")
            processor.scipy_filter_epochs(
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                notch_freq=args.notch_freq,
                notch_width=args.notch_width)

        else:
            print(f"\nFiltering epoched data with highpass {args.l_freq} Hz... and lowpass {args.h_freq} Hz... ...")
            processor.scipy_filter_epochs(
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                notch_freq=args.notch_freq,
                notch_width=args.notch_width,
            )

            if args.save_preproc:
                save_epochs_data(processor.epochs, args.output_dir, session_name=session_name, step_name='4_filtered_epochs')

        if not args.no_second_ica:
            print("\nRunning second ICA...")
            if check_stop(): return []

            processor.run_second_ica(
                method=args.second_ica_method,
                select_with_iclabel=True,
                use_icalabel=True,  # Always run ICA label classification
                icalabel_exclude_labels=args.icalabel_exclude_labels
            )

            if args.save_preproc:
                save_epochs_data(processor.epochs, args.output_dir, session_name=session_name, step_name='4b_ica2')

        if args.apply_ssp:    
            print("\nApplying SSP...")
            processor.apply_ssp(n_eeg=args.ssp_n_eeg)

        #print("\nApplying baseline correction...")
        #processor.apply_baseline_correction(baseline=(baseline_start_sec, baseline_end_sec))
        if args.save_preproc:
            save_epochs_data(processor.epochs, args.output_dir, session_name=session_name, step_name='4b_ica2')
        if args.apply_csd:
            print("\nApplying CSD transformation...")
            processor.apply_csd(lambda2=args.lambda2, stiffness=args.stiffness)

        print("\nPerforming final downsampling...")
        processor.final_downsample()

        if args.save_preproc:
            save_epochs_data(processor.epochs, args.output_dir, session_name, step_name='5_final')

        if not args.no_preproc_output:

            epochs = processor.epochs
            epochs.save(Path(args.output_dir) / f"{session_name}_preproc-epo.fif", verbose=True, overwrite=True)

        if args.analyze_teps:
            try:
                print("\nAnalyzing TEPs...")
                if check_stop(): return []

                # Define our standard TEP components exactly as in TESA
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

                # Process peak windows if provided
                if args.peak_windows:
                    print("Using custom peak windows...")
                    windows = []
                    for window in args.peak_windows:
                        try:
                            start, end = map(float, window.split(','))
                            windows.append((start, end))
                            print(f"Added window: {start}-{end}ms")
                        except ValueError:
                            print(f"Warning: Could not parse window {window}, skipping...")

                    # If override requested, modify component windows
                    if args.manual_windows and windows:
                        print("Overriding default component windows...")
                        for (start, end), name in zip(windows, DEFAULT_TEP_COMPONENTS.keys()):
                            DEFAULT_TEP_COMPONENTS[name]['time'] = (start, end)
                            print(f"Modified window for {name}: {start}-{end}ms")

                results = {}

                # Analyze GMFA if requested
                if args.tep_analysis_type in ['gmfa', 'both']:
                    print("Analyzing GMFA...")
                    gmfa_results = analyze_gmfa(
                        epochs=epochs,
                        components=DEFAULT_TEP_COMPONENTS,
                        samples=args.tep_samples,
                        method=args.tep_method
                    )
                    results['gmfa'] = gmfa_results

                # Analyze ROI if requested
                if args.tep_analysis_type in ['roi', 'both']:
                    # Verify specified channels exist
                    available_channels = [ch for ch in args.tep_roi_channels
                                          if ch in epochs.ch_names]

                    if available_channels:
                        print(f"Analyzing ROI: {available_channels}")
                        roi_results = analyze_roi(
                            epochs=epochs,
                            channels=available_channels,
                            components=DEFAULT_TEP_COMPONENTS,
                            samples=args.tep_samples,
                            method=args.tep_method
                        )
                        results['roi'] = roi_results
                    else:
                        print(f"Warning: None of the specified ROI channels {args.tep_roi_channels} "
                              "found in data")

                if args.no_channel_peaks:
                    channel_peaks = False
                else:
                    channel_peaks = True

                # Create visualization
                print("Generating TEP plots...")
                plot_tep_analysis(
                    epochs=epochs,
                    output_dir=args.output_dir,
                    session_name=session_name,
                    components=DEFAULT_TEP_COMPONENTS,
                    analysis_type=args.tep_analysis_type,
                    channels=available_channels if args.tep_analysis_type in ['roi', 'both'] else None,
                    n_samples=args.tep_samples,
                    method=args.tep_method,
                    peak_mode=args.peak_mode,
                    show_channel_peaks=channel_peaks,
                    peak_windows=args.peak_windows,
                    override_windows=args.manual_windows
                )

                # Print channel-level peak information if requested
                if not args.no_channel_peaks:
                    print("\nChannel-level peak analysis:")
                    evoked = epochs.average()
                    for name, comp in DEFAULT_TEP_COMPONENTS.items():
                        tmin, tmax = comp['time']
                        try:
                            # Get the polarity from component definition if no peak_mode specified
                            if args.peak_mode is None:
                                polarity = comp.get('polarity', 'positive')
                                detection_mode = 'neg' if polarity == 'negative' else 'pos'
                            else:
                                detection_mode = args.peak_mode

                            ch_name, lat, amp = evoked.get_peak(
                                tmin=tmin / 1000,  # Convert to seconds
                                tmax=tmax / 1000,  # Convert to seconds
                                mode=detection_mode,
                                return_amplitude=True
                            )
                            print(f"\n{name} window ({tmin}-{tmax}ms):")
                            print(f"  Channel: {ch_name}")
                            print(f"  Latency: {lat * 1000:.1f}ms")
                            print(f"  Amplitude: {amp:.2f}µV")
                        except Exception as e:
                            print(f"\nWarning: Could not find peak for {name}: {str(e)}")

                # Generate validation summary if requested
                if args.save_validation:
                    print("Generating validation summary...")
                    # Use appropriate results based on analysis type
                    validation_results = (results.get('gmfa', None) or
                                          results.get('roi', None))

                    if validation_results:
                        generate_validation_summary(
                            components=validation_results,
                            output_dir=args.output_dir,
                            session_name=session_name
                        )
                        print("TEP validation summary saved")
                    else:
                        print("Warning: No results available for validation summary")

            except Exception as e:
                print(f"Warning: TEP analysis failed: {str(e)}")
                import traceback
                print("Detailed error:")
                print(traceback.format_exc())

        
        recording_id = f"session_{n}"
        if not args.no_pcist:

            # PCIst analysis
            pcist = PCIst(epochs)
            par = {
                'baseline_window': (args.baseline_start, args.baseline_end),
                'response_window': (args.response_start, args.response_end),
                'k': args.k,
                'min_snr': args.min_snr,
                'max_var': args.max_var,
                'embed': args.embed,
                'n_steps': args.n_steps
            }
            value, details = pcist.calc_PCIst(**par, return_details=True)
            fig = pcist.plot_analysis(details, session_name=session_name)
            print(f"PCI: {value}")
            subject_pcist_values.append(value)
            fig.savefig(f"{args.output_dir}/pcist_{session_name}.png")
            plt.close(fig)

            pcist_objects.append(pcist)
            pcist_details.append(details)
            session_names.append(session_name)



    if args.research:
        output_file = generate_research_stats(
            subject_pcist_values,
            pcist_objects,
            pcist_details,
            session_names,
            args.output_dir
        )
        print(f"Research statistics saved to: {output_file}")

    print("Processing complete! (wow it works)")
    return subject_pcist_values


### Do not touch this unless you know what you are doing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process EEG data.')
    # Add this new argument

    parser.add_argument('--data_dir', type=str, default=str(Path.cwd() / 'data'), 
                        help='Path to the data directory (default: ./data)')

    parser.add_argument('--output_dir', type=str, default=str(Path.cwd() / 'output'),
                        help='Path to the output directory (default: ./output)')

    parser.add_argument('--data_format', type=str, default='neurone',
                   choices=['neurone', 'brainvision', 'edf', 'cnt', 'eeglab', 'auto'],
                   help='Format of input data (default: neurone)')

    parser.add_argument('--no_preproc_output', action='store_true', default=False,
                    help='Skip saving preprocessed epochs (default: False)')

    parser.add_argument('--montage_name', type=str, default='easycap-M1',
                    help='Name of the montage from MNE Pythons builtin montages. Use mne.channels.get_builtin_montages() to display all (default: easycap-M1)')

    parser.add_argument('--no_pcist', action='store_true', default=False,
                    help='Skip PCIst calculation and only preprocess (default: False)')

    parser.add_argument('--stim_channel', type=str, default='STI 014',
                    help='Name of the stimulus channel (default: STI 014)')

    parser.add_argument('--save_preproc', action='store_true', default=False,
                    help='Save plots between preprocessing steps (default: False)')

    parser.add_argument('--final_sfreq', type=float, default=725,
                        help='Final downsampling frequency (default: 725)')

    parser.add_argument('--auto_detect_artifacts', action='store_true', default=False,
                        help='Use automatic artifact detection instead of triggers (default: False)')

    parser.add_argument('--artifact_threshold_std', type=float, default=10,
                        help='Standard deviations above mean for artifact detection (default: 10)')

    parser.add_argument('--min_artifact_distance_ms', type=float, default=50,
                        help='Minimum distance between artifacts in ms (default: 50)')
    # Trying to match TESA
    parser.add_argument('--initial_window_start', type=float, default=-2,

                    help='Initial TMS artifact window start (TESA default: -2)')

    parser.add_argument('--second_artifact_removal', action='store_true', default=False,
                    help='Skip the second stage of TMS artifact removal (default: False)')

    parser.add_argument('--mne_filter_epochs', action='store_true', default=False,
                    help='Use built in filter in mne (default: False)')

    parser.add_argument('--scipy_filter_epochs', action='store_true', default=False,
                    help='Use custom filter from scipy (default: False)')

    parser.add_argument('--plot_raw', action='store_true',
                    help='Plot raw data (takes time) (default: False)')

    parser.add_argument('--filter_raw', action='store_true', default=True,
                        help='Whether to filter raw data instead of epoched (default: True)')

    parser.add_argument('--l_freq', type=float, default=None,
                        help='Lower frequency for filtering (default: None)')

    parser.add_argument('--raw_l_freq', type=float, default=1,
                        help='Lower frequency for filtering (default: 1)')

    parser.add_argument('--h_freq', type=float, default=45,
                        help='Upper frequency for filtering (default: 45)')

    parser.add_argument('--raw_h_freq', type=float, default=250,
                        help='Upper frequency for filtering the raw eeg data (default: 250)')

    parser.add_argument('--notch_freq', type=float, default=None,
                        help='Notch filter frequency (default: None)')

    parser.add_argument('--notch_width', type=float, default=None,
                        help='Notch filter width (default: None)')

    parser.add_argument('--epochs_tmin', type=float, default=-0.8,
                        help='Start time for epochs (default: -0.8)')

    parser.add_argument('--epochs_tmax', type=float, default=0.8,
                        help='End time for epochs (default: 0.8)')

    parser.add_argument('--bad_channels_threshold', type=float, default=3,
                        help='Threshold (std) for removing bad channels with mne_faster (default: 3)')

    parser.add_argument('--bad_epochs_threshold', type=float, default=3,
                        help='Threshold (std) for removing bad epochs with mne_faster (default: 3)')

    parser.add_argument('--ica_method', type=str, default='fastica',
                        help='ICA method (default: fastica)')

    parser.add_argument('--parafac_muscle_artifacts', action='store_true', default=False,
                    help='Enable muscle artifact cleaning (default: False)')

    parser.add_argument('--muscle_window_start', type=float, default=0.005,
                    help='Start time for muscle artifact window (default: 0.005)')

    parser.add_argument('--muscle_window_end', type=float, default=0.030,
                    help='End time for muscle artifact window (default: 0.030)')

    parser.add_argument('--threshold_factor', type=float, default=1.0,
                    help='Threshold factor for muscle artifact cleaning (default: 1.0)')

    parser.add_argument('--use_icalabel', action='store_true', default=False,
                        help='Use ICA label for second ICA component classification (default: False)')

    parser.add_argument('--icalabel_exclude_labels', type=str, nargs='+',
                        default=["eye", "heart", "muscle", "line_noise", "channel_noise", "unknown"],
                        help='Labels to exclude in ICA label classification (default: all except brain and other)')

    parser.add_argument('--n_components', type=int, default=5,
                    help='Number of components for PARAFAC muscle artifact cleaning (default: 5)')

    parser.add_argument('--no_first_ICA', action='store_true', default=False,
                    help='Disable first ICA (default: False)')

    parser.add_argument('--no_second_ICA', action='store_true', default=False,
                    help='Disable seconds ICA ´ (default: False)')

    parser.add_argument('--second_ica_method', type=str, default='infomax',
                        help='Second ICA method that can be infomax or fastica (default: infp,ax)')

    parser.add_argument('--ica_topo', action='store_true', default=False,
                        help='Use topography-based automatic ICA component classification (default: False)')

    parser.add_argument('--topo_edge_threshold', type=float, default=0.15,
                                help='Distance threshold for edge detection in topography classifier (default: 0.15)')

    parser.add_argument('--topo_zscore_threshold', type=float, default=3.5,
                                help='Z-score threshold for focal point detection (default: 3.5)')

    parser.add_argument('--topo_peak_threshold', type=float, default=3,
                                help='Peak count threshold for artifact detection (default: 3)')

    parser.add_argument('--topo_focal_threshold', type=float, default=0.2,
                                help='Threshold for focal area detection (default: 0.2)')

    parser.add_argument('--no_select_with_nn', action='store_true', default=False,
                                help='Disable automatic neural network component selection (enabled by default)')

    parser.add_argument('--select_with_topo', action='store_true', default=False,
                        help='Automatically select components using topography-based classification (default: False)')

    parser.add_argument('--nn_model_path', type=str, default=None,
                        help='Path to neural network model file (default: use built-in)')

    parser.add_argument('--nn_encoder_path', type=str, default=None,
                        help='Path to label encoder file (default: use built-in)')

    parser.add_argument('--nn_probability_threshold', type=float, default=0.05,
                        help='Probability threshold for component exclusion (default: 0.05)')

    parser.add_argument('--apply_ssp', action='store_true',
                    help='Apply SSP (default: False)')

    parser.add_argument('--ssp_n_eeg', type=int, default=2,
                        help='Number of EEG components for SSP (default: 2)')

    parser.add_argument('--apply_csd', action='store_true',
                    help='Apply CSD transformation (default: True)')

    parser.add_argument('--lambda2', type=float, default=1e-3,
                    help='Lambda2 parameter for CSD transformation (default: 1e-5)')

    parser.add_argument('--stiffness', type=int, default=4,
                    help='Stiffness parameter for CSD transformation (default: 4)')

    parser.add_argument('--save_evoked', action='store_true',
                    help='Save evoked plot with TEPs (default: False)')

    parser.add_argument('--save_raw_data', action='store_true',
                    help='Save initial raw eeg as .fif (default: False)')

    parser.add_argument('--analyze_teps', action='store_true', default=True,
                help='Find TEPs that normally exist (default: True)')

    parser.add_argument('--peak_mode', type=str, default=None,
                        choices=['pos', 'neg', 'abs'],
                        help='Mode for MNE peak detection (default: abs)')

    parser.add_argument('--peak_windows', type=str, nargs='*',
                        help='Time windows for peak detection in format start,end (in ms). Example: --peak_windows 80,140 150,250')

    parser.add_argument('--no_channel_peaks', action='store_true', default=False,
                        help='Disable plotting of individual channel peaks using MNE get_peaks (default: False)')

    # Allow overriding the default component windows
    parser.add_argument('--manual_windows', action='store_true',
                        help='Use manual peak_windows to override default component windows (default: False)')

    parser.add_argument('--save_validation', action='store_true',
                help='Save TEP validation summary (default: False)')

    parser.add_argument('--tep_analysis_type', type=str, default='gmfa',
                        choices=['gmfa', 'roi', 'both'],
                        help='Type of TEP analysis to perform (default: gmfa)')

    parser.add_argument('--tep_roi_channels', type=str, nargs='+',
                        default=['C3', 'C4'],
                        help='Channels to use for ROI analysis (default: C3 C4)')

    parser.add_argument('--tep_method', type=str, default='largest',
                        choices=['largest', 'centre'],
                        help='Method for peak detection (default: largest)')

    parser.add_argument('--tep_samples', type=int, default=5,
                        help='Number of samples for peak detection (default: 5)')

    parser.add_argument('--baseline_start', type=int, default=-400,
                        help='Start time for baseline in ms (default: -400)')

    parser.add_argument('--baseline_end', type=int, default=-50,
                        help='End time for baseline in ms (default: -50)')

    parser.add_argument('--response_start', type=int, default=0,
                        help='Start of response window in ms (default: 0)')

    parser.add_argument('--response_end', type=int, default=299,
                        help='End of response window in ms (default: 299)')

    parser.add_argument('--k', type=float, default=1.2,
                        help='PCIst parameter k (default: 1.2)')

    parser.add_argument('--min_snr', type=float, default=1.1,
                        help='PCIst parameter min_snr (default: 1.1)')

    parser.add_argument('--max_var', type=float, default=99.0,

                        help='PCIst parameter max_var (default: 99.0)')
    parser.add_argument('--embed', action='store_true',
                        help='PCIst parameter embed (default: False)')

    parser.add_argument('--n_steps', type=int, default=100,
                        help='PCIst parameter n_steps (default: 100)')

    parser.add_argument('--pre_window_start', type=int, default=-400,
                        help='Start of the pre-TMS window in ms (default: -400)')

    parser.add_argument('--pre_window_end', type=int, default=-50,
                        help='End of the pre-TMS window in ms (default: -50)')

    parser.add_argument('--post_window_start', type=int, default=0,
                        help='Start of the post-TMS window in ms (default: 0)')

    parser.add_argument('--post_window_end', type=int, default=300,
                        help='End of the post-TMS window in ms (default: 300)')

    parser.add_argument('--research', action='store_true',
                    help='Output summary statistics of measurements (default: False)')

    args = parser.parse_args()


    pcists, subject_pcist_values, pcist_objects, pcist_details, session_names = process_subjects(args)
    print(f"PCIst values: {pcists}")
