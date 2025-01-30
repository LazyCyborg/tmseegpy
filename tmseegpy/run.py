# run.py
import os
from pathlib import Path
import sys


# Automatically set up Qt plugin path
def setup_qt_plugin_path():
    try:
        # Get conda environment path
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            # Look for PyQt6 plugins in the conda environment
            possible_plugin_paths = [
                Path(conda_prefix) / "lib" / "python3.9" / "site-packages" / "PyQt6" / "Qt6" / "plugins" / "platforms",
                Path(conda_prefix) / "lib" / "python3.9" / "site-packages" / "PyQt6" / "Qt" / "plugins" / "platforms",
                Path(conda_prefix) / "Library" / "plugins" / "platforms",  # Windows path
            ]

            # Find the first valid path
            for path in possible_plugin_paths:
                if path.exists():
                    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(path)
                    print(f"Set QT_QPA_PLATFORM_PLUGIN_PATH to: {path}")
                    break
        else:
            # If not in conda environment, try to find PyQt6 in system Python
            import PyQt6
            qt_path = Path(PyQt6.__file__).parent / "Qt6" / "plugins" / "platforms"
            if qt_path.exists():
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(qt_path)
                print(f"Set QT_QPA_PLATFORM_PLUGIN_PATH to: {qt_path}")
    except Exception as e:
        print(f"Warning: Could not automatically set Qt plugin path: {e}")


# Call this before any Qt imports
setup_qt_plugin_path()
from tabnanny import verbose

import numpy as np
import matplotlib.pyplot as plt
from tmseegpy.preproc import TMSEEGPreprocessor  # absolute import
from tmseegpy.dataloader import TMSEEGLoader
from tmseegpy.ica_selector_gui.websocket_ica_selector import WebSocketICASelector

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
from .neurone_loader import Recording
import argparse
import queue

mne.viz.use_browser_backend("matplotlib")
plt.rcParams['figure.figsize'] = [8, 6]


def get_ica_callback(gui_mode=False):
    """Get appropriate ICA callback based on mode"""
    if gui_mode:
        from .ica_selector_gui.websocket_ica_selector import WebSocketICASelector
        def callback(ica_instance, inst, component_scores=None):
            try:
                print("Starting ICA component selection...")

                # Create queues for ICA component selection
                from queue import Queue
                selection_queue = Queue()
                result_queue = Queue()

                # Create selector instance
                selector = WebSocketICASelector(selection_queue, result_queue)

                # Format and send component data
                component_data = selector._get_component_data(ica_instance, inst)
                selection_queue.put(component_data)

                # Emit event to notify frontend
                from .server.server import socketio
                print("Emitting ica_required event...")
                socketio.emit('ica_required', {
                    'componentCount': len(component_data)
                })

                # Wait for result
                print("Waiting for component selection...")
                selected_components = selector.select_components(ica_instance, inst)
                print(f"Received selected components: {selected_components}")

                return selected_components

            except Exception as e:
                print(f"Error in ICA GUI callback: {str(e)}")
                import traceback
                traceback.print_exc()
                return []

        return callback
    else:
        # Use existing CLI callback for command line usage
        from .cli_ica_selector import get_cli_ica_callback
        return get_cli_ica_callback(is_gui_mode=False)

def generate_preproc_stats(processor, session_name, output_dir):
    """
    Generate simplified preprocessing quality control report.
    
    Args:
        processor: TMSEEGPreprocessor object
        session_name: Name of the current session
        output_dir: Directory to save the output file
    """
    from pathlib import Path
    import datetime
    import numpy as np
    
    output_file = Path(output_dir) / f"preproc_stats_{session_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(output_file, 'w') as f:
        # Header with key session info
        f.write(f"TMS-EEG Preprocessing Report: {session_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Recording parameters
        f.write("Recording Parameters\n")
        f.write("-" * 20 + "\n")
        f.write(f"Duration: {processor.raw.times[-1]:.1f} seconds\n")
        f.write(f"Sampling rate: {processor.raw.info['sfreq']} → {processor.final_sfreq} Hz\n")
        f.write(f"Channels: {len(processor.raw.ch_names)}\n\n")
        
        '''# TMS pulse information
        events = processor._get_events()
        n_events = len(events)
        f.write("TMS Pulses\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total pulses: {n_events}\n")
        if n_events > 1:
            intervals = np.diff([event[0] for event in events]) / processor.raw.info['sfreq']
            f.write(f"Mean interval: {np.mean(intervals):.3f} s (±{np.std(intervals):.3f})\n\n")'''
        
        # Data quality summary
        f.write("Data Quality Metrics\n")
        f.write("-" * 20 + "\n")
        
        # Channel quality
        n_bad_channels = len(getattr(processor, 'bad_channels', []))
        channel_retention = (len(processor.raw.ch_names) - n_bad_channels) / len(processor.raw.ch_names) * 100
        f.write(f"Channel retention: {channel_retention:.1f}%")
        if hasattr(processor, 'bad_channels') and processor.bad_channels:
            f.write(f" (Removed: {', '.join(processor.bad_channels)})")
        f.write("\n")
        
        # Epoch quality
        if hasattr(processor, 'epochs'):
            n_bad_epochs = len(getattr(processor, 'bad_epochs', []))
            epoch_retention = (len(processor.epochs) - n_bad_epochs) / len(processor.epochs) * 100
            f.write(f"Epoch retention: {epoch_retention:.1f}%")
            if n_bad_epochs > 0:
                f.write(f" ({n_bad_epochs} epochs removed)")
            f.write("\n")
        
        # Artifact removal summary
        f.write("\nArtifact Removal\n")
        f.write("-" * 20 + "\n")
        
        # ICA components
        if hasattr(processor, 'muscle_components'):
            f.write(f"Muscle components removed: {len(processor.muscle_components)}\n")
        if hasattr(processor, 'excluded_components'):
            f.write(f"Other artifacts removed: {len(processor.excluded_components)}\n")
        
        # Overall quality assessment
        f.write("\nQuality Assessment\n")
        f.write("-" * 20 + "\n")
        
        # Calculate simplified quality score
        quality_score = np.mean([
            channel_retention / 100,
            epoch_retention / 100 if hasattr(processor, 'epochs') else 1.0
        ])
        
        f.write(f"Overall quality score: {quality_score*100:.1f}%\n")
        
        # Add focused warnings if needed
        if quality_score < 0.7:
            f.write("\nWarnings:\n")
            if channel_retention < 80:
                f.write("• High number of channels removed\n")
            if hasattr(processor, 'epochs') and epoch_retention < 80:
                f.write("• High number of epochs removed\n")
    
    return output_file

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


def process_subjects(args, status_callback=None):
    import builtins
    setattr(builtins, 'STOP_PROCESSING', False)
    def check_stop():
        if getattr(builtins, 'STOP_PROCESSING', False):
            print("\nProcessing stopped by user")
            return True
        return False

    def combined_callback(msg, progress=None):
        print(msg)  # This will go through the output capturer
        if status_callback:
            status_callback(msg, progress)

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
        substitute_zero_events_with=args.substitute_zero_events_with,
        eeglab_montage_units=args.eeglab_montage_units,
        verbose=True
    )

    raw_list = loader.load_data()
    session_info = loader.get_session_info()

    np.random.seed(args.random_seed)
    baseline_start_sec = args.baseline_start / 1000.0
    baseline_end_sec = args.baseline_end / 1000.0

    # Loop through the loaded raw data
    for n, raw in enumerate(raw_list):
        if check_stop():
            return []

        
        session_name = session_info[n]['name']

        if combined_callback:
            combined_callback(f"Starting epoched processing of Session {n}: {session_name}...", progress=0)
        print(f"\nProcessing Session {n}: {session_name}")

        if check_stop(): return []

        # Process session...
        try:
            # First try user-specified channel
            events = mne.find_events(raw, stim_channel=args.stim_channel)
        except ValueError:
            try:
                # If no stim channel found, try to get events from annotations
                print("No stim channel found, looking for events in annotations...")
                
                # Get unique annotation descriptions
                unique_descriptions = set(raw.annotations.description)
                print(f"Found annotation types: {unique_descriptions}")
                
                # For TMS-EEG we typically want 'Stimulation' or similar annotations
                tms_annotations = ['Stimulation', 'TMS', 'R128', 'Response']
                
                # Create mapping for event IDs
                event_id = {}
                for desc in unique_descriptions:
                    # Look for TMS-related annotations
                    if any(tms_str.lower() in desc.lower() for tms_str in tms_annotations):
                        event_id[desc] = args.substitute_zero_events_with
                
                if not event_id:
                    print("No TMS-related annotations found. Using all annotations...")
                    # If no TMS annotations found, use all annotations
                    for i, desc in enumerate(unique_descriptions, 1):
                        event_id[desc] = i
                
                print(f"Using event mapping: {event_id}")
                
                # Get events from annotations
                events, _ = mne.events_from_annotations(raw, event_id=event_id)
                
                if len(events) == 0:
                    raise ValueError("No events found in annotations")
                    
                print(f"Found {len(events)} events from annotations")

                if combined_callback:
                    combined_callback(f"found {len(events)} events from annotations", progress=5)
                
            except Exception as e:
                # If both methods fail, try common stim channels
                print(f"Could not get events from annotations: {str(e)}")
                print("Trying common stim channel names...")
                
                stim_channels = mne.pick_types(raw.info, stim=True, exclude=[])
                if len(stim_channels) > 0:
                    stim_ch_name = raw.ch_names[stim_channels[0]]
                    print(f"Using detected stim channel: {stim_ch_name}")
                    events = mne.find_events(raw, stim_channel=stim_ch_name)
                else:
                    common_stim_names = ['STI 014', 'STIM', 'STI101', 'trigger', 'STI 001']
                    found = False
                    for ch_name in common_stim_names:
                        if ch_name in raw.ch_names:
                            print(f"Using stim channel: {ch_name}")
                            events = mne.find_events(raw, stim_channel=ch_name)
                            found = True
                            break
                    if not found:
                        raise ValueError(f"Could not find events in data. Available channels: {raw.ch_names}")

        annotations = mne.annotations_from_events(
            events=events, 
            sfreq=raw.info['sfreq'],
            event_desc={args.substitute_zero_events_with: 'Stimulation'}
        )
        raw.set_annotations(annotations)

        # Drop unnecessary channels
        channels_to_drop = []
        if 'EMG1' in raw.ch_names:
            channels_to_drop.append('EMG1')
        if channels_to_drop:
            print(f"Dropping channels: {channels_to_drop}")
            raw.drop_channels(channels_to_drop)
        if args.plot_raw:
           save_raw_data(raw, args.output_dir, step_name='raw', session_name=session_name)

        # Preprocessing
        processor = TMSEEGPreprocessor(raw, initial_sfreq=args.initial_sfreq, final_sfreq=args.final_sfreq)
        print("\nRemoving TMS artifact...")
        if combined_callback:
            combined_callback("Removing TMS artifact", progress=10)
        if check_stop(): return []
        processor.remove_tms_artifact(cut_times_tms=(-2, 10))  # Step 8

        print("\nInterpolating TMS artifact...")
        if combined_callback:
            combined_callback("Interpolating artifact", progress=20)
        processor.interpolate_tms_artifact(method='cubic', 
                                        interp_window=1.0,  # 1ms window for initial interpolation
                                        cut_times_tms=(-2, 10))  # Step 9

        if args.save_preproc:
            save_raw_data(raw, args.output_dir, step_name='raw_i')
        #processor.fix_tms_artifact(window=(args.fix_artifact_window_start, args.fix_artifact_window_end))
        if args.filter_raw:
            print("\nFiltering raw eeg data...")
            if check_stop(): return []
            processor.filter_raw(l_freq=args.l_freq, h_freq=args.h_freq, notch_freq=args.notch_freq, notch_width=args.notch_width)

        #print("\nPerforming initial downsampling...")
       # processor.initial_downsample()

        #if args.save_preproc:
          #  save_raw_data(raw, args.output_dir, step_name='raw_f',)

        print("\nCreating epochs...")
        processor.create_epochs(tmin=args.epochs_tmin, tmax=args.epochs_tmax, baseline=None, amplitude_threshold=args.amplitude_threshold)
        epochs = processor.epochs

        print("\nRemoving bad channels...")
        if combined_callback:
            combined_callback("Removing bad channels", progress=30)
        processor.remove_bad_channels(threshold=args.bad_channels_threshold)

        print("\nRemoving bad epochs...")
        if combined_callback:
            combined_callback("Removing bad epochs", progress=40)
        processor.remove_bad_epochs(threshold=args.bad_epochs_threshold)
        if args.save_preproc:
            save_epochs_data(processor.epochs, args.output_dir, session_name=session_name, step_name='epochs')

        print("\nSetting average reference...")
        processor.set_average_reference()

        print("\nRunning first ICA...")
        if combined_callback:
            combined_callback("Running first ICA", progress=50)
        if check_stop(): return []

        #plot_components = False

        if args.first_ica_manual:
            # Check if we're running in GUI mode
            gui_mode = hasattr(args, 'gui_mode') and args.gui_mode
            first_ica_callback = get_ica_callback(gui_mode=gui_mode)

        # Modified first ICA call
        if args.first_ica_manual:
            processor.run_ica(
                output_dir=args.output_dir,
                session_name=session_name,
                method=args.ica_method,
                tms_muscle_thresh=args.tms_muscle_thresh,
                blink_thresh=args.blink_thresh,  # Add these
                lat_eye_thresh=args.lat_eye_thresh,  # Add these
                muscle_thresh=args.muscle_thresh,  # Add these
                noise_thresh=args.noise_thresh,  # Add these
                manual_mode=True,
                ica_callback=first_ica_callback
            )
        else:
            processor.run_ica(
                output_dir=args.output_dir,
                session_name=session_name,
                method=args.ica_method,
                tms_muscle_thresh=args.tms_muscle_thresh,
                manual_mode=False
            )

        if args.save_preproc:
            save_epochs_data(processor.epochs, args.output_dir, session_name=session_name, step_name='ica1')

        if args.parafac_muscle_artifacts:
            print("\nCleaning muscle artifacts with PARAFAC decomposition...")
            if combined_callback:
                combined_callback("Cleaning muscle artifacts with PARAFAC decomposition...", progress=55)
            if check_stop(): return []
            processor.clean_muscle_artifacts(
                muscle_window=(args.muscle_window_start, args.muscle_window_end),
                threshold_factor=args.threshold_factor,
                n_components=args.n_components,
                verbose=True
            )

        if not args.skip_second_artifact_removal:
            print("\nExtending TMS artifact removal window...")
            if combined_callback:
                combined_callback("Extending TMS artifact removal window...", progress=60)
            if check_stop(): return []
            processor.remove_tms_artifact(cut_times_tms=(-2, 15))

            print("\nInterpolating extended TMS artifact...")
            if combined_callback:
                combined_callback("Interpolating extended TMS artifact...", progress=65)
            processor.interpolate_tms_artifact(method='cubic',
                                               interp_window=5.0,
                                               cut_times_tms=(-2, 15))

        if not args.filter_raw:
            print("\nFiltering epoched data...")
            if combined_callback:
                combined_callback("Filtering epoched data...", progress=75)
            if check_stop(): return []
            if args.mne_filter_epochs:
                processor.mne_filter_epochs(
                    l_freq=args.l_freq,
                    h_freq=args.h_freq,
                    notch_freq=args.notch_freq,
                    notch_width=args.notch_width,
                )
            else:
                processor.scipy_filter_epochs(
                    l_freq=args.l_freq,
                    h_freq=args.h_freq,
                    notch_freq=args.notch_freq,
                    notch_width=args.notch_width,
                )

            if args.save_preproc:
                save_epochs_data(processor.epochs, args.output_dir, session_name=session_name, step_name='filtered')


        print("\nRunning second ICA...")
        if combined_callback:
            combined_callback("Running second ICA...", progress=85)
        if check_stop(): return []
        if args.second_ica_manual:
            # Check if we're running in GUI mode
            gui_mode = hasattr(args, 'gui_mode') and args.gui_mode
            second_ica_callback = get_ica_callback(gui_mode=gui_mode)
            processor.run_second_ica(
                method=args.second_ica_method,
                blink_thresh=args.blink_thresh,  # Add these
                lat_eye_thresh=args.lat_eye_thresh,  # Add these
                muscle_thresh=args.muscle_thresh,  # Add these
                noise_thresh=args.noise_thresh,  # Add these
                manual_mode=True,
                ica_callback=second_ica_callback
            )
        else:
            processor.run_second_ica(
                method=args.second_ica_method,
                manual_mode=False
            )

        if args.apply_ssp:    
            print("\nApplying SSP...")
            processor.apply_ssp(n_eeg=args.ssp_n_eeg)

        print("\nApplying baseline correction...")
        processor.apply_baseline_correction(baseline=(baseline_start_sec, baseline_end_sec))
        if args.save_preproc:
            save_epochs_data(processor.epochs, args.output_dir, session_name=session_name, step_name='ica2')
        if args.apply_csd:
            print("\nApplying CSD transformation...")
            processor.apply_csd(lambda2=args.lambda2, stiffness=args.stiffness)

        print("\nPerforming final downsampling...")
        if combined_callback:
            combined_callback("Performing final downsampling...", progress=95)
        processor.final_downsample()

        if args.save_preproc:
            save_epochs_data(processor.epochs, args.output_dir, session_name, step_name='final')

        if not args.no_preproc_output:

            epochs = processor.epochs
            epochs.save(Path(args.output_dir) / f"{session_name}_preproc-epo.fif", verbose=True, overwrite=True)

        if args.validate_teps:
            try:
                print("\nAnalyzing TEPs...")
                if combined_callback:
                    combined_callback("Performing final downsampling...", progress=99)
                if check_stop(): return []

                # Define our standard TEP components exactly as in TESA
                DEFAULT_TEP_COMPONENTS = {
                    'N15': {'time': (12, 18), 'polarity': 'negative', 'peak': 15},
                    'P30': {'time': (25, 35), 'polarity': 'positive', 'peak': 30},
                    'N45': {'time': (36, 57), 'polarity': 'negative', 'peak': 45},
                    'P60': {'time': (58, 80), 'polarity': 'positive', 'peak': 60},
                    'N100': {'time': (81, 144), 'polarity': 'negative', 'peak': 100},
                    'P180': {'time': (145, 250), 'polarity': 'positive', 'peak': 180}
                }

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

                # Create visualization
                print("Generating TEP plots...")
                plot_tep_analysis(
                    epochs=epochs,
                    output_dir=args.output_dir,
                    session_name=session_name,
                    components=DEFAULT_TEP_COMPONENTS,
                    analysis_type=args.tep_analysis_type
                )

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
        
        # Final quality check
        # Maybe use ylim={'eeg': [-2, 2]},
        fig = processor.plot_evoked_response(xlim=(-0.1, 0.4), title="Final Evoked Response", show=args.save_evoked)
        fig.savefig(f"{args.output_dir}/evoked_{session_name}.png")  
        plt.close(fig)
        
        recording_id = f"session_{n}"

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
    if combined_callback:
        combined_callback("Processing complete! (wow it works)", progress=100)
    return subject_pcist_values


def process_continuous_data(args, status_callback=None):
    """
    Process TMS-EEG data maintaining continuous signal.
    Focuses on raw data processing with visualization.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments controlling processing parameters

    Returns
    -------
    list
        List of processed Raw objects
    """
    import builtins
    setattr(builtins, 'STOP_PROCESSING', False)
    from .preproc_vis import save_raw_data, create_step_directory

    processed_raws = []
    subject_pcist_values = []  # Initialize as an empty list
    pcist_objects = []  # Initialize as an empty list
    pcist_details = []  # Initialize as an empty list
    session_names = []

    def check_stop():
        if getattr(builtins, 'STOP_PROCESSING', False):
            print("\nProcessing stopped by user")
            return True
        return False

    # Initialize paths
    data_dir = Path(args.data_dir)
    TMS_DATA_PATH = data_dir / 'TMSEEG'

    # Verify paths
    required_paths = {
        'Data Directory': data_dir,
        'TMS Data': TMS_DATA_PATH,
    }
    for name, path in required_paths.items():
        if not path.exists():
            print(f"WARNING: {name} not found at: {path}")
        else:
            print(f"✓ Found {name} at: {path}")

    # Store processed raw objects
    processed_raws = []

    # Load the raw data
    from .dataloader import TMSEEGLoader
    loader = TMSEEGLoader(
        data_path=TMS_DATA_PATH,
        format=args.data_format,
        substitute_zero_events_with=args.substitute_zero_events_with,
        eeglab_montage_units=args.eeglab_montage_units,
        verbose=True
    )

    raw_list = loader.load_data()
    session_info = loader.get_session_info()

    # Process each raw file
    for n, raw in enumerate(raw_list):
        if check_stop():
            return processed_raws

        session_name = session_info[n]['name']
        print(f"\nProcessing Session {n}: {session_name}")

        try:
            # Plot initial raw data
            if args.plot_raw:
                print("\nPlotting initial raw data...")
                save_raw_data(raw, args.output_dir, session_name, "01_initial")

            # Get events (needed for TMS artifact removal)
            try:
                events = mne.find_events(raw, stim_channel=args.stim_channel)
            except ValueError:
                try:
                    print("No stim channel found, looking for events in annotations...")
                    unique_descriptions = set(raw.annotations.description)
                    print(f"Found annotation types: {unique_descriptions}")

                    tms_annotations = ['Stimulation', 'TMS', 'R128', 'Response']
                    event_id = {}
                    for desc in unique_descriptions:
                        if any(tms_str.lower() in desc.lower() for tms_str in tms_annotations):
                            event_id[desc] = args.substitute_zero_events_with

                    if not event_id:
                        print("No TMS-related annotations found. Using all annotations...")
                        for i, desc in enumerate(unique_descriptions, 1):
                            event_id[desc] = i

                    print(f"Using event mapping: {event_id}")
                    events, _ = mne.events_from_annotations(raw, event_id=event_id)

                    if len(events) == 0:
                        raise ValueError("No events found in annotations")

                except Exception as e:
                    print(f"Could not get events from annotations: {str(e)}")
                    print("Trying common stim channel names...")

                    stim_channels = mne.pick_types(raw.info, stim=True, exclude=[])
                    if len(stim_channels) > 0:
                        stim_ch_name = raw.ch_names[stim_channels[0]]
                        print(f"Using detected stim channel: {stim_ch_name}")
                        events = mne.find_events(raw, stim_channel=stim_ch_name)
                    else:
                        common_stim_names = ['STI 014', 'STIM', 'STI101', 'trigger', 'STI 001']
                        found = False
                        for ch_name in common_stim_names:
                            if ch_name in raw.ch_names:
                                print(f"Using stim channel: {ch_name}")
                                events = mne.find_events(raw, stim_channel=ch_name)
                                found = True
                                break
                        if not found:
                            raise ValueError(f"Could not find events in data. Available channels: {raw.ch_names}")

            print(f"Found {len(events)} events")
            annotations = mne.annotations_from_events(
                events=events,
                sfreq=raw.info['sfreq'],
                event_desc={args.substitute_zero_events_with: 'Stimulation'}
            )
            raw.set_annotations(annotations)

            # Drop unnecessary channels
            channels_to_drop = []
            if 'EMG1' in raw.ch_names:
                channels_to_drop.append('EMG1')
            if channels_to_drop:
                print(f"Dropping channels: {channels_to_drop}")
                raw.drop_channels(channels_to_drop)

            # Initialize preprocessor with raw data
            processor = TMSEEGPreprocessor(raw, initial_sfreq=args.initial_sfreq, final_sfreq=args.final_sfreq)

            print("\nRemoving TMS artifact and muscle peaks...")
            if check_stop():
                return processed_raws

            processor.remove_tms_artifact(cut_times_tms=(-2, 10))
            if args.save_preproc:
                save_raw_data(processor.raw, args.output_dir, session_name, "02_post_tms_removal")

            print("\nInterpolating TMS artifact...")
            processor.interpolate_tms_artifact(
                method='cubic',
                interp_window=1.0,
                cut_times_tms=(-2, 10)
            )
            if args.save_preproc:
                save_raw_data(processor.raw, args.output_dir, session_name, "03_post_interpolation")

           # print("\nPerforming initial downsampling...")
            #processor.initial_downsample()
           # if args.save_preproc:
            #    save_raw_data(processor.raw, args.output_dir, session_name, "04_post_downsample")

            print("\nRunning ICA...")
            if check_stop():
                return processed_raws

            # Modify the ICA handling to properly use the callback:
            if args.first_ica_manual and hasattr(args, 'ica_callback'):
                processor.run_ica(
                    output_dir=args.output_dir,
                    session_name=session_name,
                    method=args.ica_method,
                    tms_muscle_thresh=args.tms_muscle_thresh,
                    blink_thresh=args.blink_thresh,  # Add these
                    lat_eye_thresh=args.lat_eye_thresh,  # Add these
                    muscle_thresh=args.muscle_thresh,  # Add these
                    noise_thresh=args.noise_thresh,  # Add these
                    manual_mode=True,
                    ica_callback=args.ica_callback
                )
            else:
                processor.run_ica(
                    output_dir=args.output_dir,
                    session_name=session_name,
                    method=args.ica_method,
                    tms_muscle_thresh=args.tms_muscle_thresh,
                    manual_mode=False
                )

            if args.save_preproc:
                save_raw_data(processor.raw, args.output_dir, session_name, "05_post_ica")


            print("\nFiltering raw data...")
            if check_stop():
                return processed_raws
            processor.filter_raw(
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                notch_freq=args.notch_freq,
                notch_width=args.notch_width
            )

            if not args.no_second_ICA:
                print("\nRunning second ICA...")
                if check_stop():
                    return processed_raws

                if args.second_ica_manual and hasattr(args, 'ica_callback'):
                    processor.run_second_ica(
                        method=args.second_ica_method,
                        blink_thresh=args.blink_thresh,  # Add these
                        lat_eye_thresh=args.lat_eye_thresh,  # Add these
                        muscle_thresh=args.muscle_thresh,  # Add these
                        noise_thresh=args.noise_thresh,  # Add these
                        manual_mode=True,
                        ica_callback=args.ica_callback
                    )
                else:
                    processor.run_second_ica(
                        method=args.second_ica_method,
                        manual_mode=False
                    )

                if args.save_preproc:
                    save_raw_data(processor.raw, args.output_dir, session_name, "06_post_filter")

            # Final resampling if needed
            if processor.raw.info['sfreq'] > args.final_sfreq:
                print("\nPerforming final resampling...")
                processor.raw.resample(args.final_sfreq)
                if args.save_preproc:
                    save_raw_data(processor.raw, args.output_dir, session_name, "07_final")

            # Save processed raw data
            output_path = Path(args.output_dir) / f"{session_name}_processed_continuous_raw.fif"
            processor.raw.save(output_path, overwrite=True)
            processed_raws.append(processor.raw)

            print(f"\nProcessed continuous data saved to: {output_path}")

            # After continuous processing, create epochs with baseline correction
            print("\nCreating epochs from continuous data...")
            baseline_start_sec = args.baseline_start / 1000.0
            baseline_end_sec = args.baseline_end / 1000.0
            processor.create_epochs(
                tmin=args.epochs_tmin,
                tmax=args.epochs_tmax,
                baseline=(baseline_start_sec, baseline_end_sec),
                amplitude_threshold=args.amplitude_threshold
            )

            print("\nRemoving bad channels...")
            processor.remove_bad_channels(threshold=args.bad_channels_threshold)

            print("\nRemoving bad epochs...")
            processor.remove_bad_epochs(threshold=args.bad_epochs_threshold)

            if args.save_preproc:
                save_epochs_data(processor.epochs, args.output_dir, session_name, step_name='post_continuous')

            if args.apply_csd:
                print("\nApplying CSD transformation...")
                processor.apply_csd(lambda2=args.lambda2, stiffness=args.stiffness)

            print("\nPerforming final downsampling...")
            processor.final_downsample()

            if args.save_preproc:
                save_epochs_data(processor.epochs, args.output_dir, session_name, step_name='final')

            epochs = processor.epochs
            epochs.save(Path(args.output_dir) / f"{session_name}_preproc-epo.fif", verbose=True, overwrite=True)

            if args.validate_teps:
                try:
                    print("\nAnalyzing TEPs...")
                    if check_stop(): return []

                    # Define our standard TEP components exactly as in TESA
                    DEFAULT_TEP_COMPONENTS = {
                        'N15': {'time': (12, 18), 'polarity': 'negative', 'peak': 15},
                        'P30': {'time': (25, 35), 'polarity': 'positive', 'peak': 30},
                        'N45': {'time': (36, 57), 'polarity': 'negative', 'peak': 45},
                        'P60': {'time': (58, 80), 'polarity': 'positive', 'peak': 60},
                        'N100': {'time': (81, 144), 'polarity': 'negative', 'peak': 100},
                        'P180': {'time': (145, 250), 'polarity': 'positive', 'peak': 180}
                    }

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

                    # Create visualization
                    print("Generating TEP plots...")
                    plot_tep_analysis(
                        epochs=epochs,
                        output_dir=args.output_dir,
                        session_name=session_name,
                        components=DEFAULT_TEP_COMPONENTS,
                        analysis_type=args.tep_analysis_type
                    )

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

                # Plot evoked response
            fig = processor.plot_evoked_response(xlim=(-0.1, 0.4), title="Final Evoked Response", show=args.save_evoked)
            fig.savefig(f"{args.output_dir}/evoked_{session_name}.png")
            plt.close(fig)
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

                processed_raws.append(processor.raw)

        except Exception as e:
            print(f"Error processing session {session_name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue

    return processed_raws, subject_pcist_values, pcist_objects, pcist_details, session_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process EEG data.')
    # Add this new argument
    parser.add_argument('--processing_mode',
                        choices=['epoched', 'continuous'],
                        default='epoched',
                        help='Processing mode: epoched or continuous (default: epoched)')
    parser.add_argument('--data_dir', type=str, default=str(Path.cwd() / 'data'), 
                        help='Path to the data directory (default: ./data)')
    parser.add_argument('--output_dir', type=str, default=str(Path.cwd() / 'output'), 
                        help='Path to the output directory (default: ./output)')
    parser.add_argument('--data_format', type=str, default='neurone',
                   choices=['neurone', 'brainvision', 'edf', 'cnt', 'eeglab', 'auto'],
                   help='Format of input data (default: neurone)')
    parser.add_argument('--no_preproc_output', action='store_true', default=False,
                    help='Skip saving preprocessed epochs (default: False)')
    parser.add_argument('--no_pcist', action='store_true', default=False,
                    help='Skip PCIst calculation and only preprocess (default: False)')
    parser.add_argument('--eeglab_montage_units', type=str, default='auto',
                   help='Units for EEGLAB channel positions (default: auto)')
    parser.add_argument('--stim_channel', type=str, default='STI 014',
                    help='Name of the stimulus channel (default: STI 014)')
    parser.add_argument('--save_preproc', action='store_true', default=False,
                    help='Save plots between preprocessing steps (default: False)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--substitute_zero_events_with', type=int, default=10,
                        help='Value to substitute zero events with (default: 10)')
    parser.add_argument('--initial_sfreq', type=float, default=1000,
                        help='Initial downsampling frequency (default: 1000)')
    parser.add_argument('--final_sfreq', type=float, default=725,
                        help='Final downsampling frequency (default: 725)')
    # Trying to match TESA
    parser.add_argument('--initial_window_start', type=float, default=-2,
                    help='Initial TMS artifact window start (TESA default: -2)')
    parser.add_argument('--initial_window_end', type=float, default=10,
                        help='Initial TMS artifact window end (TESA default: 10)')
    parser.add_argument('--extended_window_start', type=float, default=-2,
                        help='Extended TMS artifact window start (TESA default: -2)')
    parser.add_argument('--extended_window_end', type=float, default=15,
                        help='Extended TMS artifact window end (TESA default: 15)')
    parser.add_argument('--initial_interp_window', type=float, default=1.0,
                        help='Initial interpolation window (TESA default: 1.0)')
    parser.add_argument('--extended_interp_window', type=float, default=5.0,
                        help='Extended interpolation window (TESA default: 5.0)')
    parser.add_argument('--interpolation_method', type=str, default='cubic',
                        choices=['cubic'],
                        help='Interpolation method (TESA requires cubic)')
    parser.add_argument('--skip_second_artifact_removal', action='store_true',
                    help='Skip the second stage of TMS artifact removal')
    parser.add_argument('--mne_filter_epochs', action='store_true', default=False,
                    help='Use built in filter in mne (default: False)')
    parser.add_argument('--plot_raw', action='store_true',
                    help='Plot raw data (takes time) (default: False)')
    parser.add_argument('--filter_raw', type=bool, default=False,
                        help='Whether to filter raw data instead of epoched (default: False)')
    parser.add_argument('--l_freq', type=float, default=0.1,
                        help='Lower frequency for filtering (default: 1)')
    parser.add_argument('--h_freq', type=float, default=45,
                        help='Upper frequency for filtering (default: 45)')
    parser.add_argument('--notch_freq', type=float, default=50,
                        help='Notch filter frequency (default: 50)')
    parser.add_argument('--notch_width', type=float, default=2,
                        help='Notch filter width (default: 2)')
    parser.add_argument('--epochs_tmin', type=float, default=-0.41,
                        help='Start time for epochs (default: -0.41)')
    parser.add_argument('--epochs_tmax', type=float, default=0.41,
                        help='End time for epochs (default: 0.41)')
    parser.add_argument('--bad_channels_threshold', type=float, default=2,
                        help='Threshold (std) for removing bad channels with mne_faster (default: 2)')
    parser.add_argument('--bad_epochs_threshold', type=float, default=2,
                        help='Threshold (std) for removing bad epochs with mne_faster (default: 2)')
    parser.add_argument('--ica_method', type=str, default='fastica',
                        help='ICA method (default: fastica)')
    parser.add_argument('--blink_thresh', type=float, default=2.5,
                        help='Threshold for blink detection (default: 2.5)')
    parser.add_argument('--lat_eye_thresh', type=float, default=2.0,
                        help='Threshold for lateral eye movement detection (default: 2.0)')
    parser.add_argument('--noise_thresh', type=float, default=4.0,
                        help='Threshold for noise detection (default: 4.0)')
    parser.add_argument('--tms_muscle_thresh', type=float, default=2.0,
                        help='Threshold for TMS muscle artifact (default: 2.0)')
    parser.add_argument('--muscle_thresh', type=float, default=0.6,
                        help='Threshold for ongoing muscle contamination (default: 0.6)')
    parser.add_argument('--parafac_muscle_artifacts', action='store_true', default=False,
                    help='Enable muscle artifact cleaning (default: False)')
    parser.add_argument('--muscle_window_start', type=float, default=0.005,
                    help='Start time for muscle artifact window (default: 0.005)')
    parser.add_argument('--muscle_window_end', type=float, default=0.030,
                    help='End time for muscle artifact window (default: 0.030)')
    parser.add_argument('--threshold_factor', type=float, default=1.0,
                    help='Threshold factor for muscle artifact cleaning (default: 1.0)')
    parser.add_argument('--first_ica_manual', action='store_true', default=False,
                        help='Enable manual component selection for first ICA (default: False)')
    parser.add_argument('--second_ica_manual', action='store_true', default=False,
                        help='Enable manual component selection for second ICA (default: False)')
    parser.add_argument('--n_components', type=int, default=5,
                    help='Number of components for muscle artifact cleaning (default: 5)')
    parser.add_argument('--no_second_ICA', action='store_true',
                    help='Disable seconds ICA using ICA_label (default: False)')
    parser.add_argument('--second_ica_method', type=str, default='fastica',
                        help='Second ICA method (default: fastica)')
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
    parser.add_argument('--validate_teps', action='store_true', default=True,
                help='Find TEPs that normally exist (default: True)')
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
    parser.add_argument('--amplitude_threshold', type=float, default=300.0,
                    help='Threshold for epoch rejection based on peak-to-peak amplitude in µV (default: 300.0)')
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

    if args.processing_mode == 'epoched':
        pcists, subject_pcist_values, pcist_objects, pcist_details, session_names = process_subjects(args)
        print(f"PCIst values: {pcists}")
    else:
        processed_raws, subject_pcist_values, pcist_objects, pcist_details, session_names = process_continuous_data(
            args)
        print(f"PCIst values: {subject_pcist_values}")