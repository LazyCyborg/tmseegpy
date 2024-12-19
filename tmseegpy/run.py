# run.py
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .preproc import TMSEEGPreprocessor
from .pcist import PCIst
from .preproc_vis import plot_raw_segments, plot_epochs_grid
from .validate_tep import plot_tep_analysis, generate_validation_summary
import mne
import time
from .neurone_loader import Recording
import argparse

mne.viz.use_browser_backend("matplotlib")
plt.rcParams['figure.figsize'] = [8, 6]


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
        f.write(f"Sampling rate: {processor.raw.info['sfreq']} → {processor.ds_sfreq} Hz\n")
        f.write(f"Channels: {len(processor.raw.ch_names)}\n\n")
        
        # TMS pulse information
        events = processor._get_events()
        n_events = len(events)
        f.write("TMS Pulses\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total pulses: {n_events}\n")
        if n_events > 1:
            intervals = np.diff([event[0] for event in events]) / processor.raw.info['sfreq']
            f.write(f"Mean interval: {np.mean(intervals):.3f} s (±{np.std(intervals):.3f})\n\n")
        
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


def process_subjects(args):
    import builtins
    setattr(builtins, 'STOP_PROCESSING', False)
    def check_stop():
        if getattr(builtins, 'STOP_PROCESSING', False):
            print("\nProcessing stopped by user")
            return True
        return False
    data_dir = Path(args.data_dir)
    # Specific paths for the subject
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
        print(f"\nProcessing Session {n}: {session_name}")

        if check_stop(): return []

        # Process session...
        try:
            # First try user-specified channel
            events = mne.find_events(raw, stim_channel=args.stim_channel)
        except ValueError:
            # If not found, try to detect stimulus channel automatically
            stim_channels = mne.pick_types(raw.info, stim=True, exclude=[])
            if len(stim_channels) > 0:
                # Use the first detected stim channel
                stim_ch_name = raw.ch_names[stim_channels[0]]
                print(f"Using detected stim channel: {stim_ch_name}")
                events = mne.find_events(raw, stim_channel=stim_ch_name)
            else:
                # If no stim channels found, try common names
                common_stim_names = ['STI 014', 'STIM', 'STI101', 'trigger', 'STI 001']
                found = False
                for ch_name in common_stim_names:
                    if ch_name in raw.ch_names:
                        print(f"Using stim channel: {ch_name}")
                        events = mne.find_events(raw, stim_channel=ch_name)
                        found = True
                        break
                if not found:
                    raise ValueError(f"Could not find stimulus channel. Available channels: {raw.ch_names}")

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
        #if args.plot_preproc:
           # plot_raw_segments(raw, args.output_dir, step_name='raw',)

        # Preprocessing
        processor = TMSEEGPreprocessor(raw, ds_sfreq=args.ds_sfreq)
        print("\nRemoving TMS artifact and muscle peaks...")
        if check_stop(): return []
        processor.remove_tms_artifact(cut_times_tms=(-2, 10))  # Step 8

        print("\nInterpolating TMS artifact...")
        processor.interpolate_tms_artifact(method='cubic', 
                                        interp_window=1.0,  # 1ms window for initial interpolation
                                        cut_times_tms=(-2, 10))  # Step 9

        events = processor._get_events()
        event_id = processor._get_event_ids()
        #if args.plot_preproc:
           # plot_raw_segments(raw, args.output_dir, step_name='raw_i',)
        #processor.fix_tms_artifact(window=(args.fix_artifact_window_start, args.fix_artifact_window_end))
        print("\nFiltering raw eeg data...")
        if check_stop(): return []
        processor.filter_raw(l_freq=args.l_freq, h_freq=args.h_freq, notch_freq=args.notch_freq, notch_width=args.notch_width, plot_psd=False)

        #if args.plot_preproc:
          #  plot_raw_segments(raw, args.output_dir, step_name='raw_f',)

        print("\nCreating epochs...")
        processor.create_epochs(tmin=args.epochs_tmin, tmax=args.epochs_tmax, baseline=None, amplitude_threshold=args.amplitude_threshold)
        epochs = processor.epochs

        print("\nRemoving bad channels...")
        processor.remove_bad_channels(threshold=args.bad_channels_threshold)

        print("\nRemoving bad epochs...")
        processor.remove_bad_epochs(threshold=args.bad_epochs_threshold)
        if args.plot_preproc:
            plot_epochs_grid(epochs, args.output_dir, session_name=session_name, step_name='epochs')

        print("\nSetting average reference...")
        processor.set_average_reference()

        print("\nRunning first ICA...")
        if check_stop(): return []
        if args.plot_preproc:
            plot_components=True
        else:
            plot_components=False
        processor.run_ica(method=args.ica_method, tms_muscle_thresh=args.tms_muscle_thresh, plot_components=plot_components)
        if args.plot_preproc:
            plot_epochs_grid(epochs, args.output_dir, session_name=session_name, step_name='ica1')
        if args.clean_muscle_artifacts:
            print("\nCleaning muscle artifacts...")
            if check_stop(): return []
            processor.clean_muscle_artifacts(
                muscle_window=(args.muscle_window_start, args.muscle_window_end),
                threshold_factor=args.threshold_factor,
                n_components=args.n_components,
                verbose=True
            )
        if args.plot_preproc:
            plot_epochs_grid(epochs, args.output_dir, session_name=session_name, step_name='clean')
        if not args.skip_second_artifact_removal:
            print("\nExtending TMS artifact removal window...")
            processor.remove_tms_artifact(cut_times_tms=(-2, 15))  
            
            print("\nInterpolating extended TMS artifact...")
            processor.interpolate_tms_artifact(method='cubic',
                                            interp_window=5.0,  
                                            cut_times_tms=(-2, 15))
            
        # https://mne.tools/mne-icalabel/stable/generated/examples/00_iclabel.html#sphx-glr-generated-examples-00-iclabel-py

        if not args.no_second_ICA:
            print("\nRunning second ICA...")
            if check_stop(): return []
            processor.run_second_ica(method=args.second_ica_method, exclude_labels=["eye blink", "heart beat", "muscle artifact", "channel noise", "line noise"])

        if not args.apply_ssp:    
            print("\nApplying SSP...")
            processor.apply_ssp(n_eeg=args.ssp_n_eeg)

        print("\nApplying baseline correction...")
        processor.apply_baseline_correction(baseline=(baseline_start_sec, baseline_end_sec))
        if args.plot_preproc:
            plot_epochs_grid(epochs, args.output_dir, session_name=session_name, step_name='ica2')
        if args.apply_csd:
            print("\nApplying CSD transformation...")
            processor.apply_csd(lambda2=args.lambda2, stiffness=args.stiffness)
    
        print(f"\nDownsampling to {processor.ds_sfreq} Hz")
        processor.downsample()
        if args.plot_preproc:
            plot_epochs_grid(epochs, args.output_dir, session_name, step_name='final')

        epochs = processor.epochs

        if args.validate_teps:
            try:
                print("\nAnalyzing TEPs...")
                if check_stop(): return []
                
                # Calculate evoked response
                evoked = epochs.average()
                
                # Basic validation checks
                if not evoked.times.size:
                    raise ValueError("No time points in evoked data")
                if not evoked.data.any():
                    raise ValueError("No data in evoked response")
                    
                print(f"Processing TEPs from {evoked.times[0]:.3f}s to {evoked.times[-1]:.3f}s")
                
                # Run TEP analysis
                components = plot_tep_analysis(
                    evoked=evoked,
                    output_dir=args.output_dir,
                    session_name=session_name,
                    prominence=args.prominence
                )
                
                # Check if any components were found
                if not components:
                    print("Warning: No TEP components were detected")
                else:
                    print(f"Found {len(components)} TEP components")
                    
                    if args.save_validation:
                        generate_validation_summary(
                            components,
                            args.output_dir,
                            session_name
                        )
                        print("TEP validation summary saved")
                        
            except Exception as e:
                print(f"Warning: TEP analysis failed: {str(e)}")
                import traceback
                print("Detailed error:")
                print(traceback.format_exc())
                components = None
        
        # Final quality check
        fig = processor.plot_evoked_response(ylim={'eeg': [-2, 2]}, xlim=(-0.3, 0.3), title="Final Evoked Response", show=args.show_evoked)
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

    
    #if args.preproc_qc:
       # preproc_stats_file = generate_preproc_stats(
           # processor,
           # session_name,
           # args.output_dir
       #)
       # print(f"Preprocessing statistics saved to: {preproc_stats_file}")

    if args.research:
        output_file = generate_research_stats(
            subject_pcist_values,
            pcist_objects,
            pcist_details,
            session_names,
            args.output_dir
        )
        print(f"Research statistics saved to: {output_file}")
 
    return subject_pcist_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process EEG data.')
    parser.add_argument('--data_dir', type=str, default=str(Path.cwd() / 'data'), 
                        help='Path to the data directory (default: ./data)')
    parser.add_argument('--output_dir', type=str, default=str(Path.cwd() / 'output'), 
                        help='Path to the output directory (default: ./output)')
    parser.add_argument('--data_format', type=str, default='neurone',
                   choices=['neurone', 'brainvision', 'edf', 'cnt', 'eeglab', 'auto'],
                   help='Format of input data (default: neurone)')
    parser.add_argument('--eeglab_montage_units', type=str, default='auto',
                   help='Units for EEGLAB channel positions (default: auto)')
    parser.add_argument('--stim_channel', type=str, default='STI 014',
                   help='Name of the stimulus channel (default: STI 014)')
    parser.add_argument('--plot_preproc', action='store_true',
                    help='Enable muscle artifact cleaning (default: False)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--substitute_zero_events_with', type=int, default=10,
                        help='Value to substitute zero events with (default: 10)')
    parser.add_argument('--ds_sfreq', type=float, default=725,
                        help='Downsampling frequency (default: 725)')
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
    parser.add_argument('--l_freq', type=float, default=0.1,
                        help='Lower frequency for filtering (default: 0.1)')
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
    parser.add_argument('--bad_channels_threshold', type=float, default=3,
                        help='Threshold for removing bad channels (default: 3)')
    parser.add_argument('--bad_epochs_threshold', type=float, default=3,
                        help='Threshold for removing bad epochs (default: 3)')
    parser.add_argument('--ica_method', type=str, default='fastica',
                        help='ICA method (default: fastica)')
    parser.add_argument('--tms_muscle_thresh', type=float, default=3.0,
                        help='Threshold for TMS muscle artifact (default: 3.0)')
    parser.add_argument('--clean_muscle_artifacts', action='store_true',
                    help='Enable muscle artifact cleaning (default: False)')
    parser.add_argument('--muscle_window_start', type=float, default=0.005,
                    help='Start time for muscle artifact window (default: 0.005)')
    parser.add_argument('--muscle_window_end', type=float, default=0.030,
                    help='End time for muscle artifact window (default: 0.030)')
    parser.add_argument('--threshold_factor', type=float, default=1.0,
                    help='Threshold factor for muscle artifact cleaning (default: 1.0)')
    parser.add_argument('--n_components', type=int, default=5,
                    help='Number of components for muscle artifact cleaning (default: 5)')
    parser.add_argument('--no_second_ICA', action='store_true',
                    help='Disable seconds ICA using ICA_label (default: False)')
    parser.add_argument('--second_ica_method', type=str, default='infomax',
                        help='Second ICA method (default: infomax)')
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
    parser.add_argument('--show_evoked', action='store_true',
                    help='Display the evoked plot with TEPs (default: False)')
    parser.add_argument('--validate_teps', action='store_true',
                help='Perform TEP validation against established criteria')
    parser.add_argument('--save_validation', action='store_true',
                help='Save TEP validation summary (default: False)')
    parser.add_argument('--prominence', type=float, default=0.01,
                    help='Minimum prominence for peak detection (default: 0.01)')
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

    pcists = process_subjects(args)
    print(f"PCIst values: {pcists}")