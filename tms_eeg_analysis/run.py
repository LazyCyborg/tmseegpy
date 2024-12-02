import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from preproc import TMSEEGPreprocessor
from pcist import PCIst
import mne
import time
from neurone_loader import Recording
import argparse

mne.viz.use_browser_backend("matplotlib")
plt.rcParams['figure.figsize'] = [8, 6]

def plot_eeg(eeg=None, duration=None, start=None):
    with mne.viz.use_browser_backend("matplotlib"):
        fig = eeg.copy().set_eeg_reference('average').plot(duration=duration, start=start, scalings=None)
        for ax in fig.get_axes():
            if hasattr(ax, 'invert_yaxis'):
                ax.invert_yaxis()
        fig.canvas.draw()
        
def plot_epochs(eeg=None):
    with mne.viz.use_browser_backend("matplotlib"):
        fig = eeg.copy().set_eeg_reference('average').plot(scalings=None)
        for ax in fig.get_axes():
            if hasattr(ax, 'invert_yaxis'):
                ax.invert_yaxis()
        fig.canvas.draw()

def generate_preproc_stats(processor, session_name, output_dir):
    """
    Generate detailed preprocessing quality control statistics.
    
    Args:
        processor: TMSEEGPreprocessor object containing preprocessing information
        session_name: Name of the current session
        output_dir: Directory to save the output file
    """
    import numpy as np
    from pathlib import Path
    import datetime
    
    output_file = Path(output_dir) / f"preproc_stats_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(output_file, 'w') as f:
        # Header
        f.write("TMS-EEG Preprocessing Quality Control Report\n")
        f.write("=" * 50 + "\n\n")
        
        # 1. Recording Information
        f.write("1. RECORDING INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Session: {session_name}\n")
        f.write(f"Original sampling rate: {processor.raw.info['sfreq']} Hz\n")
        f.write(f"Downsampled to: {processor.ds_sfreq} Hz\n")
        f.write(f"Duration: {processor.raw.times[-1]:.1f} seconds\n")
        f.write(f"Number of channels: {len(processor.raw.ch_names)}\n")
        
        # 2. TMS Events Analysis
        f.write("\n2. TMS EVENTS ANALYSIS\n")
        f.write("-" * 40 + "\n")
        events = processor._get_events()
        n_events = len(events)
        f.write(f"Total TMS pulses: {n_events}\n")
        
        if n_events > 1:
            intervals = np.diff([event[0] for event in events]) / processor.raw.info['sfreq']
            f.write(f"Inter-pulse intervals:\n")
            f.write(f"  Mean: {np.mean(intervals):.3f} seconds\n")
            f.write(f"  Std: {np.std(intervals):.3f} seconds\n")
            f.write(f"  Range: {np.min(intervals):.3f} - {np.max(intervals):.3f} seconds\n")
            f.write(f"  CV: {(np.std(intervals)/np.mean(intervals))*100:.1f}%\n")
        
        # 3. Data Quality Metrics
        f.write("\n3. DATA QUALITY METRICS\n")
        f.write("-" * 40 + "\n")
        
        # Channel info
        if hasattr(processor, 'bad_channels'):
            f.write("\nChannel Quality:\n")
            f.write(f"Original channels: {len(processor.raw.ch_names)}\n")
            f.write(f"Bad channels detected: {len(processor.bad_channels)}\n")
            if processor.bad_channels:
                f.write(f"Bad channels: {', '.join(processor.bad_channels)}\n")
            f.write(f"Channel retention rate: {(len(processor.raw.ch_names) - len(processor.bad_channels))/len(processor.raw.ch_names)*100:.1f}%\n")
        
        # Epoch info
        if hasattr(processor, 'epochs'):
            f.write("\nEpoch Quality:\n")
            f.write(f"Total epochs created: {len(processor.epochs)}\n")
            if hasattr(processor, 'bad_epochs'):
                f.write(f"Bad epochs detected: {len(processor.bad_epochs)}\n")
                retention_rate = (len(processor.epochs) - len(processor.bad_epochs))/len(processor.epochs)*100
                f.write(f"Epoch retention rate: {retention_rate:.1f}%\n")
        
        # 4. Artifact Removal Performance
        f.write("\n4. ARTIFACT REMOVAL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        
        f.write("\nTMS Artifact Removal:\n")
        if hasattr(processor, 'interpolation_method'):
            f.write(f"Method: {processor.interpolation_method}\n")
        else:
            f.write("Method: Not available\n")
        
        if hasattr(processor, 'interp_window'):
            f.write(f"Interpolation window: {processor.interp_window} ms\n")
        else:
            f.write("Interpolation window: Not available\n")
        
        # ICA info
        if hasattr(processor, 'ica'):
            f.write("\nFirst ICA:\n")
            f.write(f"Method: {processor.ica.method}\n")
            if hasattr(processor, 'muscle_components'):
                f.write(f"Muscle components removed: {len(processor.muscle_components)}\n")
                if processor.muscle_components:
                    f.write(f"Component indices: {processor.muscle_components}\n")
        
        if hasattr(processor, 'ica2'):
            f.write("\nSecond ICA:\n")
            f.write(f"Method: {processor.ica2.method}\n")
            if hasattr(processor, 'excluded_components'):
                f.write(f"Components excluded: {len(processor.excluded_components)}\n")
                if processor.excluded_components:
                    f.write(f"Component indices: {processor.excluded_components}\n")
        
        # 5. Quality Control Summary
        f.write("\n5. QUALITY CONTROL SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        # Calculate quality metrics
        quality_metrics = {}
        
        # Channel quality (weight: 0.3)
        channel_quality = (len(processor.raw.ch_names) - len(getattr(processor, 'bad_channels', []))) / len(processor.raw.ch_names)
        quality_metrics['Channel Quality'] = channel_quality
        
        # Epoch quality (weight: 0.3)
        if hasattr(processor, 'epochs') and hasattr(processor, 'bad_epochs'):
            epoch_quality = (len(processor.epochs) - len(processor.bad_epochs)) / len(processor.epochs)
        else:
            epoch_quality = 1.0
        quality_metrics['Epoch Quality'] = epoch_quality
        
        # Artifact removal quality (weight: 0.4)
        if hasattr(processor, 'muscle_components') and hasattr(processor, 'excluded_components'):
            artifact_quality = 1 - (len(processor.muscle_components) + len(processor.excluded_components)) / (processor.ica.n_components_ * 2)
        else:
            artifact_quality = 1.0
        quality_metrics['Artifact Removal Quality'] = artifact_quality
        
        # Calculate weighted overall quality
        weights = {'Channel Quality': 0.3, 'Epoch Quality': 0.3, 'Artifact Removal Quality': 0.4}
        overall_quality = sum(quality_metrics[k] * weights[k] for k in quality_metrics)
        
        # Write quality metrics
        for metric, value in quality_metrics.items():
            f.write(f"{metric}: {value*100:.1f}%\n")
        f.write(f"\nOverall Quality Score: {overall_quality*100:.1f}%\n")
        
        # Add warnings if needed
        if overall_quality < 0.7:
            f.write("\nWARNINGS:\n")
            if channel_quality < 0.8:
                f.write("- High number of bad channels detected\n")
            if epoch_quality < 0.8:
                f.write("- High number of epochs rejected\n")
            if artifact_quality < 0.7:
                f.write("- Large number of artifact components removed\n")
            
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
    # Load the raw data for the session
    neurone_path = str(TMS_DATA_PATH)
    rec = Recording(neurone_path)
    np.random.seed(args.random_seed)


    baseline_start_sec = args.baseline_start / 1000.0
    baseline_end_sec = args.baseline_end / 1000.0

    # Loop through sessions 
    for n, r in enumerate(range(len(rec.sessions))):
        print(f"\nProcessing Session {n}: {rec.sessions[r].path}")
        # Load and prepare raw data
        session = rec.sessions[r]
        session_path = rec.sessions[r].path
        session_name = os.path.basename(session_path) 
        raw = rec.sessions[r].to_mne(substitute_zero_events_with=args.substitute_zero_events_with)
        # Process session...
        events = mne.find_events(raw, stim_channel='STI 014')
        annotations = mne.annotations_from_events(events=events, sfreq=raw.info['sfreq'],event_desc={args.substitute_zero_events_with: 'Stimulation'})
        raw.set_annotations(annotations)
        # Drop unnecessary channels
        channels_to_drop = []
        if 'EMG1' in raw.ch_names:
            channels_to_drop.append('EMG1')
        if channels_to_drop:
            print(f"Dropping channels: {channels_to_drop}")
            raw.drop_channels(channels_to_drop)

        # Preprocessing
        processor = TMSEEGPreprocessor(raw, ds_sfreq=args.ds_sfreq)
        print("\nRemoving TMS artifact...")
        processor.remove_tms_artifact(cut_times_tms=(args.cut_times_tms_start, args.cut_times_tms_end))
        processor.interpolate_tms_artifact(method=args.interpolation_method, interp_window=(args.interp_window_start, args.interp_window_end),  cut_times_tms=(args.cut_times_tms_start, args.cut_times_tms_end))
        # 2. Fix stimulus artifact with native mne.fix_stim_artifact
        print("\nFixing stimulus artifact...")
        events = processor._get_events()
        event_id = processor._get_event_ids()
        processor.fix_tms_artifact(window=(args.fix_artifact_window_start, args.fix_artifact_window_end))
        processor.filter_raw(l_freq=args.l_freq, h_freq=args.h_freq, notch_freq=args.notch_freq, notch_width=args.notch_width, plot_psd=False) 
        print("\nCreating epochs...")
        processor.create_epochs(tmin=args.epochs_tmin, tmax=args.epochs_tmax, baseline=None)
        print("\nRemoving bad channels...")
        processor.remove_bad_channels(threshold=args.bad_channels_threshold)
        print("\nRemoving bad epochs...")
        processor.remove_bad_epochs(threshold=args.bad_epochs_threshold)
        processor.set_average_reference()
        print("\nRunning first ICA...")
        if args.plot_preproc:
            plot_components=True
        else:
            plot_components=False
        processor.run_ica(method=args.ica_method, tms_muscle_thresh=args.tms_muscle_thresh, plot_components=plot_components)
        if args.clean_muscle_artifacts:
            print("\nCleaning muscle artifacts...")
            processor.clean_muscle_artifacts(
                muscle_window=(args.muscle_window_start, args.muscle_window_end),
                threshold_factor=args.threshold_factor,
                n_components=args.n_components,
                verbose=True
            )
        if not args.no_second_ICA:
            print("\nRunning second ICA...")
            processor.run_second_ica(method=args.second_ica_method, exclude_labels=["eye blink", "heart beat", "muscle artifact"])

        print("\nApplying SSP...")
        processor.apply_ssp(n_eeg=args.ssp_n_eeg)
        print("\nApplying baseline correction...")
        processor.apply_baseline_correction(baseline=(baseline_start_sec, baseline_end_sec))
        if args.apply_csd:
            print("\nApplying CSD transformation...")
            processor.apply_csd(lambda2=args.lambda2, stiffness=args.stiffness)
        print(f"\nDownsampling to {processor.ds_sfreq} Hz")
        processor.downsample()

        epochs = processor.epochs
        if args.plot_preproc:
            plot_epochs(eeg=epochs)
        
        # Final quality check
        fig = processor.plot_evoked_response(ylim={'eeg': [-5, 5]}, xlim=(-0.3, 0.3), title="Final Evoked Response", show=args.show_evoked)
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

    
    if args.preproc_qc:
        preproc_stats_file = generate_preproc_stats(
            processor,
            session_name,
            args.output_dir
        )
        print(f"Preprocessing statistics saved to: {preproc_stats_file}")

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
    parser.add_argument('--plot_preproc', action='store_true',
                    help='Enable muscle artifact cleaning (default: False)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--substitute_zero_events_with', type=int, default=10,
                        help='Value to substitute zero events with (default: 10)')
    parser.add_argument('--ds_sfreq', type=float, default=725,
                        help='Downsampling frequency (default: 725)')
    parser.add_argument('--cut_times_tms_start', type=float, default=-2,
                        help='Start time for cutting TMS artifact (default: -2)')
    parser.add_argument('--cut_times_tms_end', type=float, default=10,
                        help='End time for cutting TMS artifact (default: 10)')
    parser.add_argument('--interpolation_method', type=str, default='cubic',
                        help='Interpolation method (default: cubic)')
    parser.add_argument('--interp_window_start', type=float, default=20,
                        help='Start time for interpolation window (default: 20)')
    parser.add_argument('--interp_window_end', type=float, default=20,
                        help='End time for interpolation window (default: 20)')
    parser.add_argument('--fix_artifact_window_start', type=float, default=-0.005,
                        help='Start time for fixing artifact window (default: -0.005)')
    parser.add_argument('--fix_artifact_window_end', type=float, default=0.015,
                        help='End time for fixing artifact window (default: 0.015)')
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
    parser.add_argument('--ssp_n_eeg', type=int, default=2,
                        help='Number of EEG components for SSP (default: 2)')
    parser.add_argument('--apply_csd', action='store_true',
                    help='Apply CSD transformation (default: False)')
    parser.add_argument('--lambda2', type=float, default=1e-5,
                    help='Lambda2 parameter for CSD transformation (default: 1e-5)')
    parser.add_argument('--stiffness', type=int, default=4,
                    help='Stiffness parameter for CSD transformation (default: 4)')
    parser.add_argument('--show_evoked', action='store_true',
                    help='Display the evoked plot with TEPs (default: False)')
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
    parser.add_argument('--preproc_qc', action='store_true',
                help='Generate preprocessing quality control statistics (default: False)')
    parser.add_argument('--research', action='store_true',
                    help='Output summary statistics of measurements (default: False)')
    args = parser.parse_args()

    pcists = process_subjects(args)
    print(f"PCIst values: {pcists}")