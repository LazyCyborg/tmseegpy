import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tmseegpy.preproc import TMSEEGPreprocessor
from tmseegpy.pcist import PCIst
from tmseegpy.microstates import Microstate
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

def process_subjects(args):
    data_dir = Path(args.data_dir)
    # Specific paths for the subject
    TMS_DATA_PATH = data_dir / 'TMS1'
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
    # Load the raw data for the session
    neurone_path = str(TMS_DATA_PATH)
    rec = Recording(neurone_path)
    np.random.seed(args.random_seed)

    ma = Microstate()
    # Loop through sessions 
    for n, r in enumerate(range(len(rec.sessions))):
        print(f"\nProcessing Session {n}: {rec.sessions[r].path}")
        # Load and prepare raw data
        session = rec.sessions[r]
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
        processor.run_ica(method=args.ica_method, tms_muscle_thresh=args.tms_muscle_thresh, plot_components=True)
        if args.clean_muscle_artifacts:
            print("\nCleaning muscle artifacts...")
            processor.clean_muscle_artifacts(
                muscle_window=(args.muscle_window_start, args.muscle_window_end),
                threshold_factor=args.threshold_factor,
                n_components=args.n_components,
                verbose=True
            )
        print("\nRunning second ICA...")
        processor.run_second_ica(method=args.second_ica_method, exclude_labels=["eye blink", "heart beat", "muscle artifact"])
        print("\nApplying baseline correction...")
        if args.apply_ssp:
            print("\nApplying SSP...")
            processor.apply_ssp(n_eeg=args.ssp_n_eeg)
        processor.apply_baseline_correction(baseline=(args.baseline_start, args.baseline_end))
        if args.apply_csd:
            print("\nApplying CSD transformation...")
            processor.apply_csd(lambda2=args.lambda2, stiffness=args.stiffness)
        print(f"\nDownsampling to {processor.ds_sfreq} Hz")
        processor.downsample()
        # Final quality check
        processor.plot_evoked_response(ylim={'eeg': [-5, 5]}, xlim=(-0.3, 0.3), title="Final Evoked Response")
        epochs = processor.epochs
        
        ## Add microstate analysis
        recording_id = f"session_{n}"
        ma.add_recording(epochs, recording_id)

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
        fig = pcist.plot_analysis(details)
        print(f"PCI: {value}")
        subject_pcist_values.append(value)
        fig.savefig(f"pcist_session_{n}.png")  # Save the plot as PNG
        plt.close(fig)  # Close the figure to free up memory

    print("Clustering all sessions for microstate analysis")
    # Perform global clustering across all recordings
    results = ma.perform_global_clustering(
        n_clusters=args.n_clusters,
        n_resamples=args.n_resamples,
        n_samples=args.n_samples,
        min_peak_distance=args.min_peak_distance
    )
    
    # Plot global cluster results
    ma.plot_final_clusters(results)

    # Process individual recordings
    for n in range(len(rec.sessions)):
        recording_id = f"session_{n}"
        
        # Plot segmentation for this recording
        fig = ma.plot_recording_segmentation(recording_id, show=False)
        fig.savefig(f"segmentation_session_{n}.png")  # Save the plot as PNG
        plt.close(fig)  # Close the figure
        
        # Compare pre/post TMS for this recording
        comparison = ma.compare_pre_post_tms(
            recording_id,
            pre_window=(args.pre_window_start, args.pre_window_end),
            post_window=(args.post_window_start, args.post_window_end),
            return_details=True
        )
        
        # Plot comparison for this recording
        fig = ma.plot_pre_post_comparison(recording_id, comparison)
        fig.savefig(f"pre_post_comparison_session_{n}.png")  # Save the plot as PNG
        plt.close(fig)  # Close the figure
    
    return subject_pcist_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process EEG data.')
    parser.add_argument('--data_dir', type=str, default=str(Path.cwd() / 'data'), 
                        help='Path to the data directory (default: ./data)')
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
    parser.add_argument('--tms_muscle_thresh', type=float, default=2.0,
                        help='Threshold for TMS muscle artifact (default: 2.0)')
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
    parser.add_argument('--second_ica_method', type=str, default='infomax',
                        help='Second ICA method (default: infomax)')
    parser.add_argument('--apply_ssp', action='store_true',
                    help='Apply SSP (default: True)')
    parser.add_argument('--ssp_n_eeg', type=int, default=2,
                        help='Number of EEG components for SSP (default: 2)')
    parser.add_argument('--apply_csd', action='store_true',
                    help='Apply CSD transformation (default: False)')
    parser.add_argument('--lambda2', type=float, default=1e-5,
                    help='Lambda2 parameter for CSD transformation (default: 1e-5)')
    parser.add_argument('--stiffness', type=int, default=4,
                    help='Stiffness parameter for CSD transformation (default: 4)')
    parser.add_argument('--baseline_start', type=float, default=-0.4,
                        help='Start time for baseline correction (default: -0.4)')
    parser.add_argument('--baseline_end', type=float, default=-0.005,
                        help='End time for baseline correction (default: -0.005)')
    parser.add_argument('--response_start', type=int, default=0,
                        help='Start of the response window in ms (default: 0)')
    parser.add_argument('--response_end', type=int, default=299,
                        help='End of the response window in ms (default: 299)')
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
    parser.add_argument('--n_clusters', type=int, default=4,
                        help='Number of clusters for microstate analysis (default: 4)')
    parser.add_argument('--n_resamples', type=int, default=20,
                        help='Number of resamples for microstate analysis (default: 20)')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples for microstate analysis (default: 1000)')
    parser.add_argument('--min_peak_distance', type=int, default=1,
                        help='Minimum peak distance for microstate analysis (default: 1)')
    args = parser.parse_args()

    pcists = process_subjects(args)
    print(f"PCIst values: {pcists}")