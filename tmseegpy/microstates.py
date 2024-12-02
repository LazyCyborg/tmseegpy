import numpy as np
from typing import Optional, Union, Dict, List, Tuple
import matplotlib.pyplot as plt
import mne
from pycrostates.preprocessing import extract_gfp_peaks, resample
from pycrostates.cluster import ModKMeans
from pycrostates.segmentation import EpochsSegmentation
from pycrostates.io import ChData
from pycrostates.viz import plot_cluster_centers

class Microstate:
    """
    Class implementation of Microstate Analysis for EEG epochs using pycrostates.
    Supports combined analysis across multiple recordings.
    """
    
    def __init__(self):
        self.epochs_list = []  # Store multiple recordings
        self.segmentations = {}  # Store segmentations for each recording
        self.global_clustering = None  # Store global clustering results
        
    def add_recording(self, epochs: mne.epochs.Epochs, recording_id: str):
        """Add a recording to the analysis."""
        if not isinstance(epochs, mne.epochs.Epochs):
            raise ValueError("Input must be an MNE Epochs object")
        self.epochs_list.append((recording_id, epochs))
        
    def perform_global_clustering(self, 
                                n_clusters: int = 5,
                                n_resamples: int = 3,
                                n_samples: int = 1000,
                                picks: str = 'eeg',
                                min_peak_distance: int = 1,
                                random_state: int = 42,
                                n_jobs: int = -1,
                                return_details: bool = True) -> Union[Dict, float]:
        """
        Performs microstate analysis across all recordings with GFP peak extraction and resampling.
        """
        if not self.epochs_list:
            raise ValueError("No recordings added. Use add_recording() first.")
            
        # Initialize storage for all recordings
        all_gfp_data = []
        all_resamples = []
        all_info = None
        
        # Process each recording
        for recording_id, epochs in self.epochs_list:
            # Extract GFP peaks
            gfp_data = extract_gfp_peaks(
                epochs,
                picks=picks,
                min_peak_distance=min_peak_distance,
                return_all=False
            )
            
            # Store channel info from first recording
            if all_info is None:
                all_info = gfp_data.info
                
            # Perform resampling
            recording_resamples = resample(
                gfp_data,
                n_resamples=n_resamples,
                n_samples=n_samples,
                random_state=random_state
            )
            
            all_gfp_data.append(gfp_data)
            all_resamples.extend(recording_resamples)
        
        # Initialize storage for clustering results
        resample_results = []
        gev_scores = []
        
        # Perform clustering on each resample
        for resamp in all_resamples:
            mod_kmeans = ModKMeans(
                n_clusters=n_clusters,
                random_state=random_state
            )
            mod_kmeans.fit(resamp, n_jobs=n_jobs, verbose=True)
            
            resample_results.append(mod_kmeans.cluster_centers_)
            gev_scores.append(mod_kmeans.GEV_)
        
        # Compute final clustering on concatenated centers
        all_resampling_results = np.vstack(resample_results).T
        all_resampling_results = ChData(all_resampling_results, all_info)
        
        final_clustering = ModKMeans(
            n_clusters=n_clusters,
            random_state=random_state
        )
        final_clustering.fit(
            all_resampling_results,
            n_jobs=n_jobs,
            verbose="WARNING"
        )
        
        # Store global clustering results
        self.global_clustering = final_clustering
        
        # Backfit to individual recordings
        for recording_id, epochs in self.epochs_list:
            self.segmentations[recording_id] = final_clustering.predict(
                epochs,
                picks=picks
            )
        
        if return_details:
            details = {
                'resample_results': resample_results,
                'gev_scores': gev_scores,
                'mean_gev': np.mean(gev_scores),
                'std_gev': np.std(gev_scores),
                'final_clusters': final_clustering.cluster_centers_,
                'final_gev': final_clustering.GEV_,
                'all_gfp_data': all_gfp_data,
                'resamples': all_resamples,
                'segmentations': self.segmentations,
                'picks': picks
            }
            return details
        
        return np.mean(gev_scores)
    
    def get_recording_segmentation(self, recording_id: str) -> Optional[EpochsSegmentation]:
        """Get segmentation for a specific recording."""
        return self.segmentations.get(recording_id)
    
    def compute_recording_parameters(self, 
                                   recording_id: str,
                                   norm_gfp: bool = True, 
                                   return_dist: bool = False) -> Dict:
        """Compute microstate parameters for a specific recording."""
        segmentation = self.get_recording_segmentation(recording_id)
        if segmentation is None:
            raise ValueError(f"No segmentation found for recording {recording_id}")
        return segmentation.compute_parameters(norm_gfp=norm_gfp, return_dist=return_dist)
    
    def compute_recording_transitions(self,
                                    recording_id: str,
                                    stat: str = 'probability',
                                    ignore_repetitions: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Compute transition matrices for a specific recording."""
        segmentation = self.get_recording_segmentation(recording_id)
        if segmentation is None:
            raise ValueError(f"No segmentation found for recording {recording_id}")
            
        observed = segmentation.compute_transition_matrix(
            stat=stat,
            ignore_repetitions=ignore_repetitions
        )
        
        expected = segmentation.compute_expected_transition_matrix(
            stat=stat,
            ignore_repetitions=ignore_repetitions
        )
        
        return observed, expected
    
    def plot_recording_segmentation(self, 
                                  recording_id: str,
                                  cmap=None, 
                                  axes=None, 
                                  block=False, 
                                  show=None) -> plt.Figure:
        """Plot segmentation for a specific recording."""
        segmentation = self.get_recording_segmentation(recording_id)
        if segmentation is None:
            raise ValueError(f"No segmentation found for recording {recording_id}")
            
        if axes is None:
            fig, axes = plt.subplots(figsize=(12, 6))
        else:
            fig = axes.figure
            
        labels = segmentation.labels
        
        if cmap is None:
            colors = ['navy', 'darkturquoise', 'forestgreen', 'gold', 'darkorange']
            cmap = plt.cm.colors.ListedColormap(colors[:labels.max()+1])
        
        im = axes.imshow(labels.T, aspect='auto', cmap=cmap, interpolation='nearest')
        
        axes.set_title(f'Microstate Segmentation - Recording {recording_id}', fontsize=12, pad=10)
        axes.set_xlabel('Epochs', fontsize=10)
        axes.set_ylabel('Time Points', fontsize=10)
        
        axes.set_xticks([])
        axes.grid(False)
        
        cbar = plt.colorbar(im, ax=axes)
        cbar.set_label('Microstate', fontsize=10)
        
        n_states = labels.max() + 1
        cbar.set_ticks(np.arange(n_states))
        cbar.set_ticklabels([f'State {i}' for i in range(n_states)])
        
        text = (
            f"Recording: {recording_id}\n"
            "Colors indicate different brain states\n"
            f"Analysis identified {n_states} distinct states"
        )
        plt.figtext(0.02, 0.02, text, fontsize=8, style='italic',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, pad=5))
        
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return fig
    
    def plot_final_clusters(self, details: Dict) -> plt.Figure:
        """Plot the final clustering results."""
        if self.global_clustering is None:
            raise ValueError("Must run perform_global_clustering first")
            
        # Create figure and axes
        n_clusters = details['final_clusters'].shape[0]
        f, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 3))
        
        # Use info from first recording's GFP data
        info = details['all_gfp_data'][0].info
        
        # Plot cluster centers
        return plot_cluster_centers(
            cluster_centers=details['final_clusters'],
            info=info,
            axes=axes
        )
    
    def compare_pre_post_tms(self,
                            recording_id: str,
                            pre_window: Tuple[float, float] = (-500, -10),
                            post_window: Tuple[float, float] = (10, 500),
                            ignore_repetitions: bool = False,
                            log_base: Union[float, str] = 2,
                            return_details: bool = True) -> Union[Dict, float]:
        """Compare microstate characteristics before and after TMS pulse for a specific recording."""
        segmentation = self.get_recording_segmentation(recording_id)
        if segmentation is None:
            raise ValueError(f"No segmentation found for recording {recording_id}")
            
        # Get recording data
        _, epochs = next((rec for rec in self.epochs_list if rec[0] == recording_id), (None, None))
        if epochs is None:
            raise ValueError(f"Recording {recording_id} not found")
            
        # Get original times and labels
        orig_times = epochs.times * 1000
        orig_labels = segmentation.labels
        
        # Find indices for pre and post windows
        pre_start_idx = np.searchsorted(orig_times, pre_window[0])
        pre_end_idx = np.searchsorted(orig_times, pre_window[1])
        post_start_idx = np.searchsorted(orig_times, post_window[0])
        post_end_idx = np.searchsorted(orig_times, post_window[1])
        
        # Extract times and labels for pre and post periods
        pre_times = orig_times[pre_start_idx:pre_end_idx]
        post_times = orig_times[post_start_idx:post_end_idx]
        
        pre_labels = orig_labels[:, pre_start_idx:pre_end_idx]
        post_labels = orig_labels[:, post_start_idx:post_end_idx]
        
        # Calculate entropy directly from labels
        def calc_entropy(labels, ignore_reps=False, base=2):
            # Flatten labels across epochs
            flat_labels = labels.reshape(-1)
            if ignore_reps:
                # Remove consecutive repeats
                mask = np.diff(flat_labels, prepend=flat_labels[0]-1) != 0
                flat_labels = flat_labels[mask]
            
            # Calculate probabilities
            unique, counts = np.unique(flat_labels, return_counts=True)
            probs = counts / len(flat_labels)
            
            # Calculate entropy
            return -np.sum(probs * np.log(probs) / np.log(base))
        
        pre_entropy = calc_entropy(pre_labels, ignore_reps=ignore_repetitions, base=log_base)
        post_entropy = calc_entropy(post_labels, ignore_reps=ignore_repetitions, base=log_base)
        
        # Calculate transition matrices
        def calc_transition_matrix(labels, ignore_reps=False):
            n_states = len(np.unique(labels))
            trans_mat = np.zeros((n_states, n_states))
            
            # Process each epoch separately
            for epoch_labels in labels:
                if ignore_reps:
                    mask = np.diff(epoch_labels, prepend=epoch_labels[0]-1) != 0
                    epoch_labels = epoch_labels[mask]
                
                # Count transitions
                for i in range(len(epoch_labels)-1):
                    from_state = epoch_labels[i]
                    to_state = epoch_labels[i+1]
                    trans_mat[from_state, to_state] += 1
            
            # Convert to probabilities
            row_sums = trans_mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            trans_mat = trans_mat / row_sums
            
            return trans_mat
        
        pre_trans = calc_transition_matrix(pre_labels, ignore_reps=ignore_repetitions)
        post_trans = calc_transition_matrix(post_labels, ignore_reps=ignore_repetitions)
        
        # Calculate basic statistics for each microstate
        def calc_state_stats(labels):
            n_states = len(np.unique(labels.reshape(-1)))
            stats = {}
            
            # Process each state
            for state in range(n_states):
                state_name = f"state_{state}"
                
                # Calculate overall occurrence rate
                mask = (labels == state)
                stats[f"{state_name}_occurrence"] = np.mean(mask)
                
                # Calculate mean duration for each epoch, then average
                durations = []
                for epoch_labels in labels:
                    epoch_mask = (epoch_labels == state)
                    if np.any(epoch_mask):
                        # Find runs of True values
                        runs = np.diff(np.concatenate(([False], epoch_mask, [False])).astype(int))
                        starts = np.where(runs == 1)[0]
                        ends = np.where(runs == -1)[0]
                        durations.extend(ends - starts)
                
                if durations:
                    stats[f"{state_name}_mean_duration"] = np.mean(durations)
                else:
                    stats[f"{state_name}_mean_duration"] = 0
                    
            return stats
        
        pre_stats = calc_state_stats(pre_labels)
        post_stats = calc_state_stats(post_labels)
        
        if return_details:
            details = {
                'pre_window': pre_window,
                'post_window': post_window,
                'pre_entropy': pre_entropy,
                'post_entropy': post_entropy,
                'entropy_difference': post_entropy - pre_entropy,
                'pre_transition_matrix': pre_trans,
                'post_transition_matrix': post_trans,
                'pre_statistics': pre_stats,
                'post_statistics': post_stats,
                'pre_labels': pre_labels,
                'post_labels': post_labels,
                'pre_times': pre_times,
                'post_times': post_times
            }
            return details
        
        return post_entropy - pre_entropy

    def plot_pre_post_comparison(self, recording_id: str, comparison_details: Dict) -> plt.Figure:
        """Plot comparison of pre and post TMS microstate characteristics for a specific recording."""
        # Create figure with adjusted size
        fig = plt.figure(figsize=(12, 8))
        gs = plt.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.4, wspace=0.3)
        
        # Update title to include recording ID
        title_suffix = f' - Recording {recording_id}'

            
        # Plot entropy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        entropy_bars = ax1.bar(['Pre-TMS', 'Post-TMS'], 
                            [comparison_details['pre_entropy'], 
                            comparison_details['post_entropy']],
                            color=['lightblue', 'darkblue'],
                            width=0.5)
        ax1.set_title('Microstate Entropy\n(Higher = More Complex/Random)', fontsize=10, pad=10)
        ax1.set_ylabel('Shannon Entropy (bits)')
        
        # Add value labels on bars with increased spacing
        for bar in entropy_bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # Add entropy change text
        change = comparison_details['entropy_difference']
        change_text = f"Change: {change:+.3f} bits"
        ax1.text(0.5, -0.2, change_text, ha='center', transform=ax1.transAxes,
                fontsize=9, style='italic')
        
        # Plot transition matrix
        ax2 = fig.add_subplot(gs[0, 1])
        pre_trans = comparison_details['pre_transition_matrix']
        im_pre = ax2.imshow(pre_trans, cmap='Blues', aspect='auto')
        ax2.set_title('Pre-TMS State Transitions\n(Darker = More Frequent)', fontsize=10, pad=10)
        cbar = plt.colorbar(im_pre, ax=ax2)
        cbar.set_label('Transition Probability', fontsize=8)
        ax2.set_xlabel('To State')
        ax2.set_ylabel('From State')
        
        # Statistics plot with simplified design
        ax3 = fig.add_subplot(gs[1, :])
        
        # Extract statistics
        pre_stats = comparison_details['pre_statistics']
        post_stats = comparison_details['post_statistics']
        
        # Get all unique state numbers
        all_states = sorted(list(set(int(key.split('_')[1]) for key in pre_stats.keys())))
        n_states = len(all_states)
        
        # Create grouped bar plot
        x = np.arange(n_states)
        width = 0.35
        
        # Plot occurrence rates
        pre_occurrence = [pre_stats.get(f'state_{state}_occurrence', 0) for state in all_states]
        post_occurrence = [post_stats.get(f'state_{state}_occurrence', 0) for state in all_states]
        
        bars1 = ax3.bar(x - width/2, pre_occurrence, width, label='Pre-TMS',
                    color='lightblue', alpha=0.7)
        bars2 = ax3.bar(x + width/2, post_occurrence, width, label='Post-TMS',
                    color='darkblue', alpha=0.7)
        
        # Customize the plot
        ax3.set_ylabel('Occurrence Rate\n(proportion of time)', fontsize=9)
        ax3.set_title('Microstate Occurrence Rates\n(How often each state appears)', fontsize=10, pad=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'State {i}' for i in all_states])
        ax3.legend(loc='upper right')
        ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels with reduced fontsize
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        
        # Add detailed explanatory text
        explanation = (
            "Microstate Analysis Results:\n"
            "• Entropy: Measures the randomness/complexity of state sequences\n"
            "• Transitions: Shows how likely states change from one to another\n"
            "• Occurrence Rate: Shows the proportion of time each state is active\n"
            f"Analysis identified {n_states} distinct brain states"
        )
        
        fig.text(0.5, 0.02, explanation, ha='center', va='center',
                fontsize=8, style='italic',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, pad=5))
        
        plt.tight_layout()
        # Adjust layout to make room for text
        plt.subplots_adjust(bottom=0.2)
        return fig