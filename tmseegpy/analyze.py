# analyze.py
# Old class that I was using when I used freesurfer and MNE-Python to try and compute the PCIlz from the original Casali paper from 2013
import numpy as np
from scipy import stats, signal
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.linalg import svd
from scipy.spatial.distance import pdist, squareform


from typing import Optional, Union, Dict, List, Tuple, Any
import warnings   
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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
from mne.simulation import simulate_evoked, simulate_raw
from mne.viz import plot_alignment, plot_bem


class PCIlz:
    """
    A class for analyzing preprocessed TMS-EEG data.
    
    This class implements various analysis methods including source modeling,
    connectivity analysis, and perturbational complexity index calculation.
    
    Parameters
    ----------
    preprocessed_epochs : mne.Epochs
        The preprocessed TMS-EEG epochs
    subjects_dir : str
        FreeSurfer subjects directory
    subject : str
        Subject name in FreeSurfer directory
        
    Attributes
    ----------
    epochs : mne.Epochs
        The preprocessed epochs
    evoked : mne.Evoked
        The averaged evoked response
    stc : mne.SourceEstimate
        The source estimates (after source modeling)
    """
    
    def __init__(self, 
                 preprocessed_epochs: mne.Epochs,
                 subjects_dir: str,
                 subject: str,
                 trans_path: str):
        
        self.epochs = preprocessed_epochs
        self.evoked = preprocessed_epochs.average()
        self.evoked_sim = None
        self.subjects_dir = subjects_dir
        self.subject = subject
        self.trans_path = trans_path
        
        # Initialize attributes that will be set later
        self.stc = None
        self.forward = None
        self.inverse_operator = None
        self.source_space = None
        self.bem_solution = None
        self.noise_cov = None
        
        # Ensure average reference
        self.evoked.set_eeg_reference('average', projection=True)

    def load_source_space(self, fpath):
       print('Loading source space.. If it does not work there might be file an issue with how paths are passed as strings')
       self.source_space = mne.read_source_spaces(fpath)

    def load_bem_solution(self, fpath):
        print('Loading bem solution.. If it does not work there might be file an issue with how paths are passed as strings')
        self.bem_solution = mne.read_bem_solution(fpath)

    def setup_source_space(self, spacing: str = 'oct6', add_dist: bool = False) -> None:
        """
        Setup the source space for source modeling.
        
        Parameters
        ----------
        spacing : str
            The spacing to use for the source space
        add_dist : bool
            Whether to add distances to the source space
        """
        try:
            self.source_space = mne.setup_source_space(
                subject=self.subject,
                spacing=spacing,
                subjects_dir=self.subjects_dir,
                add_dist=add_dist,
                n_jobs=1
            )
            print("Source space setup complete")
        except Exception as e:
            print(f"Error setting up source space: {str(e)}")
            
    def setup_bem_model(self, ico: Optional[int] = None, 
                       conductivity: Tuple[float, float, float] = (0.3, 0.006, 0.3)) -> None:
        """
        Setup the BEM model for forward modeling.
        
        Parameters
        ----------
        ico : int or None
            The ico parameter for BEM model
        conductivity : tuple
            Conductivity values for the three layers
        """
        try:
            model = mne.make_bem_model(
                subject=self.subject,
                subjects_dir=self.subjects_dir,
                ico=ico,
                conductivity=conductivity
            )
            self.bem_solution = mne.make_bem_solution(model)
            print("BEM model setup complete")
        except Exception as e:
            print(f"Error setting up BEM model: {str(e)}")

    def compute_forward_solution(self, mindist: float = 5.0) -> None:
        """
        Compute the forward solution.
        
        Parameters
        ----------
        mindist : float
            Minimum distance between sources
        """
        if self.source_space is None or self.bem_solution is None:
            raise ValueError("Must setup source space and BEM model first")
            
        try:
            # Get trans file path
            trans_path = self.trans_path
            trans = mne.read_trans(trans_path)
            
            self.forward = mne.make_forward_solution(
                self.evoked.info,
                trans=trans,
                src=self.source_space,
                bem=self.bem_solution,
                mindist=mindist,
                eeg=True,
                meg=False,
                n_jobs=-1
            )
            print("Forward solution computation complete")
        except Exception as e:
            print(f"Error computing forward solution: {str(e)}")

    def compute_inverse_operator(self, loose: float = 1.0, depth: float = 0.8, 
                               fixed: bool = False) -> None:
        """
        Compute the inverse operator.
        
        Parameters
        ----------
        loose : float
            Loose orientation constraint
        depth : float
            Depth weighting
        fixed : bool
            Whether to use fixed orientation
        """
        if self.forward is None:
            raise ValueError("Must compute forward solution first")
            
        # Compute noise covariance from baseline
        self.noise_cov = mne.compute_covariance(
            self.epochs,
            tmin=-0.5,
            tmax=-0.001,
            method='empirical',
            rank='info'
        )
        
        self.inverse_operator = mne.minimum_norm.make_inverse_operator(
            self.evoked.info,
            self.forward,
            self.noise_cov,
            loose=loose,
            depth=depth,
            fixed=fixed
        )
        print("Inverse operator computation complete")

    def compute_source_estimate(self, evoked=True, evoked_sim=False, lambda2: float = 1.0/9.0, 
                              method: str = 'MNE') -> None:
        """
        Compute the source estimate.
        
        Parameters
        ----------
        lambda2 : float
            Regularization parameter
        method : str
            Inverse method to use
        """
        if evoked:
            if self.inverse_operator is None:
                raise ValueError("Must compute inverse operator first")
                
            self.stc = mne.minimum_norm.apply_inverse(
                self.evoked,
                self.inverse_operator,
                lambda2=lambda2,
                method=method,
                pick_ori=None
            )
            print("Source estimate computation complete")

        elif evoked_sim:
            if self.inverse_operator is None:
                raise ValueError("Must compute inverse operator first")
                
            self.stc = mne.minimum_norm.apply_inverse(
                self.evoked_sim,
                self.inverse_operator,
                lambda2=lambda2,
                method=method,
                pick_ori=None
            )
            print("Source estimate computation complete using simulated evoked object")



    def detect_significant_sources(self, n_bootstraps: int = 500, 
                                alpha: float = 0.01,
                                baseline: Tuple[float, float] = (-0.5, -0.001)) -> np.ndarray:
        """
        Detect significant TMS-evoked cortical activation following Casali et al. 2013.
        """
        if self.inverse_operator is None:
            raise ValueError("Must compute inverse operator first")
            
        # Get source estimates for all trials
        stcs = []
        for trial in self.epochs:
            stc = mne.minimum_norm.apply_inverse(trial, self.inverse_operator,
                                                lambda2=1.0/9.0, method='MNE')
            stcs.append(stc.data)
        source_data = np.stack(stcs)  # Shape: (trials, sources, times)
        
        # Get baseline data
        baseline_mask = ((stc.times >= baseline[0]) & (stc.times <= baseline[1]))
        baseline_data = source_data[:, :, baseline_mask]
        
        # Calculate baseline mean and std for each source
        baseline_mean = np.mean(np.mean(baseline_data, axis=0), axis=1)[:, np.newaxis]
        baseline_std = np.std(np.mean(baseline_data, axis=0), axis=1)[:, np.newaxis]
        
        # Normalize source data
        normalized_data = ((np.mean(source_data, axis=0) - baseline_mean) / baseline_std)
        
        # Run bootstrap procedure
        max_abs_bootstraps = []
        for _ in range(n_bootstraps):
            # Shuffle baseline data at single-trial level for each source
            surrogate_data = np.stack([np.stack([np.random.permutation(bd) for bd in baseline_data[:, s]]) 
                                    for s in range(baseline_data.shape[1])])
            
            # Calculate mean and normalize
            surrogate_mean = np.mean(surrogate_data, axis=0)
            normalized_surrogate = (surrogate_mean - baseline_mean) / baseline_std
            
            # Store maximum absolute value across sources and times
            max_abs_bootstraps.append(np.max(np.abs(normalized_surrogate)))
            
        # Calculate significance threshold
        threshold = np.percentile(max_abs_bootstraps, (1 - alpha) * 100)
        
        # Create binary significance matrix
        SS = (np.abs(normalized_data) > threshold).astype(int)
        
        return SS

    
    def calculate_pci(self, SS: np.ndarray, min_duration_ms: int = 10) -> float:
        """
        Calculate Perturbational Complexity Index from significant sources matrix.
        
        Parameters
        ----------
        SS : np.ndarray
            Binary matrix of significant sources
        min_duration_ms : int
            Minimum duration for temporal compression
            
        Returns
        -------
        float
            PCI value
        """
        # Check data validity
        p_1 = np.mean(SS)
        if p_1 < 0.01 or p_1 > 0.99:
            print(f"Warning: Proportion of significant values ({p_1:.3f}) outside valid range")
            return 0
        print("Calculating source entropy.. ")   
        # Calculate source entropy
        H_L = -p_1 * np.log2(p_1) - (1-p_1) * np.log2(1-p_1)
        if H_L < 0.08:
            print(f"Warning: Initial entropy ({H_L:.3f}) too low")
            return 0
        print(f"Source entropy: {H_L}")    
        # Convert to binary sequence and calculate complexity
        binary_sequence = SS.flatten(order='F')
        c_L = self._custom_lempel_ziv_complexity(binary_sequence)
        print("Calculating PCI..")
        # Calculate final PCI
        L = len(binary_sequence)
        pci = (c_L * np.log2(L)) / (L * H_L)
        print(f"PCI: {pci}")
        return pci
        
    @staticmethod
    def _custom_lempel_ziv_complexity(binary_sequence: np.ndarray) -> int:
        """
        Calculate Lempel-Ziv complexity of a binary sequence.
        
        Parameters
        ----------
        binary_sequence : np.ndarray
            Binary sequence to analyze
            
        Returns
        -------
        int
            Lempel-Ziv complexity value
        """
        binary_str = ''.join(binary_sequence.astype(str))
        n = len(binary_str)
        
        complexity = 1
        current_position = 0
        current_length = 1
        patterns = {binary_str[0]: True}
        
        while current_position + current_length <= n:
            pattern = binary_str[current_position:current_position + current_length]
            
            if pattern in patterns:
                current_length += 1
            else:
                patterns[pattern] = True
                complexity += 1
                current_position += current_length
                current_length = 1
                
                if current_position == n:
                    break
                    
        return complexity

    def plot_source_activations(self, timepoints: List[float], 
                              save_path: Optional[str] = None) -> None:
        """
        Plot source activations at specific timepoints.
        
        Parameters
        ----------
        timepoints : list
            List of timepoints (in seconds) to plot
        save_path : str, optional
            Path to save the plots
        """
        if self.stc is None:
            raise ValueError("Must compute source estimate first")
            
        # Find closest times in data
        time_indices = [np.abs(self.stc.times - tp).argmin() for tp in timepoints]
        actual_times = [self.stc.times[idx] for idx in time_indices]
        
        # Visualization parameters
        surfer_kwargs = dict(
            subject=self.subject,
            hemi='split',
            subjects_dir=self.subjects_dir,
            views=['lat', 'med'],
            clim=dict(kind='percent', pos_lims=[80, 90, 95]),
            colormap='mne',
            background='white',
            time_viewer=False,
            smoothing_steps=10,
            size=(1000, 400)
        )
        
        # Plot each timepoint
        brains = []
        for actual_time in actual_times:
            brain = self.stc.plot(
                initial_time=actual_time,
                **surfer_kwargs
            )
            brain.add_text(0.1, 0.9, f'Time: {actual_time*1000:.0f} ms',
                         font_size=14)
            brains.append(brain)
            
            if save_path:
                brain.save_image(f"{save_path}_time_{int(actual_time*1000)}ms.png")
                
        return brains
    
    def simulate_eeg(self, evoked=False, simple=False, nave=30, 
                    iir_filter=None, random_state=None, verbose=True,
                    pick_ori='normal'):
        """
        Simulate EEG data using either raw or evoked simulation.
        
        Parameters
        ----------
        evoked : bool
            If True, simulates evoked data. If False, simulates raw data.
        simple : bool
            Whether to only use epochs.info for simulation or to use the source estimate 
            and forward solution
        nave : int
            Number of averaged epochs (only for evoked simulation)
        iir_filter : None | array
            IIR filter coefficients (denominator) e.g. [1, -1, 0.2] (only for evoked simulation)
        random_state : None | int | instance of RandomState
            Random seed for reproducibility
        verbose : bool
            Controls the logging of mne simulation functions
        pick_ori : str
            The source orientation to use. Options:
            - 'normal' : Orient sources normal to cortical surface
            - None : Use magnitude of source currents
            - 'vector' : Keep vectorial source directions

        Returns
        -------
        raw_sim : Raw object or Evoked object
            The simulated EEG data
        """
        info = self.epochs.info
        
        # If noise covariance isn't computed yet, compute it
        if self.noise_cov is None:
            print("Computing noise covariance from baseline period...")
            self.noise_cov = mne.compute_covariance(
                self.epochs,
                tmin=-0.5,
                tmax=-0.001,
                method='empirical',
                rank='info'
            )
        
        if evoked:
            if not simple:
                    try:
                        self.evoked_sim = simulate_evoked(
                            fwd=self.forward,
                            stc=self.stc,
                            info=info,
                            cov=self.noise_cov,  # Using class noise covariance
                            nave=nave,
                            iir_filter=iir_filter,
                            random_state=random_state,
                            use_cps=True,
                            verbose=verbose
                        )
                        
                        # Add projectors back if needed
                        if len(info['projs']) > 0:
                            self.evoked_sim.add_proj(info['projs'])
                            
                        return self.evoked_sim
                        
                    except Exception as e:
                        print(f"Error during evoked simulation: {str(e)}")
                        return None
            else:
                print("Simple simulation not available for evoked data. Please compute forward solution and source estimate.")
                return None
        else:
            if not simple:
                    try:
                        raw_sim = simulate_raw(
                            info=info, 
                            stc=self.stc, 
                            forward=self.forward, 
                            verbose=verbose
                        )
                        return raw_sim
                    except Exception as e:
                        print(f"Error during raw simulation: {str(e)}")
                        return None
            else:
                raw_sim = simulate_raw(info=info, verbose=verbose)
                return raw_sim



    def plot_tms_eeg_analysis(self, SS: np.ndarray, pci: float, 
                             title: Optional[str] = None,
                             save_path: Optional[str] = None,
                             color_palette: Optional[List[str]] = None,
                            evoked_ylim: List[float] = [-7, 7],
                            evoked_xlim: Optional[Tuple[float, float]] = (-0.1, 0.5)) -> None:
        
        """
        Create a visualization of TMS-EEG analysis results.
        
        Parameters
        ----------
        SS : np.ndarray
            Binary matrix of significant sources
        pci : float
            Calculated PCI value
        title : str, optional
            Main title for the entire figure
        save_path : str, optional
            Path to save the figure
        color_palette : list of str, optional
            Custom color palette for plots. If None, uses default seaborn deep palette
        evoked_ylim : list of float
            Y-axis limits for evoked plot [lower, upper]
        evoked_xlim : tuple of float, optional
            X-axis limits for evoked plot in seconds (start_time, end_time)
        """
        # Set color parameters
        if color_palette is None:
            colors = sns.color_palette("deep")
        else:
            colors = color_palette
            
        # Create figure
        fig = plt.figure(figsize=(16, 14))
        fig.patch.set_facecolor('white')
        gs = plt.GridSpec(4, 2, height_ratios=[1, 1, 1, 0.8])
        
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Global Field Power
        ax_gfp = fig.add_subplot(gs[0, :])
        ax_gfp.set_facecolor('#f8f8f8')
        ax_gfp.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        gfp = np.std(self.stc.data, axis=0)
        times = self.stc.times * 1000  # Convert to ms
        
        # Plot GFP with confidence interval and styling
        ax_gfp.plot(times, gfp, color=colors[0], linewidth=2, label='GFP')
        ax_gfp.fill_between(times, 
                        gfp-stats.sem(self.stc.data, axis=0),
                        gfp+stats.sem(self.stc.data, axis=0),
                        color=colors[0], alpha=0.2)
        
        # Add component markers with enhanced styling
        components = {
            'P30': (30, colors[1]), 
            'N45': (45, colors[2]), 
            'P55': (55, colors[3]),
            'N100': (100, colors[4]), 
            'P180': (180, colors[5])
        }
        
        max_gfp = np.max(gfp)
        for comp, (t, color) in components.items():
            if t > times[0] and t < times[-1]:
                ax_gfp.axvline(x=t, color=color, linestyle=':', alpha=0.7)
                ax_gfp.text(t, max_gfp*1.1, comp, rotation=90,
                        horizontalalignment='center', 
                        color=color, fontweight='bold')
        
        # Add TMS pulse marker
        ax_gfp.axvline(x=0, color='red', linestyle='--', linewidth=2, label='TMS pulse')
        
        ax_gfp.set_title('Global Field Power Over Time', fontsize=12, fontweight='bold', pad=15)
        ax_gfp.set_xlabel('Time (ms)', fontsize=10)
        ax_gfp.set_ylabel('Global Field Power (μV)', fontsize=10)
        ax_gfp.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # 2. Butterfly Plot with enhanced styling
        ax_butterfly = fig.add_subplot(gs[1, :])
        ax_butterfly.set_facecolor('#f8f8f8')
        ax_butterfly.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        self.evoked.plot(axes=ax_butterfly, show=False, spatial_colors=True,
                        ylim=dict(eeg=evoked_ylim), 
                        xlim=evoked_xlim,  # Changed this line
                        titles=dict(eeg='Averaged Evoked Response'))
        
        # Add component markers to butterfly plot
        max_val = max(evoked_ylim)
        for comp, (t, color) in components.items():
            if t > times[0] and t < times[-1]:
                ax_butterfly.axvline(x=t/1000, color=color, linestyle=':', alpha=0.7)
                ax_butterfly.text(t/1000, max_val*1.1, comp, rotation=90,
                                horizontalalignment='center', 
                                color=color, fontweight='bold')
        
        # 3. Significant Sources Heatmap with enhanced styling
        ax_ss = fig.add_subplot(gs[2, :])
        ax_ss.set_facecolor('#f8f8f8')
        ax_ss.grid(True, linestyle='--', alpha=0.3, color='gray')

        im = ax_ss.imshow(SS, aspect='auto', cmap='RdBu_r',
                        extent=[0, 300, 0, SS.shape[0]])

        # Add percentage of significant sources with styled twin axis
        ax_ss2 = ax_ss.twinx()
        sig_percent = np.mean(SS, axis=0) * 100

        # Fix: Properly mask the time points to match significant sources
        time_mask = (times >= 0) & (times <= 300)
        plot_times = times[time_mask]
        if len(plot_times) > len(sig_percent):
            plot_times = plot_times[:len(sig_percent)]
        elif len(plot_times) < len(sig_percent):
            sig_percent = sig_percent[:len(plot_times)]

        ax_ss2.plot(plot_times, sig_percent, 
                    color=colors[2], linewidth=2, alpha=0.7)
        ax_ss2.set_ylabel('% Significant Sources', color=colors[2], 
                        fontweight='bold', fontsize=10)
        ax_ss2.tick_params(axis='y', colors=colors[2])
        
        ax_ss.set_xlabel('Time (ms)', fontsize=10)
        ax_ss.set_ylabel('Source Index', fontsize=10)
        ax_ss.set_title('Significant Sources Over Time', 
                        fontsize=12, fontweight='bold', pad=15)
        
        # Add colorbar with styling
        divider = make_axes_locatable(ax_ss)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Significance')
        
        # 4. Analysis Statistics with enhanced styling
        ax_stats = fig.add_subplot(gs[3, 0])
        ax_stats.set_facecolor('#f8f8f8')
        ax_stats.axis('off')
        
        stats_text = f"""
        TMS-EEG Analysis Metrics:
        -------------------------
        
        PCI Value: {pci:.3f}
        Total Significant Sources: {np.sum(SS):,}
        Peak Time: {times[np.argmax(np.mean(SS, axis=0))]: .1f} ms
        Duration: {np.sum(np.any(SS, axis=0)) * (times[1]-times[0]): .1f} ms
        
        Coverage Statistics:
        ------------------
        Total Sources: {SS.shape[0]:,}
        Time Points: {SS.shape[1]:,}
        Significance Rate: {100 * np.sum(SS) / SS.size:.2f}%
        """
        ax_stats.text(0.05, 0.95, stats_text, 
                    fontsize=11, va='top', 
                    family='monospace',
                    bbox=dict(facecolor='white', 
                            alpha=0.8,
                            edgecolor='gray',
                            boxstyle='round,pad=1'))
        
        # 5. Source Distribution with enhanced styling
        ax_hist = fig.add_subplot(gs[3, 1])
        ax_hist.set_facecolor('#f8f8f8')
        ax_hist.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        peak_time_idx = np.argmax(np.std(self.stc.data, axis=0))
        peak_data = self.stc.data[:, peak_time_idx]
        
        sns.histplot(data=peak_data, ax=ax_hist, 
                    bins=50, color=colors[0], 
                    alpha=0.7, stat='density')
        
        ax_hist.set_xlabel('Source Amplitude', fontsize=10)
        ax_hist.set_ylabel('Density', fontsize=10)
        ax_hist.set_title(f'Source Distribution at Peak ({times[peak_time_idx]:.1f} ms)',
                        fontsize=12, fontweight='bold', pad=15)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    
