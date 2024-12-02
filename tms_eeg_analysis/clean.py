from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import tensorly as tl
from tensorly.decomposition import parafac, non_negative_parafac
from tensorly.decomposition import tucker
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import mne
from typing import Dict, Tuple, Optional
import warnings
from tqdm import tqdm

class TMSArtifactCleaner:
    def __init__(self, epochs: mne.Epochs, verbose: bool = True):
        """Initialize the TMS artifact cleaner."""
        self.epochs = epochs.copy()
        self.data = epochs.get_data()
        self.times = epochs.times
        self.n_epochs, self.n_channels, self.n_times = self.data.shape
        self.verbose = verbose
        self.sfreq = epochs.info['sfreq']
        
        # Initialize tensor backend
        tl.set_backend('numpy')
        
        # Normalize the data
        self.normalized_data, self.scalers = self._normalize_data(self.data)
        
        # Initialize artifact info dictionary
        self.artifact_info = {'muscle': {'times': [], 'stats': {}}}

    def _normalize_data(self, data):
        """Normalize data across channels and time points."""
        normalized_data = np.zeros_like(data)
        scalers = []
        
        # Normalize each channel separately
        for ch in range(self.n_channels):
            scaler = StandardScaler()
            ch_data = data[:, ch, :].reshape(-1, 1)
            normalized_data[:, ch, :] = scaler.fit_transform(ch_data).reshape(self.n_epochs, -1)
            scalers.append(scaler)
            
        return normalized_data, scalers

    def _detect_single_epoch(self, epoch_idx: int, 
                        start_idx: int, 
                        end_idx: int,
                        threshold_factor: float) -> Tuple[Optional[float], Optional[float]]:
        """Detect muscle artifacts in a single epoch using tensor decomposition."""
        try:
            # Extract time window of interest
            epoch_data = self.normalized_data[epoch_idx:epoch_idx+1, :, start_idx:end_idx]
            
            # Apply non-negative tensor factorization
            weights, factors = non_negative_parafac(
                epoch_data,
                rank=3,
                n_iter_max=100,
                init='random',
                random_state=42,
                tol=1e-6,
                return_errors=False
            )
            
            # The factors list contains [temporal factors, spatial factors]
            # No need to access index 2 anymore
            temporal_factors = factors[1]  # Time factors
            spatial_factors = factors[0]   # Channel factors
            
            # Compute combined artifact score
            artifact_score = np.max(np.abs(temporal_factors)) * np.max(np.abs(spatial_factors))
            
            # Detect artifact if score exceeds threshold
            if artifact_score > threshold_factor:
                return (self.times[start_idx], self.times[end_idx])
            
            return (None, None)
            
        except Exception as e:
            warnings.warn(f"Error processing epoch {epoch_idx}: {str(e)}")
            return (None, None)

    def detect_muscle_artifacts(self,
                              muscle_window: Tuple[float, float] = (0.005, 0.05),
                              threshold_factor: float = 5.0,
                              n_jobs: int = -1,
                              verbose: Optional[bool] = None) -> Dict:
        """Detect TMS-evoked muscle artifacts using parallel processing."""
        verbose = self.verbose if verbose is None else verbose
        
        # Convert time window to sample indices
        start_idx = np.searchsorted(self.times, muscle_window[0])
        end_idx = np.searchsorted(self.times, muscle_window[1])
        
        # Reset artifact info
        self.artifact_info = {'muscle': {'times': [], 'stats': {}}}
        
        # Determine number of workers
        n_jobs = min(self.n_epochs, (os.cpu_count() or 1)) if n_jobs == -1 else n_jobs
        
        # Process epochs in parallel with progress bar
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for epoch_idx in range(self.n_epochs):
                future = executor.submit(
                    self._detect_single_epoch,
                    epoch_idx,
                    start_idx,
                    end_idx,
                    threshold_factor
                )
                futures.append(future)
            
            # Collect results with progress bar
            if verbose:
                futures = tqdm(futures, total=self.n_epochs, desc="Detecting artifacts")
            
            for future in futures:
                self.artifact_info['muscle']['times'].append(future.result())
        
        # Calculate statistics
        n_detected = sum(1 for x in self.artifact_info['muscle']['times'] if x[0] is not None)
        detection_rate = n_detected / self.n_epochs
        
        self.artifact_info['muscle']['stats'].update({
            'n_detected': n_detected,
            'detection_rate': detection_rate,
            'threshold_used': threshold_factor
        })
        
        if verbose:
            print(f"\nDetected {n_detected}/{self.n_epochs} artifacts ({detection_rate*100:.1f}%)")
            
        return self.artifact_info

   
    def clean_muscle_artifacts(self,
                            n_components: int = 2,
                            verbose: Optional[bool] = None) -> mne.Epochs:
        """Clean detected muscle artifacts using tucker decomposition."""
        verbose = self.verbose if verbose is None else verbose
        cleaned_data = self.normalized_data.copy()
        
        if verbose:
            print("\nCleaning detected artifacts...")
            
        print(f"Input data shape: {self.normalized_data.shape}")
        
        for epoch_idx, (start_time, end_time) in enumerate(tqdm(
            self.artifact_info['muscle']['times'],
            disable=not verbose
        )):
            if start_time is None or end_time is None:
                continue
                
            try:
                # Get time indices
                start_idx = np.searchsorted(self.times, start_time)
                end_idx = np.searchsorted(self.times, end_time)
                
                # Extract artifact data 
                artifact_data = self.normalized_data[epoch_idx:epoch_idx+1, :, start_idx:end_idx]
                
                if artifact_data.size == 0:
                    warnings.warn(f"Empty artifact data for epoch {epoch_idx}, skipping")
                    continue
                    
                #print(f"\nEpoch {epoch_idx} artifact data shape: {artifact_data.shape}")
                
                # Apply tucker decomposition 
                ranks = [1, n_components, n_components]
                core, factors = tucker(artifact_data, rank=ranks, init='random', tol=1e-6)
                
                #print(f"Core shape: {core.shape}")
                #print(f"Factor shapes: {[f.shape for f in factors]}")
                
                # Reconstruct cleaned data
                cleaned_artifact = tl.tucker_to_tensor((core, factors))
                
                # Store cleaned data
                cleaned_data[epoch_idx:epoch_idx+1, :, start_idx:end_idx] = cleaned_artifact
                
            except Exception as e:
                warnings.warn(f"Error cleaning epoch {epoch_idx}: {str(e)}")
                continue

        # Convert back to original scale
        cleaned_data = np.array([
            scaler.inverse_transform(cleaned_data[:, i, :].reshape(-1, 1)).reshape(self.n_epochs, -1)
            for i, scaler in enumerate(self.scalers)
        ]).transpose(1, 0, 2)
        
        # Create new epochs object with cleaned data
        epochs_clean = self.epochs.copy()
        epochs_clean._data = cleaned_data
        
        # Validate results
        if self.validate_cleaning(epochs_clean):
            if verbose:
                print("\nArtifact cleaning completed successfully")
        else:
            warnings.warn("Cleaning validation failed")
            
        return epochs_clean


    def find_optimal_threshold(self,
                         muscle_window: Tuple[float, float] = (0.005, 0.05),
                         target_detection_rate: float = 0.5,
                         initial_threshold: float = 0.9,
                         max_iter: int = 100,
                         tol: float = 0.01,
                         verbose: Optional[bool] = None) -> float:
        """Find optimal threshold using binary search."""
        verbose = self.verbose if verbose is None else verbose
        
        if verbose:
            print("\nFinding optimal threshold...")
        
        # Initialize binary search
        left = 0.1  # Minimum threshold > 0
        right = initial_threshold
        best_threshold = initial_threshold
        best_cost = float('inf')
        
        for iteration in range(max_iter):
            threshold = (left + right) / 2
            
            # Test current threshold
            self.detect_muscle_artifacts(
                muscle_window=muscle_window,
                threshold_factor=threshold,
                verbose=False
            )
            detection_rate = self.artifact_info['muscle']['stats']['detection_rate']
            cost = abs(detection_rate - target_detection_rate)
            
            if verbose:
                print(f"Iteration {iteration + 1}: "
                    f"threshold={threshold:.4f}, "
                    f"detection_rate={detection_rate:.4f}, "
                    f"cost={cost:.4f}")
                
            # Update best threshold if improved
            if cost < best_cost:
                best_cost = cost
                best_threshold = threshold
                
            # Check convergence
            if cost < tol:
                break
                
            # Update search interval
            if detection_rate > target_detection_rate:
                left = threshold
            else:
                right = threshold
                
            # Stop if interval is too small
            if abs(right - left) < 1e-6:
                break
        
        if verbose:
            print(f"\nBest threshold found: {best_threshold:.4f}")
            
        return best_threshold
    
    def validate_cleaning(self, cleaned_epochs: mne.Epochs) -> bool:
        """Validate the cleaning results."""
        if cleaned_epochs.get_data().shape != self.epochs.get_data().shape:
            return False
        if np.any(np.isnan(cleaned_epochs.get_data())):
            return False
        if np.any(np.isinf(cleaned_epochs.get_data())):
            return False
            
        return True