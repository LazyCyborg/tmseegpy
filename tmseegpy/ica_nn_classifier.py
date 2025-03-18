# ica_classifier.py
import numpy as np
import os
import pickle
from pathlib import Path
import tensorflow as tf
from scipy import signal
import mne

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



def plot_ica_classification(ica, inst, classification_results, figsize=(15, 10)):
    """
    Plot ICA components with their neural network classifications.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        The fitted ICA object
    inst : mne.io.Raw or mne.Epochs
        The data instance used to fit the ICA
    classification_results : dict
        Results from ICAComponentClassifier.classify_ica()
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Get component data
    sources = ica.get_sources(inst)
    if isinstance(sources, mne.Epochs):
        source_data = np.mean(sources.get_data(), axis=0)
    else:
        source_data = sources.get_data()

    # Get classifications
    classifications = classification_results['classifications']
    details = classification_results['details']
    exclude = classification_results['exclude']

    # Determine component indices to plot
    n_components = ica.n_components_

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Determine grid layout
    n_rows = int(np.ceil(n_components / 4))

    # Create subplots
    gs = GridSpec(n_rows, 4, figure=fig)

    # Plot each component
    for comp_idx in range(n_components):
        ax = fig.add_subplot(gs[comp_idx // 4, comp_idx % 4])

        # Get classification info
        if comp_idx in classifications:
            comp_class = classifications[comp_idx]
            prob = details[comp_idx]['probability']
            title = f"IC{comp_idx}: {comp_class} ({prob:.2f})"

            # Set color based on exclusion
            if comp_idx in exclude:
                color = 'red'
            else:
                color = 'black'
        else:
            title = f"IC{comp_idx}"
            color = 'black'

        # Plot time series
        ax.plot(source_data[comp_idx], color=color)
        ax.set_title(title, fontsize=10, color=color)
        ax.set_xticks([])

        # Add topography inset
        ax_topo = plt.axes([ax.get_position().x0 + 0.01, ax.get_position().y0 + 0.01,
                            0.05, 0.05])
        mne.viz.plot_topomap(ica.get_components()[:, comp_idx], ica.info, axes=ax_topo,
                             show=False)

    plt.tight_layout()
    return fig


def plot_classification_summary(classification_results, figsize=(10, 6)):
    """
    Plot summary of component classifications.

    Parameters
    ----------
    classification_results : dict
        Results from ICAComponentClassifier.classify_ica()
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Get classifications
    classifications = classification_results['classifications']

    # Count component types
    class_counts = {}
    for comp_idx, comp_class in classifications.items():
        if comp_class not in class_counts:
            class_counts[comp_class] = 0
        class_counts[comp_class] += 1

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bar chart
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    classes = [classes[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]

    bars = ax.bar(classes, counts)

    # Mark excluded component types
    exclude_classes = ['muscle', 'eye blink', 'eye movement', 'tms-pulse', 'tms-decay',
                       'tms-ringing', 'line noise']

    for i, c in enumerate(classes):
        if c in exclude_classes:
            bars[i].set_color('red')

    ax.set_title('ICA Component Classification Summary')
    ax.set_ylabel('Count')
    ax.set_xlabel('Component Type')
    ax.set_xticklabels(classes, rotation=45, ha='right')

    plt.tight_layout()
    return fig


class ICAComponentClassifier:
    """Neural network-based classifier for TMS-EEG ICA components."""

    def __init__(self, model_path=None, label_encoder_path=None, probability_threshold=0.3):
        self.probability_threshold = probability_threshold
        """Initialize the classifier with model and label encoder.

        Parameters
        ----------
        model_path : str
            Path to the saved Keras model
        label_encoder_path : str
            Path to the saved LabelEncoder pickle file
        """
        # Default paths within the package
        if model_path is None:
            model_path = Path(__file__).parent / 'models' / 'tms_eeg_classifier_spec_model.keras'

        if label_encoder_path is None:
            label_encoder_path = Path(__file__).parent / 'models' / 'label_encoder.pkl'

        # Load model and label encoder
        self.model = self._load_model(model_path)
        self.label_encoder = self._load_label_encoder(label_encoder_path)
        self.class_names = self.label_encoder.classes_ if self.label_encoder else None

    def _load_model(self, model_path):
        """Load the Keras model."""
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            return model
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {str(e)}")
            return None

    def _load_label_encoder(self, label_encoder_path):
        """Load the LabelEncoder."""
        try:
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            return label_encoder
        except Exception as e:
            print(f"Warning: Could not load label encoder from {label_encoder_path}: {str(e)}")
            return None

    def extract_topo_features(self, topo_map):
        """Extract statistical features from topography that work with any channel count"""
        features = [
            np.mean(topo_map),  # Mean value
            np.std(topo_map),  # Standard deviation
            np.median(topo_map),  # Median value
            np.max(topo_map),  # Maximum value
            np.min(topo_map),  # Minimum value
            np.percentile(topo_map, 25),  # 25th percentile
            np.percentile(topo_map, 75),  # 75th percentile
            np.sum(topo_map > 0) / len(topo_map),  # Proportion of positive values
            np.sum(topo_map < 0) / len(topo_map),  # Proportion of negative values
            np.mean(np.abs(topo_map)),  # Mean absolute deviation
            np.mean(np.abs(np.diff(topo_map))),  # Mean gradient (spatial smoothness)
            np.sum(topo_map ** 2)  # Power (squared magnitude)
        ]
        return np.array(features)

    def extract_spectral_features(self, time_series, fs=None):
        """Extract spectral features from time series"""
        # If sampling rate not provided, estimate based on array length
        if fs is None:
            # If original data was 10000 Hz with 20001 points
            # Adjust sampling rate proportionally for resampled data
            original_length = 20001
            original_fs = 10000
            fs = original_fs * (len(time_series) / original_length)

        # Compute power spectrum with Welch's method
        f, Pxx = signal.welch(time_series, fs=fs, nperseg=min(1024, len(time_series) // 2))

        # Helper function to safely compute mean of possibly empty slices
        def safe_mean(arr):
            return np.mean(arr) if len(arr) > 0 else 0.0

        # Extract band powers
        delta_mask = (f >= 0.5) & (f < 4)
        delta_power = safe_mean(Pxx[delta_mask])

        theta_mask = (f >= 4) & (f < 8)
        theta_power = safe_mean(Pxx[theta_mask])

        alpha_mask = (f >= 8) & (f < 13)
        alpha_power = safe_mean(Pxx[alpha_mask])

        beta_mask = (f >= 13) & (f < 30)
        beta_power = safe_mean(Pxx[beta_mask])

        gamma_mask = (f >= 30) & (f < 80)
        gamma_power = safe_mean(Pxx[gamma_mask])

        high_freq_mask = (f >= 80) & (f < 250)
        high_freq = safe_mean(Pxx[high_freq_mask])

        line_noise_mask = (f >= 45) & (f <= 55)
        line_noise = safe_mean(Pxx[line_noise_mask])

        # Spectral metrics
        spectral_edge = np.percentile(Pxx, 95) if len(Pxx) > 0 else 0.0

        # Mean frequency - handle empty/zero cases
        mean_freq = np.sum(f * Pxx) / max(np.sum(Pxx), 1e-10)

        # Check for sharp peaks (line noise)
        try:
            peak_info = signal.find_peaks(Pxx, height=np.mean(Pxx) * 3)[1]
            peak_heights = peak_info.get('peak_heights', [])
            has_peaks = len(peak_heights) > 0
        except:
            has_peaks = False

        # Small constant to avoid log(0)
        epsilon = 1e-10

        # Ensure all values are positive before taking log
        features = np.log10(np.array([
            delta_power + epsilon,
            theta_power + epsilon,
            alpha_power + epsilon,
            beta_power + epsilon,
            gamma_power + epsilon,
            high_freq + epsilon,
            line_noise + epsilon,
            spectral_edge + epsilon,
            mean_freq + epsilon,
            float(has_peaks) + epsilon
        ]))

        return features

    def prepare_component_data(self, component_time_series, component_topo):
        """Prepare component data for classification."""
        # Handle input size mismatch - resample time series to match expected model input
        expected_length = 20001  # From the error message
        actual_length = len(component_time_series)

        if actual_length != expected_length:
            # Resample the time series to match the expected length
            from scipy import signal
            time_data = signal.resample(component_time_series, expected_length)
        else:
            time_data = component_time_series.copy()

        # Normalize time series
        time_data = (time_data - np.mean(time_data)) / (np.std(time_data) + 1e-10)

        # Extract features
        topo_features = self.extract_topo_features(component_topo)
        spec_features = self.extract_spectral_features(time_data)

        # Reshape time data for CNN input - ensure correct shape
        time_data_reshaped = time_data.reshape(1, expected_length, 1)
        topo_features_reshaped = topo_features.reshape(1, -1)
        spec_features_reshaped = spec_features.reshape(1, -1)

        return [time_data_reshaped, topo_features_reshaped, spec_features_reshaped]

    def classify_component(self, component_time_series, component_topo):
        """Classify a single ICA component.

        Parameters
        ----------
        component_time_series : array
            Time series of the ICA component
        component_topo : array
            Topography map of the ICA component

        Returns
        -------
        dict
            Classification result with class, probability, and all class probabilities
        """
        if self.model is None or self.label_encoder is None:
            return {'class': 'unknown', 'probability': 0.0, 'probabilities': {}}

        # Prepare data
        model_inputs = self.prepare_component_data(component_time_series, component_topo)

        # Make prediction
        prediction = self.model.predict(model_inputs, verbose=0)
        pred_class_idx = np.argmax(prediction[0])
        pred_class = self.label_encoder.inverse_transform([pred_class_idx])[0]
        pred_prob = prediction[0][pred_class_idx]

        # Create probabilities dictionary
        all_probs = {self.label_encoder.inverse_transform([i])[0]: float(prediction[0][i])
                     for i in range(len(prediction[0]))}

        return {
            'class': pred_class,
            'probability': float(pred_prob),
            'probabilities': all_probs
        }

    def classify_ica(self, ica, inst):
        """Classify all components in an ICA decomposition.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            The fitted ICA object
        inst : mne.io.Raw or mne.Epochs
            The data instance used to fit the ICA

        Returns
        -------
        dict
            Dictionary with component classifications, suggested exclusions, and details
        """
        if self.model is None or self.label_encoder is None:
            return {'classifications': {}, 'exclude': [], 'details': {}}

        # Get components and topographies
        components = ica.get_sources(inst)
        component_data = components.get_data()

        # For epochs, average across epochs
        if isinstance(components, mne.Epochs):
            component_data = np.mean(component_data, axis=0)

        weights = ica.get_components()
        n_components = ica.n_components_

        # Store classifications
        classifications = {}
        details = {}

        # Classes to automatically exclude
        exclude_classes = ['muscle', 'eye blink', 'eye movement', 'tms-pulse', 'tms-decay',
                           'tms-ringing', 'line noise', 'auditory evoked']

        # Classify each component
        exclude_idx = []
        for comp_idx in range(n_components):
            # Get time series
            time_series = component_data[comp_idx]

            # Get topography
            topo = weights[:, comp_idx]

            # Classify
            result = self.classify_component(time_series, topo)

            # Store classification
            classifications[comp_idx] = result['class']
            details[comp_idx] = result

            if result['class'] in exclude_classes and result['probability'] > self.probability_threshold:
                exclude_idx.append(comp_idx)

        return {
            'classifications': classifications,
            'exclude': exclude_idx,
            'details': details
        }