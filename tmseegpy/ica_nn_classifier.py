# ica_nn_classifier.py
import numpy as np
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_ica_classification(ica, inst, classification_results, figsize=(15, 10)):
    """Plot ICA components with their neural network classifications."""
    sources = ica.get_sources(inst)
    if isinstance(sources, mne.Epochs):
        source_data = np.mean(sources.get_data(), axis=0)
    else:
        source_data = sources.get_data()

    classifications = classification_results['classifications']
    details = classification_results['details']
    exclude = classification_results['exclude']
    n_components = ica.n_components_

    fig = plt.figure(figsize=figsize)
    n_rows = int(np.ceil(n_components / 4))
    gs = GridSpec(n_rows, 4, figure=fig)

    for comp_idx in range(n_components):
        ax = fig.add_subplot(gs[comp_idx // 4, comp_idx % 4])

        if comp_idx in classifications:
            comp_class = classifications[comp_idx]
            prob = details[comp_idx]['probability']
            title = f"IC{comp_idx}: {comp_class} ({prob:.2f})"
            color = 'red' if comp_idx in exclude else 'black'
        else:
            title = f"IC{comp_idx}"
            color = 'black'

        ax.plot(source_data[comp_idx], color=color)
        ax.set_title(title, fontsize=10, color=color)
        ax.set_xticks([])

        ax_topo = plt.axes([ax.get_position().x0 + 0.01, ax.get_position().y0 + 0.01,
                            0.05, 0.05])
        mne.viz.plot_topomap(ica.get_components()[:, comp_idx], ica.info, axes=ax_topo,
                             show=False)

    plt.tight_layout()
    return fig


def plot_classification_summary(classification_results, figsize=(10, 6)):
    """Plot summary of component classifications."""
    classifications = classification_results['classifications']

    class_counts = {}
    for comp_idx, comp_class in classifications.items():
        if comp_class not in class_counts:
            class_counts[comp_class] = 0
        class_counts[comp_class] += 1

    fig, ax = plt.subplots(figsize=figsize)

    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    sorted_indices = np.argsort(counts)[::-1]
    classes = [classes[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]

    bars = ax.bar(classes, counts)

    exclude_classes = ['muscle', 'eye blink', 'eye movement', 'tms-pulse', 'tms-decay',
                       'tms-ringing', 'line noise', 'auditory evoked']

    for i, c in enumerate(classes):
        if c in exclude_classes:
            bars[i].set_color('red')

    ax.set_title('ICA Component Classification Summary')
    ax.set_ylabel('Count')
    ax.set_xlabel('Component Type')
    ax.set_xticklabels(classes, rotation=45, ha='right')

    plt.tight_layout()
    return fig


class VersionIndependentEncoder:
    """Simple version-independent label encoder."""

    def __init__(self, classes=None):
        self.classes_ = np.array(classes) if classes is not None else np.array([])
        self._class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}

    def transform(self, labels):
        return np.array([self._class_to_index.get(lbl, -1) for lbl in labels])

    def inverse_transform(self, indices):
        return self.classes_[indices]

    @classmethod
    def from_sklearn_encoder(cls, sklearn_encoder):
        return cls(classes=sklearn_encoder.classes_)


# Model definition
class TMSClassifier(nn.Module):
    """CNN model for ICA component classification."""

    def __init__(self, input_trials=119, seq_length=20001, num_classes=10):
        super(TMSClassifier, self).__init__()
        self.downsample_factor = 5

        self.conv1 = nn.Conv1d(
            in_channels=input_trials,
            out_channels=32,
            kernel_size=25,
            stride=5,
            padding=12
        )

        self.pool = nn.MaxPool1d(kernel_size=5, stride=5)

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x[:, :, ::self.downsample_factor]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ICAComponentClassifier:
    """Classifier for TMS-EEG ICA components."""

    def __init__(self, model_path=None, label_encoder_path=None, probability_threshold=0.3):
        self.probability_threshold = probability_threshold

        # Default paths
        if model_path is None:
            model_path = Path(__file__).parent / 'models' / 'tms_classifier_augmented.pth'

        if label_encoder_path is None:
            label_encoder_path = Path(__file__).parent / 'models' / 'label_encoder.pkl'

        # Set CPU device for initial loading
        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'mps' if torch.backends.mps.is_available() else
                                   'cpu')

        # Load model and label encoder
        self.model = self._load_model(model_path)
        self.label_encoder = self._load_label_encoder(label_encoder_path)

        # ADDED: Print label encoder classes for verification
        if hasattr(self.label_encoder, 'classes_'):
            print("Label encoder classes:")
            for i, class_name in enumerate(self.label_encoder.classes_):
                print(f"  {i}: {class_name}")

    def _load_model(self, model_path):
        """Load the PyTorch model with Apple Silicon compatibility."""
        try:
            # Always load to CPU first (especially important for MPS)
            model = TMSClassifier()
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)

            # Then move to target device
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _load_label_encoder(self, label_encoder_path):
        """Load the label encoder, handling sklearn version differences."""
        try:
            with open(label_encoder_path, 'rb') as f:
                loaded_encoder = pickle.load(f)

            # Convert sklearn encoder to version-independent
            if hasattr(loaded_encoder, 'classes_'):
                return VersionIndependentEncoder.from_sklearn_encoder(loaded_encoder)
            return loaded_encoder

        except Exception as e:
            print(f"Error loading label encoder: {e}")
            # Default encoder with common ICA classes
            return VersionIndependentEncoder(classes=[
                'auditory evoked', 'brain', 'eye blink', 'eye movement',
                'line noise', 'muscle', 'other', 'tms-decay', 'tms-pulse', 'tms-ringing'
            ])

    def prepare_component_data(self, component_time_series, component_topo, inst):
        """Prepare component time series for model input."""
        expected_trials = 119
        expected_timepoints = 20001

        # Handle epochs data
        if isinstance(inst, mne.Epochs):
            sources = inst.get_data()

            # Get this component's data across all epochs
            if hasattr(self, 'current_comp_idx'):
                comp_data = sources[:, self.current_comp_idx, :]
            else:
                comp_data = np.expand_dims(component_time_series, axis=0)

            # Handle trial count and length
            if comp_data.shape[0] < expected_trials:
                n_repeats = int(np.ceil(expected_trials / comp_data.shape[0]))
                comp_data = np.tile(comp_data, (n_repeats, 1))[:expected_trials]

            if comp_data.shape[0] > expected_trials:
                comp_data = comp_data[:expected_trials]

            if comp_data.shape[1] != expected_timepoints:
                from scipy import signal
                comp_data = np.array([signal.resample(trial, expected_timepoints) for trial in comp_data])

        # Handle raw data
        else:
            comp_data = np.tile(component_time_series, (expected_trials, 1))

            if comp_data.shape[1] != expected_timepoints:
                from scipy import signal
                comp_data = np.array([signal.resample(trial, expected_timepoints) for trial in comp_data])

        # Normalize each trial
        for i in range(comp_data.shape[0]):
            trial = comp_data[i]
            comp_data[i] = (trial - np.mean(trial)) / (np.std(trial) + 1e-10)

        # Convert to tensor and add batch dimension
        tensor_data = torch.tensor(comp_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor_data

    def classify_component(self, component_time_series, component_topo, inst):
        """Classify a single ICA component."""
        if self.model is None or self.label_encoder is None:
            return {'class': 'unknown', 'probability': 0.0, 'probabilities': {}}

        try:
            model_input = self.prepare_component_data(component_time_series, component_topo, inst)

            with torch.no_grad():
                logits = self.model(model_input)
                probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()

            pred_class_idx = np.argmax(probabilities)
            pred_class = self.label_encoder.inverse_transform([pred_class_idx])[0]
            pred_prob = probabilities[pred_class_idx]

            all_probs = {self.label_encoder.inverse_transform([i])[0]: float(probabilities[i])
                         for i in range(len(probabilities))}

            return {
                'class': pred_class,
                'probability': float(pred_prob),
                'probabilities': all_probs
            }
        except Exception as e:
            print(f"Error classifying component: {e}")
            return {'class': 'error', 'probability': 0.0, 'probabilities': {}}

    def classify_ica(self, ica, inst):
        """Classify all components in an ICA decomposition."""
        if self.model is None or self.label_encoder is None:
            return {'classifications': {}, 'exclude': [], 'details': {}}

        # Get components
        components = ica.get_sources(inst)
        component_data = components.get_data()

        # For epochs, average for visualization
        if isinstance(components, mne.Epochs):
            component_data_avg = np.mean(component_data, axis=0)
        else:
            component_data_avg = component_data

        weights = ica.get_components()
        n_components = ica.n_components_

        classifications = {}
        details = {}
        exclude_classes = ['muscle', 'eye blink', 'eye movement', 'tms-pulse', 'tms-decay',
                           'tms-ringing', 'line noise', 'auditory evoked']
        exclude_idx = []

        print("Classifying components using neural network...")
        for comp_idx in range(n_components):
            self.current_comp_idx = comp_idx

            time_series = component_data_avg[comp_idx]
            topo = weights[:, comp_idx]

            result = self.classify_component(time_series, topo, inst)

            classifications[comp_idx] = result['class']
            details[comp_idx] = result

            # MODIFIED EXCLUSION LOGIC:
            # Exclude artifacts with confidence > threshold OR brain with confidence < 0.5
            if (result['class'] in exclude_classes and result['probability'] > self.probability_threshold) or \
                    (result['class'] == 'brain' and result['probability'] < 0.5):
                exclude_idx.append(comp_idx)
                exclusion_reason = "low confidence brain" if result['class'] == 'brain' else "artifact"
                print(
                    f"  Component {comp_idx}: {result['class']} ({result['probability']:.2f}) - EXCLUDED ({exclusion_reason})")
            else:
                print(f"  Component {comp_idx}: {result['class']} ({result['probability']:.2f})")

        if hasattr(self, 'current_comp_idx'):
            delattr(self, 'current_comp_idx')

        return {
            'classifications': classifications,
            'exclude': exclude_idx,
            'details': details
        }