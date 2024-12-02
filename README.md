# TMS-EEG Preprocessing and Analysis Pipeline

This repository provides some sort of pipeline for preprocessing and analyzing Transcranial Magnetic Stimulation (TMS)-EEG data. The pipeline includes steps for artifact removal, filtering, Independent Component Analysis (ICA), muscle artifact cleaning (using Tensorly), and analysis of Perturbational Complexity Index based on State transitions (PCIst) (Comolatti et al., 2019).

Currently the code is only tested on TMS-EEG data recorded in .ses format from the Bittium NeurOne 5kHz sampliing rate amplifier. Feel free to modify the preproc module to fit other types of recorded TMS-EEG data. If you create a dataloader that is compatible with multiple systems feel free to reach out to hjarneko@gmail.com. The package uses a modified version of the neurone_loader (https://github.com/heilerich/neurone_loader) to load the data from the Bittium NeurOne and convert it to an MNE-Python raw object. 

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Example Usage](#example-usage)
- [Data Preparation](#data-preparation)
- [Processing Pipeline](#processing-pipeline)
  1. [Data Loading](#1-data-loading)
  2. [TMS Artifact Removal](#2-tms-artifact-removal)
  3. [Filtering](#3-filtering)
  4. [Epoching](#4-epoching)
  5. [Bad Channel and Epoch Detection](#5-bad-channel-and-epoch-detection)
  6. [ICA and Artifact Removal](#6-ica-and-artifact-removal)
  7. [Baseline Correction and Referencing](#7-baseline-correction-and-referencing)
  8. [Current Source Density (CSD) Transformation](#8-current-source-density-csd-transformation)
  9. [Downsampling](#9-downsampling)
  10. [Final Quality Check](#10-final-quality-check)
  11. [PCIst Analysis](#11-pcist-analysis)
  12. [Microstate Analysis](#12-microstate-analysis)
- [Modules and Classes](#modules-and-classes)
  - [TMSEEGPreprocessor](#tmseepreprocessor)
  - [TMSArtifactCleaner](#tmsartifactcleaner)
  - [PCIst](#pcist)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The TMS-EEG Preprocessing and Analysis Pipeline is designed to automate and standardize the preprocessing of TMS-EEG data. It includes:

- TMS artifact removal and interpolation
- Filtering and downsampling
- Bad channel and epoch detection using the FASTER algorithm
- Independent Component Analysis (ICA) for artifact removal
- Muscle artifact cleaning using tensor decomposition
- Baseline correction and referencing
- Current Source Density (CSD) transformation
- Advanced analyses such as PCIst and microstate analysis

## Features

- Automated Preprocessing: Streamlines the preprocessing steps required for TMS-EEG data.
- Artifact Removal: Implements both traditional and advanced methods for artifact detection and removal.
- Flexible Configuration: Allows customization of preprocessing parameters through command-line arguments.
- Advanced Analysis: Includes PCIst calculation and microstate analysis for in-depth data interpretation.
- Visualization: Provides plotting functions for quality checks and result visualization.
- Optional GUI with the gui.py


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/LazyCyborg/tmseegpy.git
   cd tmseegpy

2. Create a virtual environment (optional but recommended):

   ```bash
    conda env create -f eeg_env.yml
    conda activate eeg
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Required libraries:

   - NumPy
   - SciPy
   - MNE
   - scikit-learn
   - tensorly
   - matplotlib
   - seaborn
   - tqdm
   - mne-icalabel
   - mne-faster

4. Install the package (if applicable):

   If the code is organized as a package, you can install it locally:

   ```bash
   pip install -e .
   ```

## Usage

The pipeline is designed to be run from the command line or through the simple GUI, processing EEG data for one or multiple subjects. The main script accepts various command-line arguments to customize the preprocessing and analysis steps.

For GUI run:

   ```bash
   python gui.py 
   ```

### Command-Line Arguments

| Argument                    | Type  | Default  | Description                                                                                |
|-----------------------------|-------|----------|--------------------------------------------------------------------------------------------|
| --data_dir                  | str   | ./data   | Path to the data directory containing EEG recordings.                                     |
| --plot_preproc              | flag  | False    | Enable plotting during preprocessing for quality checks.                                   |
| --random_seed               | int   | 42       | Seed for random number generators for reproducibility.                                    |
| --substitute_zero_events_with | int   | 10       | Value to substitute zero events in the data.                                              |
| --ds_sfreq                  | float | 725      | Desired sampling frequency after downsampling.                                            |
| --cut_times_tms_start       | float | -2       | Start time (ms) for cutting around TMS pulse for artifact removal.                        |
| --cut_times_tms_end         | float | 10       | End time (ms) for cutting around TMS pulse for artifact removal.                          |
| --interpolation_method      | str   | 'cubic'  | Interpolation method ('linear' or 'cubic').                                               |
| --interp_window_start       | float | 20       | Start time (ms) for interpolation window before artifact.                                 |
| --interp_window_end         | float | 20       | End time (ms) for interpolation window after artifact.                                    |
| --fix_artifact_window_start | float | -0.005   | Start time (s) for fixing stimulus artifact using MNE's function.                         |
| --fix_artifact_window_end   | float | 0.015    | End time (s) for fixing stimulus artifact using MNE's function.                           |
| --l_freq                    | float | 0.1      | Lower cutoff frequency for bandpass filter.                                               |
| --h_freq                    | float | 45       | Upper cutoff frequency for bandpass filter.                                               |
| --notch_freq                | float | 50       | Frequency for notch filter (e.g., to remove powerline noise).                             |
| --notch_width               | float | 2        | Width of the notch filter.                                                                |
| --epochs_tmin               | float | -0.41    | Start time (s) for epochs relative to the event.                                          |
| --epochs_tmax               | float | 0.41     | End time (s) for epochs relative to the event.                                            |
| --bad_channels_threshold    | float | 3        | Threshold for detecting bad channels using the FASTER algorithm.                          |
| --bad_epochs_threshold      | float | 3        | Threshold for detecting bad epochs using the FASTER algorithm.                            |
| --ica_method                | str   | 'fastica'| ICA method for the first ICA pass ('fastica' or 'infomax').                               |
| --tms_muscle_thresh         | float | 3.0      | Threshold for detecting TMS-evoked muscle artifacts during ICA.                           |
| --clean_muscle_artifacts    | flag  | False    | Enable muscle artifact cleaning using tensor decomposition.                               |
| --muscle_window_start       | float | 0.005    | Start time (s) for muscle artifact detection window.                                      |
| --muscle_window_end         | float | 0.030    | End time (s) for muscle artifact detection window.                                        |
| --threshold_factor          | float | 1.0      | Threshold factor for muscle artifact detection.                                           |
| --n_components              | int   | 5        | Number of components for tensor decomposition during muscle artifact cleaning.            |
| --second_ica_method         | str   | 'infomax'| ICA method for the second ICA pass.                                                       |
| --ssp_n_eeg                 | int   | 2        | Number of SSP components to apply.                                                        |
| --apply_csd                 | flag  | False    | Apply Current Source Density (CSD) transformation.                                        |
| --lambda2                   | float | 1e-5     | Lambda2 parameter for CSD transformation.                                                 |
| --stiffness                 | int   | 4        | Stiffness parameter for CSD transformation.                                               |
| --baseline_start            | float | -0.4     | Start time (s) for baseline correction window.                                            |
| --baseline_end              | float | -0.005   | End time (s) for baseline correction window.                                              |
| --response_start            | int   | 0        | Start time (ms) for the response window in PCIst analysis.                                |
| --response_end              | int   | 299      | End time (ms) for the response window in PCIst analysis.                                  |
| --k                         | float | 1.2      | PCIst parameter k.                                                                        |
| --min_snr                   | float | 1.1      | Minimum SNR threshold for PCIst analysis.                                                 |
| --max_var                   | float | 99.0     | Maximum variance percentage to retain in PCA during PCIst.                                |
| --embed                     | flag  | False    | Enable time-delay embedding in PCIst analysis.                                            |
| --n_steps                   | int   | 100      | Number of steps for threshold optimization in PCIst analysis.                             |
| --pre_window_start          | int   | -400     | Start time (ms) for the pre-TMS window in microstate analysis.                            |
| --pre_window_end            | int   | -50      | End time (ms) for the pre-TMS window in microstate analysis.                              |
| --post_window_start         | int   | 0        | Start time (ms) for the post-TMS window in microstate analysis.                           |
| --post_window_end           | int   | 300      | End time (ms) for the post-TMS window in microstate analysis.                             |
| --n_clusters                | int   | 4        | Number of clusters for global microstate analysis.                                        |
| --n_resamples               | int   | 20       | Number of resamples for microstate analysis.                                              |
| --n_samples                 | int   | 1000     | Number of samples for microstate analysis.                                                |
| --min_peak_distance         | int   | 1        | Minimum peak distance for microstate analysis.                                            |
| --preproc_qc                | bool  | False    | Generate preprocessing quality control statistics.                                        |
| --research                  | bool  | False    | Output summary statistics of measurements.                                                |


### Example Usage

To run the pipeline with default settings:

```bash
python main.py --data_dir ./TMSEEG
```

To enable muscle artifact cleaning and apply CSD transformation:

```bash
python main.py --data_dir ./TMSEEG --clean_muscle_artifacts --apply_csd
```

To enable plotting during preprocessing for quality checks:

```bash
python main.py --data_dir ./TMSEEG --plot_preproc
```

## Data Preparation

Ensure that your data is organized in the following structure (currently the toolbox only supports .ses files from Bittium NeurOne):

```
data/
└── TMSEEG/
    ├── session1/
    │     ├── DataSetSession.xml 
    ├── session2/
    │     ├── DataSetSession.xml 
    └── ...
```

- The `--data_dir` argument should point to the directory containing your TMS data (e.g., `data/`).
- Each session should be in its own subdirectory under `TMS1/`.

## Processing Pipeline

The pipeline processes TMS-EEG data through the fo stages:

### 1. Data Loading

- Function: Loads raw EEG data using MNE.
- Description: Reads the EEG recordings from the specified data directory and prepares them for preprocessing.

### 2. TMS Artifact Removal

- Function: `remove_tms_artifact`
- Description: Removes TMS artifacts by cutting out the artifact period and optionally replacing it with zeros or baseline data.

### 3. Filtering

- Function: `filter_raw`
- Description: Applies bandpass and notch filters to remove unwanted frequencies, such as DC offset and powerline noise.

### 4. Epoching

- Function: `create_epochs`
- Description: Segments the continuous data into epochs around the TMS events.

### 5. Bad Channel and Epoch Detection

- Functions: `remove_bad_channels`, `remove_bad_epochs`
- Description: Detects and removes bad channels and epochs using the FASTER algorithm.

### 6. ICA and Artifact Removal

- Functions: `run_ica`, `run_second_ica`, `clean_muscle_artifacts`
- Description:
  - First ICA: Focuses on TMS-evoked muscle artifacts.
  - Second ICA: Removes remaining artifacts such as eye blinks and heartbeats using ICLabel.
  - Muscle Artifact Cleaning: Optionally cleans muscle artifacts using tensor decomposition (se bellow).

### 7. Baseline Correction and Referencing

- Functions: `set_average_reference`, `apply_baseline_correction`, `apply_ssp`
- Description: Applies average referencing, baseline correction, and Signal Space Projection (SSP).

### 8. Current Source Density (CSD) Transformation

- Function: `apply_csd`
- Description: Enhances spatial resolution by applying the CSD transformation.

### 9. Downsampling

- Function: `downsample`
- Description: Reduces the sampling frequency to the desired rate to save computational resources.

### 10. Final Quality Check

- Function: `plot_evoked_response`
- Description: Plots the averaged evoked response for visual inspection.

### 11. PCIst Analysis

- Class: `PCIst`
- Description: Calculates the Perturbational Complexity Index based on State transitions, providing a measure of brain response complexity.

### 12. Microstate Analysis

- Class: `Microstate`
- Description: Performs microstate analysis across sessions, including global clustering and pre/post-TMS comparisons.

## Modules and Classes

### TMSEEGPreprocessor

Class for preprocessing TMS-EEG data.

 Methods:

- `remove_tms_artifact`: Removes TMS artifacts from raw data.
- `interpolate_tms_artifact`: Interpolates removed TMS artifacts using specified methods.
- `filter_raw`: Applies bandpass and notch filters to raw data.
- `create_epochs`: Creates epochs from continuous data.
- `remove_bad_channels`: Identifies and interpolates bad channels.
- `remove_bad_epochs`: Removes bad epochs based on amplitude criteria.
- `run_ica`: Runs ICA to identify and remove artifacts.
- `clean_muscle_artifacts`: Cleans TMS-evoked muscle artifacts.
- `set_average_reference`: Sets the EEG reference to average.
- `apply_baseline_correction`: Applies baseline correction to epochs.
- `apply_csd`: Applies Current Source Density transformation.


### TMSArtifactCleaner Class

The TMSArtifactCleaner class is designed to detect and clean transcranial magnetic stimulation (TMS)-evoked muscle artifacts in EEG/MEG data using tensor decomposition techniques. It leverages the tensorly library for tensor operations and mne for handling electrophysiological data.

#### Features

- Artifact Detection: Utilizes Non-negative PARAFAC tensor decomposition to detect muscle artifacts in EEG/MEG epochs.
- Artifact Cleaning: Employs Tucker decomposition to clean the detected artifacts while preserving the underlying neural signals.
- Parallel Processing: Implements parallel computation to speed up artifact detection across multiple epochs.
- Threshold Optimization: Includes a method to find the optimal detection threshold based on a target detection rate.


#### Parameters:
- epochs (mne.Epochs): The EEG/MEG epochs to process.
- verbose (bool, optional): Whether to print progress messages.

#### Methods

##### _normalize_data(data)

Normalizes the data across channels and time points using standard scaling.

##### Parameters:
- data (numpy.ndarray): The raw data to normalize.

##### Returns:
- normalized_data (numpy.ndarray): The normalized data.
- scalers (list): List of scalers used for each channel.

##### detect_muscle_artifacts(muscle_window=(0.005, 0.05), threshold_factor=5.0, n_jobs=-1, verbose=None)

Detects TMS-evoked muscle artifacts using parallel processing and Non-negative PARAFAC tensor decomposition.

###### Parameters:
- muscle_window (tuple, optional): Time window for artifact detection (in seconds).
- threshold_factor (float, optional): Threshold for detecting artifacts.
- n_jobs (int, optional): Number of parallel jobs (-1 uses all available CPUs).
- verbose (bool, optional): Whether to print progress messages.

###### Returns:
- artifact_info (dict): Information about detected artifacts and statistics.

##### clean_muscle_artifacts(n_components=2, verbose=None)

Cleans the detected muscle artifacts using Tucker decomposition.

###### Parameters:
- n_components (int, optional): Number of components for Tucker decomposition.
- verbose (bool, optional): Whether to print progress messages.

###### Returns:
- epochs_clean (mne.Epochs): The cleaned epochs.

##### find_optimal_threshold(muscle_window=(0.005, 0.05), target_detection_rate=0.5, initial_threshold=0.9, max_iter=100, tol=0.01, verbose=None)

Finds the optimal threshold for artifact detection using a binary search algorithm to match a target detection rate.

###### Parameters:
- muscle_window (tuple, optional): Time window for artifact detection (in seconds).
- target_detection_rate (float, optional): Desired detection rate for artifacts.
- initial_threshold (float, optional): Starting threshold for the search.
- max_iter (int, optional): Maximum number of iterations.
- tol (float, optional): Tolerance for convergence.
- verbose (bool, optional): Whether to print progress messages.

###### Returns:
- best_threshold (float): The optimal threshold found.

##### validate_cleaning(cleaned_epochs)

Validates the cleaning results 

###### Parameters:
- cleaned_epochs (mne.Epochs): The epochs after cleaning.

###### Returns:
- is_valid (bool): True if the cleaning is valid, False otherwise.


### PCIst

Class implementation of PCIst for calculating the Perturbational Complexity Index.

Methods:

- `calc_PCIst`: Calculates the PCIst value.
- `dimensionality_reduction`: Performs dimensionality reduction using SVD.
- `state_transition_quantification`: Quantifies state transitions in the signal.
- `plot_analysis`: Plots the analysis steps for visualization.


## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License.

## Acknowledgements

The PCIst implementation is based on:

- Comolatti R et al., "A fast and general method to empirically estimate the complexity of brain responses to transcranial and intracranial stimulations" Brain Stimulation (2019)

The pipeline uses the MNE-Python library for EEG data processing.

The bad channel, and epoch detection uses MNE-FASTER:

- Nolan H, Whelan R, Reilly RB. FASTER: Fully Automated Statistical Thresholding for EEG artifact Rejection. J Neurosci Methods. 2010 Sep 30;192(1):152-62. doi: 10.1016/j.jneumeth.2010.07.015. Epub 2010 Jul 21. PMID: 20654646.

The second ICA uses MNE-ICALabel:

- Li, A., Feitelberg, J., Saini, A. P., Höchenberger, R., & Scheltienne, M. (2022). MNE-ICALabel: Automatically annotating ICA components with ICLabel in Python. Journal of Open Source Software, 7(76), 4484. https://doi.org/10.21105/joss.04484