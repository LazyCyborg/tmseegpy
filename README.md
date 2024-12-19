# tmseegpy

This repository contains my attempt at building some sort of pipeline for preprocessing and analyzing Transcranial Magnetic Stimulation (TMS)-EEG data. The pipeline includes steps for artifact removal, filtering, Independent Component Analysis (ICA), muscle artifact cleaning (using Tensorly), and analysis of Perturbational Complexity Index based on State transitions (PCIst) (Comolatti et al., 2019). The analysis of PCIst is jsut a copy paste from https://github.com/renzocom/PCIst/blob/master/PCIst/pci_st.py which is written by Renzo Comolatti. The code is mostly adapted from a very long jupyter notebook which used mostly native MNE-Python methods which I expanded to a toolbox that I have been using in my analyses. So the code base might not be very efficient. 

Currently the code is only tested on TMS-EEG data recorded in .ses format from the Bittium NeurOne 5kHz sampling rate amplifier. Feel free to modify the preproc module to fit other types of recorded TMS-EEG data. If you create a dataloader that is compatible with multiple systems feel free to reach out to hjarneko@gmail.com. The package uses a modified version of the neurone_loader (https://github.com/heilerich/neurone_loader) to load the data from the Bittium NeurOne and convert it to an MNE-Python raw object. 

# Table of Contents

- [Installation](#installation)
  - [Use the scripts](#installation-use-the-scripts)
  - [Install as a package](#install-as-a-package)
- [Usage](#usage)
  - [GUI](#gui)
  - [Command-Line Arguments](#command-line-arguments)
  - [Example Usage](#example-usage)
- [Data Preparation](#data-preparation)
- [Processing Pipeline](#processing-pipeline)
- [Example pipeline](#example-pipeline-that-i-used)
  - [Initial Data Loading and Setup](#initial-data-loading-and-setup)
  - [Stage 1: Initial TMS Artifact Handling](#stage-1-initial-tms-artifact-handling)
  - [Stage 2: Filtering and Epoching](#stage-2-filtering-and-epoching)
  - [Stage 3: First Artifact Cleaning](#stage-3-first-artifact-cleaning)
  - [Stage 4: Extended TMS Artifact Handling](#stage-4-extended-tms-artifact-handling)
  - [Stage 5: Final Cleaning](#stage-5-final-cleaning)
  - [Quality Control and Analysis](#quality-control-and-analysis)
  - [PCIst Calculation](#pcist-calculation)
- [Modules and Classes](#modules-and-classes)
  - [TMSEEGPreprocessor](#tmseegpreprocessor-main-class-for-preprocessing)
  - [TMSArtifactCleaner Class](#tmsartifactcleaner-class-which-might-work)
    - [What it does](#what-it-does)
    - [Parameters](#parameters)
    - [Methods](#methods)
  - [PCIst](#pcist)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Installation (use the scripts)

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

## Install as a package 

1. Clone the repository:

   ```bash
   git clone https://github.com/LazyCyborg/tmseegpy.git
   cd tmseegpy

2. Create a virtual environment (optional but recommended):

   ```bash
    conda env create -f eeg_env.yml
    conda activate eeg
   ```

3. Install the package:

   ```bash
   pip install -e .
   ```

## Usage

The pipeline is designed to be run from the command line or through the simple GUI, processing EEG data for one or multiple subjects. The main script accepts various command-line arguments to customize the preprocessing and analysis steps.

### GUI

For an graphical user interface run:

   ```bash
   cd tms_eeg_analysis
   python gui.py 
   ```
The GUI is basically a wrapper for the argparser bellow and is intended as the main way to test the pipeline in a clinical setting. 

### Command-Line Arguments
The following parameters can be configured either through command-line arguments or the GUI interface:

| Argument                    | Type  | Default  | Description                                                                                |
|-----------------------------|-------|----------|--------------------------------------------------------------------------------------------|
| Preprocessing Parameters |
| --data_dir                  | str   | ./data   | Path to the data directory containing EEG recordings                                       |
| --plot_preproc              | flag  | False    | Enable plotting during preprocessing for quality checks                                    |
| --ds_sfreq                  | float | 725      | Desired sampling frequency after downsampling                                             |
| --random_seed               | int   | 42       | Seed for random number generators for reproducibility                                      |
| --bad_channels_threshold    | float | 1        | Threshold for detecting bad channels using the FASTER algorithm                           |
| --bad_epochs_threshold      | float | 1        | Threshold for detecting bad epochs using the FASTER algorithm                             |
| --apply_ssp                | flag  | False    | Apply Signal Space Projection                                                               |
| --ssp_n_eeg                | int   | 2        | Number of SSP components to apply                                                         |
| --substitute_zero_events_with | int   | 10       | Value to substitute zero events in the data                                             |
| Filtering Parameters |
| --l_freq                    | float | 1        | Lower cutoff frequency for bandpass filter                                               |
| --h_freq                    | float | 45       | Upper cutoff frequency for bandpass filter                                               |
| --notch_freq               | float | 50       | Frequency for notch filter (e.g., to remove powerline noise)                             |
| --notch_width              | float | 2        | Width of the notch filter                                                                |
| TMS Artifact Removal Parameters |
| --initial_cut_start        | float | -2       | Start time (ms) for initial cutting around TMS pulse                                     |
| --initial_cut_end          | float | 10       | End time (ms) for initial cutting around TMS pulse                                       |
| --initial_interp_window    | float | 1.0      | Initial interpolation window size (ms)                                                   |
| --extended_cut_start       | float | -2       | Start time (ms) for extended artifact removal                                            |
| --extended_cut_end         | float | 15       | End time (ms) for extended artifact removal                                              |
| --extended_interp_window   | float | 5.0      | Extended interpolation window size (ms)                                                  |
| --interpolation_method     | str   | 'cubic'  | Interpolation method ('linear' or 'cubic')                                               |
| Muscle Artifact Parameters |
| --clean_muscle_artifacts   | flag  | False    | Enable muscle artifact cleaning using tensor decomposition                               |
| --muscle_window_start      | float | 0.005    | Start time (s) for muscle artifact detection window                                      |
| --muscle_window_end        | float | 0.030    | End time (s) for muscle artifact detection window                                        |
| --threshold_factor         | float | 1.0      | Threshold factor for muscle artifact detection                                           |
| --n_components             | int   | 5        | Number of components for tensor decomposition during muscle artifact cleaning             |
| ICA Parameters |
| --ica_method               | str   | 'fastica'| ICA method for the first ICA pass ('fastica' or 'infomax')                              |
| --tms_muscle_thresh        | float | 2.0      | Threshold for detecting TMS-evoked muscle artifacts during ICA                           |
| --second_ica_method        | str   | 'infomax'| ICA method for the second ICA pass ('infomax' or 'fastica')                             |
| CSD Parameters |
| --apply_csd                | flag  | True    | Apply Current Source Density (CSD) transformation                                        |
| --lambda2                  | float | 1e-3     | Lambda2 parameter for CSD transformation                                                 |
| --stiffness               | int   | 3        | Stiffness parameter for CSD transformation                                               |
| Epoching Parameters |
| --epochs_tmin              | float | -0.41    | Start time (s) for epochs relative to the event                                         |
| --epochs_tmax              | float | 0.41     | End time (s) for epochs relative to the event                                           |
| --baseline_start          | float | -400     | Start time (ms) for baseline correction window                                           |
| --baseline_end            | float | -50      | End time (ms) for baseline correction window                                             |
| --amplitude_threshold     | float | 4500     | Amplitude threshold (µV) for epoch rejection                                             |
| PCIst Parameters |
| --response_start          | int   | 0        | Start time (ms) for the response window in PCIst analysis                                |
| --response_end            | int   | 299      | End time (ms) for the response window in PCIst analysis                                  |
| --k                       | float | 1.2      | PCIst parameter k                                                                        |
| --min_snr                 | float | 1.1      | Minimum SNR threshold for PCIst analysis                                                 |
| --max_var                 | float | 99.0     | Maximum variance percentage to retain in PCA during PCIst                                |
| --embed                   | flag  | False    | Enable time-delay embedding in PCIst analysis                                            |
| --n_steps                 | int   | 100      | Number of steps for threshold optimization in PCIst analysis                             |
| Statistics                |
| --research                | bool  | False    | Output summary statistics of measurements                                                |


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

# Processing Pipeline

The pipeline processes TMS-EEG data through these stages which are roughly modelled from the reccomendations in:

Comolatti, R., Pigorini, A., Casarotto, S., Fecchio, M., Faria, G., Sarasso, S., Rosanova, M., Gosseries, O., Boly, M., Bodart, O., Ledoux, D., Brichant, J. F., Nobili, L., Laureys, S., Tononi, G., Massimini, M., & Casali, A. G. (2019). A fast and general method to empirically estimate the complexity of brain responses to transcranial and intracranial stimulations. Brain Stimulation, 12(5), 1280–1289. https://doi.org/10.1016/j.brs.2019.05.013

and from 

https://nigelrogasch.gitbook.io/tesa-user-manual/example_pipelines

And the custom functions (TMS-artifact removal and classification of ICA components) are modelled after the TESA toolbox:

Rogasch NC, Sullivan C, Thomson RH, Rose NS, Bailey NW, Fitzgerald PB, Farzan F, Hernandez-Pavon JC. Analysing concurrent transcranial magnetic stimulation and electroencephalographic data: a review and introduction to the open-source TESA software. NeuroImage. 2017; 147:934-951.

Mutanen TP, Biabani M, Sarvas J, Ilmoniemi RJ, Rogasch NC. Source-based artifact-rejection techniques available in TESA, an open-source TMS-EEG toolbox. Brain Stimulation. 2020; In press.


## Example pipeline (that I used)

This is the pipeline that I used after a lot of trial and error. The only way that I "validate" the pipeline is by checking if the final EEG-data looks reasonable, if the evoked plots contain some sort of TEPs and if the PCI-values are stabel across recordings. The pipeline is only tested on 10 recordings from the same healthy subject who either was awake or slightly somnolent.

## Initial Data Loading and Setup
1. Load raw data from Neurone format
2. Set random seed (42)
3. Create events from trigger channel ('STI 014')
4. Remove non-EEG channels (e.g., EMG1)

## Stage 1: Initial TMS Artifact Handling
5. First-pass TMS artifact removal
   - Cut window: -2 to 10 ms around TMS pulse and replace with zeros
6. Initial interpolation of TMS artifact
   - Method: cubic
   - Interpolation window: 1.0 ms
   - Cut times: -2 to 10 ms

## Stage 2: Filtering and Epoching
7. Apply frequency filters (I chose a 1 Hz lowpass since the final EEG data had some drift artifacts when using 0.1 Hz)
   - Low-pass: 1 Hz
   - High-pass: 45 Hz
   - Notch filter: 50 Hz (width: 2 Hz)
8. Create epochs
   - Time window: -0.41 to 0.41 s
   - Amplitude threshold: 4500 µV
   - Demeaning 
9. Remove bad channels
   - Using FASTER algorithm
   - Threshold: 1
10. Remove bad epochs
    - Using FASTER algorithm
    - Threshold: 1
11. Set average reference

## Stage 3: First Artifact Cleaning
12. First ICA run
    - Method: FastICA
    - TMS muscle threshold: 2.0

- Optional muscle artifact cleaning (PARAFAC) (I did not use this)
    - Time window: 5-30 ms
    - Threshold factor: 1.0
    - Components: 5

## Stage 4: Extended TMS Artifact Handling
13. Second-pass TMS artifact removal
    - Extended cut window: -2 to 15 ms
14. Extended interpolation
    - Method: cubic
    - Interpolation window: 5.0 ms
    - Cut times: -2 to 15 ms

## Stage 5: Final Cleaning
15. Second ICA run
    - Method: Infomax
    - Auto-reject components with mne_ica_label, labeled as: eye blink, heart beat, muscle artifact, channel noise, line noise
- Apply SSP (Optional and I did not use this) (https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html)
    - Number of EEG components: 2
16. Apply baseline correction
    - Window: -400 to -50 ms
17. CSD transformation (https://mne.tools/stable/auto_examples/preprocessing/eeg_csd.html)
    - Lambda2: 1e-3
    - Stiffness: 3
18. Final downsampling
    - Target frequency: 725 Hz

## Quality Control and Analysis
19. TEP validation (optional)
    - Peak prominence threshold: 0.01
20. Generate evoked response plot
    - Time window: -0.3 to 0.3 s
    - Y-limits: -2 to 2 µV

## PCIst Calculation
21. Calculate PCIst
    - Response window: 0 to 299 ms
    - k parameter: 1.2
    - Minimum SNR: 1.1
    - Maximum variance: 99.0%
    - Steps for threshold optimization: 100


## Modules and Classes

### TMSEEGPreprocessor (main class for preprocessing)

I used this in a jupyter notebook to simply chain steps from mne with FASTER and mne_ica_label as well as the the custom functions for artifact removal, interpolation and ICA component selection based on Z-score threshold based on TESA. 

 Methods included (They are pretty well documented so look in the preproc.py file for more info):

- `remove_tms_artifact`: Removes TMS artifacts from raw data.
- `interpolate_tms_artifact`: Interpolates removed TMS artifacts using scipy cubic interpolation.
- `filter_raw`: Applies bandpass and notch filters to raw data.
- `create_epochst`: Creates epochs from the mne.raw object and rejects epochs with amplitudes over >4500 (can be changed), and demeans the data. 
- `remove_bad_channels`: Identifies and interpolates bad channels using MNE-FASTER.
- `remove_bad_epochs`: Removes bad epochs based on amplitude using MNE-FASTER.
- `run_ica`: Runs ICA and attempts to classify components as TMS-evoked muscle activity using Z-core threshold.
- `clean_muscle_artifacts`: Cleans TMS-evoked muscle artifacts using non-negative tensor decomposition (se bellow).
- `run_second_ica`: Runs ICA to identify and remove residual physiological artifacts using MNE-ICALabel.
- `apply_ssp`: Aplies signal-space projection (SSP) to epochs.
- `apply_csd`: Applies current Source Density transformation.
- `set_average_reference`: Sets the EEG reference to average.
- `apply_baseline_correction`: Applies baseline correction to epochs.
- `apply_csd`: Applies Current Source Density transformation.


### TMSArtifactCleaner Class (which might work)

The TMSArtifactCleaner class is designed to detect and clean transcranial magnetic stimulation (TMS)-evoked muscle artifacts in EEG/MEG data using tensor decomposition techniques. It uses the tensorly library for tensor operations and mne for handling eeg data.

#### What it does 

- Artifact Detection: Uses Non-negative PARAFAC tensor decomposition to detect muscle artifacts in EEG epochs.
- Artifact Cleaning: Uses Tucker decomposition to clean the detected artifacts.
- Threshold Optimization: Includes a method to find the optimal detection threshold based on a target detection rate.


#### Parameters:
- epochs (mne.Epochs): The EEG epochs to process.
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

- Comolatti, R., Pigorini, A., Casarotto, S., Fecchio, M., Faria, G., Sarasso, S., Rosanova, M., Gosseries, O., Boly, M., Bodart, O., Ledoux, D., Brichant, J. F., Nobili, L., Laureys, S., Tononi, G., Massimini, M., & Casali, A. G. (2019). A fast and general method to empirically estimate the complexity of brain responses to transcranial and intracranial stimulations. Brain Stimulation, 12(5), 1280–1289. https://doi.org/10.1016/j.brs.2019.05.013

The pipeline uses the MNE-Python library for EEG data processing:

- Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7 DEC. https://doi.org/10.3389/fnins.2013.00267

The bad channel, and epoch detection uses MNE-FASTER:

- Nolan H, Whelan R, Reilly RB. FASTER: Fully Automated Statistical Thresholding for EEG artifact Rejection. J Neurosci Methods. 2010 Sep 30;192(1):152-62. doi: 10.1016/j.jneumeth.2010.07.015. Epub 2010 Jul 21. PMID: 20654646.

The second ICA uses MNE-ICALabel:

- Li, A., Feitelberg, J., Saini, A. P., Höchenberger, R., & Scheltienne, M. (2022). MNE-ICALabel: Automatically annotating ICA components with ICLabel in Python. Journal of Open Source Software, 7(76), 4484. https://doi.org/10.21105/joss.04484