# tmseegpy

This repository contains my attempt at building some sort of pipeline for preprocessing and analyzing Transcranial Magnetic Stimulation (TMS)-EEG data. The pipeline includes steps for artifact removal, filtering, Independent Component Analysis (ICA), muscle artifact cleaning (using Tensorly), and analysis of Perturbational Complexity Index based on State transitions (PCIst) (Comolatti et al., 2019). The analysis of PCIst is just a copy paste from https://github.com/renzocom/PCIst/blob/master/PCIst/pci_st.py which is written by Renzo Comolatti. The code is mostly adapted from a very long jupyter notebook which used mostly native MNE-Python methods which I expanded to a toolbox that I have been using in my analyses. So the code base might not be very efficient. 

Currently the code is only tested on TMS-EEG data recorded in .ses format from the Bittium NeurOne 5kHz sampling rate amplifer. Feel free to modify the preproc module to fit other types of recorded TMS-EEG data. If you have trouble with the current dataloader and creates one that is compatible with multiple systems (maybe out of frustration) feel free to reach out to hjarneko@gmail.com. The package uses a modified version of the neurone_loader (https://github.com/heilerich/neurone_loader) to load the data from the Bittium NeurOne and convert it to an MNE-Python raw object. I am also using the 0.7.0 version (cloned) of mne_ica_label since there were some compatibility issues with the current version of mne.

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


## Installation 

1. Clone the repository:

   ```bash
   git clone https://github.com/LazyCyborg/tmseegpy.git
   cd tmseegpy

2. Create a virtual environment (optional but very recommended):

   ```bash
   conda env create -f environment.yml
   conda activate tmseegpy-env
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
   tmseegpy
   ```
The GUI is basically a wrapper for the argparser bellow and is intended as the main way to test the pipeline in a clinical setting. 

### Command-Line Arguments
Se run.py (bottom of the file) for full list of configurable command line arguments

### Example Usage

To run the pipeline with default settings:

```bash
cd tmseegpy
```

```bash
python run.py --data_dir ./TMSEEG
```

To enable muscle artifact cleaning and apply CSD transformation:

```bash
python run.py --data_dir ./TMSEEG --clean_muscle_artifacts --apply_csd
```

To enable plotting during preprocessing for quality checks:

```bash
python run.py --data_dir ./TMSEEG --plot_preproc
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
- Each session should be in its own subdirectory under `TMSEEG/`.

# Processing Pipeline

Below is the **updated** pipeline that aligns with the **default parameters** used in the `run.py` script (and in roughly the same order). These steps are still modelled after the recommendations in:

> Comolatti, R., Pigorini, A., Casarotto, S., Fecchio, M., Faria, G., Sarasso, S., Rosanova, M., Gosseries, O., Boly, M., Bodart, O., Ledoux, D., Brichant, J. F., Nobili, L., Laureys, S., Tononi, G., Massimini, M., & Casali, A. G. (2019). A fast and general method to empirically estimate the complexity of brain responses to transcranial and intracranial stimulations. *Brain Stimulation, 12(5)*, 1280–1289. [https://doi.org/10.1016/j.brs.2019.05.013](https://doi.org/10.1016/j.brs.2019.05.013)

> [Nigel Rogasch’s TESA toolbox pipeline overview](https://nigelrogasch.gitbook.io/tesa-user-manual/example_pipelines)

And the TMS-artifact removal + ICA classification steps are guided by the open-source TESA toolbox:

> Rogasch NC, Sullivan C, Thomson RH, Rose NS, Bailey NW, Fitzgerald PB, Farzan F, Hernandez-Pavon JC. Analysing concurrent transcranial magnetic stimulation and electroencephalographic data: a review and introduction to the open-source TESA software. *NeuroImage.* 2017; 147:934-951.  
> Mutanen TP, Biabani M, Sarvas J, Ilmoniemi RJ, Rogasch NC. Source-based artifact-rejection techniques available in TESA, an open-source TMS-EEG toolbox. *Brain Stimulation.* 2020; In press.

---

## Example Pipeline 

Below is the pipeline **I use**, after iterating a lot and verifying that the final EEG looks reasonable (contains typical TEPs and stable PCI-values). It’s primarily tested on recordings from one healthy subject (awake/slightly somnolent). **Use at your own risk**—always visually check data quality.

### 1. Initial Data Loading and Setup

1. **Load raw data** from NeurOne, EEGLAB, BrainVision, or other supported formats.  
   - Default data format: `'neurone'`  
   - Data directory must contain a `TMSEEG` folder.

2. **Set random seed** to ensure reproducibility (`42`).

3. **Create events** from the chosen stim channel (by default `STI 014`).  
   - If no stim channel is present, attempts to read from annotations or other common channel names.

4. **Remove non-EEG channels** (e.g., `EMG1`, etc.) to keep only EEG.

---

### 2. First-Pass TMS Artifact Removal

5. **Remove TMS artifact** (first pass).  
   - Cut window: **-2 to 10 ms** around the TMS pulse.  
   - Replace with zeros (or equivalent marking).

6. **Interpolate TMS artifact** (first pass).  
   - Interpolation method: **cubic**  
   - Interpolation window: **1.0 ms**  
   - Same cut times: **-2 to 10 ms**

---

### 4. Initial Downsampling

8. **Downsample** from the original sampling rate (e.g., 5000 Hz, 2000 Hz) to an **initial frequency** (by default **1000 Hz**) for faster processing. 

---

### 5. Epoching and Bad Data Removal

9. **Create epochs** around each TMS event.  
   - Time window: **-0.41 to 0.41 s**  
   - Amplitude threshold: **4500 µV** (user-defined; you can change)  
   - Currently, baseline is set to `None` (the actual correction is done later).

10. **Remove bad channels** automatically (FASTER-based or similar).  
    - Threshold: **3** (default in `run.py`)

11. **Remove bad epochs** automatically (FASTER or similar).  
    - Threshold: **3** (default in `run.py`)

12. **Set average reference** across remaining channels.

---

### 6. First ICA for TMS-Muscle Artifacts

13. **First ICA run** (commonly `FastICA`):  
                         
- tms_muscle_window=(11, 30), tms_muscle_thresh=2,
         blink_thresh=2.5,
         lat_eye_thresh=2.0,
         muscle_freq_window=(30, 100),
         muscle_freq_thresh=0.6,
         noise_thresh=4.0, 
  - Optionally **manual** or **automatic** component classification by thresholds.  

14. **(Optional) Clean muscle artifacts** with PARAFAC (not used by default).  
    - Typical window: 5–30 ms  
    - Threshold factor: 1.0  
    - Up to 5 PARAFAC components

---

### 7. Second-Pass TMS Artifact Removal

15. **Remove TMS artifact** (second pass).  
    - Extended cut window: **-2 to 15 ms**

16. **Interpolate TMS artifact** (second pass).  
    - Interpolation method: **cubic**  
    - Interpolation window: **5.0 ms**  
    - Cut times: **-2 to 15 ms**

---

### 8. Filter Epoched Data

17. **Filter the epoched data** if raw wasn’t filtered earlier.  
    - Default band-pass: 0.1–45 Hz  
    - Notch filter: 50 Hz (width = 2 Hz)

---

### 9. Second ICA for Other Artifacts

18. **Second ICA run** (commonly `FastICA`).  
    - Same as above 
    -   - Optionally **manual** or **automatic** component classification by thresholds.  


19. **(Optional) Apply SSP**  
    - Typically 2 EEG components if used (we do **not** use by default).

---

### 10. Baseline Correction, CSD, and Final Downsampling

20. **Baseline correction**  
    - Window: **-400 to -50 ms** (applies after epoching but before final downsampling).

21. ** (Optional) CSD transformation**  
    - `lambda2` = 1e-3  
    - `stiffness` = 3

22. **Final downsampling**  
    - Target frequency: **725 Hz** (default).

---

### 11. (Optional) TEP Validation

23. **TEP validation**  
    - Peak prominence threshold: 0.01 (adjust as needed)  
    - We check GMFA or ROI waveforms to ensure TEP presence.

24. **Generate evoked response**  
    - Time window for plotting: -0.3 to 0.3 s  
    - Y-limits (µV): -2 to 2 (adjust as needed)

---

### 12. PCI\_st Calculation

25. **Calculate PCI\_st**  
    - Response window: 0 to 299 ms  
    - `k` = 1.2  
    - Minimum SNR: 1.1  
    - Maximum variance: 99.0%  
    - 100 steps for threshold optimization  

---

## Notes on Quality Control
- During each step, we generate simple QC metrics (e.g., channel retention, epoch retention, ICA components removed).  
- *Always visually inspect final TMS-EEG waveforms* and confirm TEP latencies/amplitudes are physiologically reasonable.  
- PCI\_st is only meaningful if the data are relatively artifact-free and well-epoched.

---

## Order of steps actually used by me


1. Load data  
2. Set seed, find/create events  
3. Drop unused channels (e.g., EMG)  
4. **(First TMS artifact removal)** -2 to 10 ms  
5. **(First interpolation)** cubic, 1.0 ms
6. **(Initial downsampling)** → 1000 Hz  
7. **Create epochs** (-0.41 to 0.41)  
8. **Remove bad channels** (threshold=3)  
9. **Remove bad epochs** (threshold=3)  
10. **Average reference**  
11. **First ICA** (FastICA, threshold=3.0)  
12. **(Optional) Clean muscle (PARAFAC)**  
13. **(Second TMS artifact removal)** -2 to 15 ms  
14. **(Second interpolation)** cubic, 5 ms  
15. **(Filter epoched data)** if raw not filtered  
16. **Second ICA** (Infomax or label-based)  
17. **(Optional) SSP**  
18. **Baseline correction** (-400 to -50 ms)
19. **Final downsampling** (725 Hz)  
20. **(Optional) TEP validation**  
21. **Plot evoked**  
22. **PCIst**  


### TMSArtifactCleaner (which might work)

The TMSArtifactCleaner class is designed to detect and clean transcranial magnetic stimulation (TMS)-evoked muscle artifacts in EEG/MEG data using tensor decomposition techniques. It uses the tensorly library for tensor operations and mne for handling eeg data. and was inspired by the article by Tangwiriyasakul et al., 2019. 

Tangwiriyasakul, C., Premoli, I., Spyrou, L., Chin, R. F., Escudero, J., & Richardson, M. P. (2019). Tensor decomposition of TMS-induced EEG oscillations reveals data-driven profiles of antiepileptic drug effects. Scientific Reports, 9(1). https://doi.org/10.1038/s41598-019-53565-9

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

The PARAFAC decomposition was modellled after:

- Tangwiriyasakul, C., Premoli, I., Spyrou, L., Chin, R. F., Escudero, J., & Richardson, M. P. (2019). Tensor decomposition of TMS-induced EEG oscillations reveals data-driven profiles of antiepileptic drug effects. Scientific Reports, 9(1). https://doi.org/10.1038/s41598-019-53565-9

Custom functions are modelled after: 

- Rogasch NC, Sullivan C, Thomson RH, Rose NS, Bailey NW, Fitzgerald PB, Farzan F, Hernandez-Pavon JC. Analysing concurrent transcranial magnetic stimulation and electroencephalographic data: a review and introduction to the open-source TESA software. NeuroImage. 2017; 147:934-951.

- Mutanen TP, Biabani M, Sarvas J, Ilmoniemi RJ, Rogasch NC. Source-based artifact-rejection techniques available in TESA, an open-source TMS-EEG toolbox. Brain Stimulation. 2020; In press.
