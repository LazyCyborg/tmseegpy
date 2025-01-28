# tmseegpy

This repository contains my attempt at building some sort of pipeline for preprocessing and analyzing Transcranial Magnetic Stimulation (TMS)-EEG data using python. I ahve attempted to implement some of the functionality found in TESA (https://github.com/nigelrogasch/TESA) which has been my guide and benchmark for the development. 

The pipeline includes steps for artifact removal, filtering, Independent Component Analysis (ICA), muscle artifact cleaning (using Tensorly), and analysis of Perturbational Complexity Index based on State transitions (PCIst) (Comolatti et al., 2019). The analysis of PCIst is just a copy paste from https://github.com/renzocom/PCIst/blob/master/PCIst/pci_st.py which is written by Renzo Comolatti. The code is mostly adapted from a very long jupyter notebook which used mostly native MNE-Python methods which I expanded to a toolbox that I have been using in my analyses. So the code base might not be very efficient. 

If you have trouble with the current dataloader and creates one that is compatible with multiple systems (maybe out of frustration) feel free to reach out to hjarneko@gmail.com. The package uses a modified version of the neurone_loader (https://github.com/heilerich/neurone_loader) to load the data from the Bittium NeurOne and convert it to an MNE-Python raw object. 

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

## GUI Application

A graphical user interface is available in the `tmseegpy-gui-react` directory. 
To use the GUI version:

### Download Releases
Download the latest GUI release for your platform from:
https://github.com/LazyCyborg/tmseegpy/releases (look for GUI releases tagged with `gui-v*`)

#### Installation

1. Install the TMSeegpy package:
   - Copy the `tmseegpy` directory to your preferred location
   - Add this location to your system's PATH

2. Install the GUI:
   - Copy the TMSeegpy GUI application to your Applications folder (Mac) or Program Files (Windows)

#### Usage

1. Start the TMSeegpy server:
   ```bash
   tmseegpy server
   ```

2. Launch the TMSeegpy GUI application.

The GUI will automatically connect to the running server. If the connection fails, ensure the server is running and retry the connection using the GUI's retry button.

The GUI is basically a wrapper for the argparser bellow and is intended as the main way to test the pipeline in a clinical setting. 

### Command-Line Arguments
Se run.py (bottom of the file) for full list of configurable command line arguments

### Example Usage

To run the pipeline with default settings:

```bash
cd tmseegpy
```

```bash
python run.py --data_dir ./data_dir_with_TMSEEG_folder --output_dir ./your_output_dir --first_ica_manual --second_ica_manual
```

To enable PARAFAC muscle artifact removal :

```bash
python run.py --data_dir ./data_dir_with_TMSEEG_folder --output_dir ./your_output_dir --parafac_muscle_artifacts
```

To enable plotting of eeg data during preprocessing for quality checks (plots will be saved in a steps directory):

```bash
python run.py --data_dir ./data_dir_with_TMSEEG_folder --output_dir ./your_output_dir --plot_preproc
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

## Notes on Quality Control
- *Always visually inspect final TMS-EEG waveforms* and confirm TEP latencies/amplitudes are physiologically reasonable.  
- PCI\_st is only meaningful if the data are relatively artifact-free and well-epoched.

---

## Order of steps 

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
16. **Second ICA** (FastICA)  
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

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License.

## Acknowledgements

The PCIst implementation is based on:

>  Comolatti, R., Pigorini, A., Casarotto, S., Fecchio, M., Faria, G., Sarasso, S., Rosanova, M., Gosseries, O., Boly, M., Bodart, O., Ledoux, D., Brichant, J. F., Nobili, L., Laureys, S., Tononi, G., Massimini, M., & Casali, A. G. (2019). A fast and general method to empirically estimate the complexity of brain responses to transcranial and intracranial stimulations. Brain Stimulation, 12(5), 1280–1289. https://doi.org/10.1016/j.brs.2019.05.013

The pipeline uses the MNE-Python library for EEG data processing:

>  Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7 DEC. https://doi.org/10.3389/fnins.2013.00267

The bad channel, and epoch detection uses MNE-FASTER:

>  Nolan H, Whelan R, Reilly RB. FASTER: Fully Automated Statistical Thresholding for EEG artifact Rejection. J Neurosci Methods. 2010 Sep 30;192(1):152-62. doi: 10.1016/j.jneumeth.2010.07.015. Epub 2010 Jul 21. PMID: 20654646.

The PARAFAC decomposition was modellled after:

>  Tangwiriyasakul, C., Premoli, I., Spyrou, L., Chin, R. F., Escudero, J., & Richardson, M. P. (2019). Tensor decomposition of TMS-induced EEG oscillations reveals data-driven profiles of antiepileptic drug effects. Scientific Reports, 9(1). https://doi.org/10.1038/s41598-019-53565-9

Custom functions are modelled after: 

>  Rogasch NC, Sullivan C, Thomson RH, Rose NS, Bailey NW, Fitzgerald PB, Farzan F, Hernandez-Pavon JC. Analysing concurrent transcranial magnetic stimulation and electroencephalographic data: a review and introduction to the open-source TESA software. NeuroImage. 2017; 147:934-951.

>  Mutanen TP, Biabani M, Sarvas J, Ilmoniemi RJ, Rogasch NC. Source-based artifact-rejection techniques available in TESA, an open-source TMS-EEG toolbox. Brain Stimulation. 2020; In press.
