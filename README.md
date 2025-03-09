# tmseegpy

This repository contains an attempt at building some sort of pipeline for preprocessing and analyzing Transcranial Magnetic Stimulation (TMS)-EEG data using python. I have attempted to implement some of the functionality found in TESA (https://github.com/nigelrogasch/TESA) which has been my guide and benchmark for the development. 

The pipeline includes steps for TMS artifact removal (code by Silvia Casarotto), filtering, Independent Component Analysis (ICA), muscle artifact removal (using Tensorly), and analysis of Perturbational Complexity Index based on State transitions (PCIst) (Comolatti et al., 2019). The analysis of PCIst is just a copy paste from https://github.com/renzocom/PCIst/blob/master/PCIst/pci_st.py which is written by Renzo Comolatti. The code is mostly adapted from a very long jupyter notebook which used mostly native MNE-Python methods which I expanded to a toolbox that I have been using in my analyses. 


## Installation (Git)

1. Clone the repository:

   ```bash
   git clone https://github.com/LazyCyborg/tmseegpy.git
   cd tmseegpy
   ``` 
   
2. Create a virtual environment (very recommended)

- For more info see: https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/

   ```bash
   conda env create -f environment.yml
  conda activate tmseegpy-env
   ```

3. Install the package:

   ```bash
   pip install .
   ```
   

## Installation (pip)

   ```bash
    pip install tmseegpy
   ``` 


## GUI Application (Recommended)

The GUI application (HePoTEP) provides an interactive way to load data, configure preprocessing steps, visualize data at each stage, and run the analysis.

### Running the GUI:

After installing tmseegpy, you can run the graphical interface.

Open a terminal or command prompt and run:

   ```bash
   tmseegpy-gui
   ```

### Command-Line Arguments
Run:

   ```bash
  tmseegpy process --help
   ```

for full list of command line arguments and default values. The CLI-version is mostly built so that experienced user can run batch processing of large datasets.

Note that, if run in fully automatic mode the code uses MNE-FASTER for both channel and epoch rejection and uses either a simple classification model (trained with Tensorflow) that was trained on ca 1200 TMS-EEG ICA components or my own classification algorithm which classifies components based on the topography and number of peaks in the components. This means that a lot of cortical activity can be 
removed and artifacts can remain which probably makes the final result unreliable due to the low SNR of TEPs. However, it might be useful as a quick first pass of a large dataset. 

### Example Usage

To run the pipeline with default settings with manual ICA component selection using MNEs QT viewer and PyQt6:

```bash
tmseegpy process --data_dir ./data_dir_with_TMSEEG_folder --output_dir ./your_output_dir
```

To run and for example disable ICA preprocessing and PCIst plots 

```bash
tmseegpy process --data_dir ./data_dir_with_TMSEEG_folder --output_dir ./your_output_dir --no_first_ica --no_second_ica --no_pcist
```

To enable PARAFAC muscle artifact removal :

```bash
tmseegpy process --data_dir ./data_dir_with_TMSEEG_folder --output_dir ./your_output_dir --parafac_muscle_artifacts
```

To enable saving of eeg data in .fif format during preprocessing for quality checks (files will be saved in a steps directory):

```bash
tmseegpy process --data_dir ./data_dir_with_TMSEEG_folder --output_dir ./your_output_dir --save_preproc
```

3. Use the scripts:

It is also possible to use the separate parts of the pipeline in a Jupyter Notebook or similar. I mostly run it like this for debugging, and it is probably impractical to run an entire analysis pipeline like this. Additionally there might be some issues when running the ICAs in manual component selection mode.

Example:

```Python
from tmseegpy.preproc import TMSEEGPreprocessor
from tmseegpy.preproc import detect_tms_artifacts

preproc = TMSEEGPreprocessor(raw=raw)

events = detect_tms_artifacts(raw=raw)

preproc.fix_tms_artifact(events=events)


```
## Data Preparation for batch processing through the CLI

If running the code through the CLI your data should be organized in the following structure:

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

Or if using Brainvision or other formats that are directly compatible with MNE-Python
```
data/
└── TMSEEG/
    ├── session1.vhdr
    │    
    └──session2.vhdr/

```

# Processing Pipeline

Below is the pipeline that I use when preprocessing sample data. These steps are still modelled after the recommendations in:

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
2. Find/create events
3. Drop unused channels (e.g., EMG)
4. Remove TMS artefact using baseline data (window: -5 - 2ms)
5. Filter raw EEG data (high-pass 0.1 Hz, low-pass: 250 Hz)
6. **Create epochs** (-0.8 to 0.8)  
7. **Average reference**  
8. **Remove bad channels** (manual or threshold=3)  
9. **Remove bad epochs** (manual or threshold=3)
10. **First ICA** (FastICA)  
11. **(Optional and very experimental) PARAFAC decomposition**
12. **Filter epoched data (high-pass 1 hz, low-pass 45 Hz and notch filter 50 Hz)**
13. **Second ICA** (Infomax)
14. **Downsampling** (725 Hz)  
15. **TEP plotting**
16. **PCIst**  


### PARAFAC decomposition (which might work)

This step is designed to detect and remove TMS-evoked muscle artifacts in EEG data using tensor decomposition techniques. It uses the tensorly library for tensor operations and mne for handling eeg data. and was inspired by the article by Tangwiriyasakul et al., 2019. 

Tangwiriyasakul, C., Premoli, I., Spyrou, L., Chin, R. F., Escudero, J., & Richardson, M. P. (2019). Tensor decomposition of TMS-induced EEG oscillations reveals data-driven profiles of antiepileptic drug effects. Scientific Reports, 9(1). https://doi.org/10.1038/s41598-019-53565-9

### Special Thanks to:

- **Dr. Silvia Casarotto** for kindly sharing code and verifying the preprocessing output
- **Dr. Renzo Comolatti for sharing the code for PCIst calculation**
- **Dr. Nigel Rogasch** for sanctioning the adaptation of TESA in Python
- **Dr. Mats Svantesson** (Linköping University Hospital) for many hours of assistance with code, signal processing, and EEG data verification
- **Dr. Magnus Thordstein** (Linköping University Hospital) for providing access to TMS and TMS-EEG equipment for sample data collection
- **Dr. Andrew Wold** for teaching me how to use the TMS equipment
- **Gramfort et al.** for creating MNE-Python, which this program is built upon
- **The creator of this repository https://github.com/heilerich/neurone_loader for creating the NeurOne loader**

  This project would not have been possible to complete without the support and contributions of these individuals.

## License

This project is licensed under the MIT License.

## References

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
