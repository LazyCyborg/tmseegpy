# app.py
import os
import streamlit as st
import mne
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tmseegpy.validate_tep import plot_tep_analysis, plotly_evoked
from data_loader import DataLoader
from data_viewer import DataViewer
from tmseegpy.preproc import TMSEEGPreprocessor


@dataclass
class ProcessingState:
    """Class to track the state of processing"""
    # Data objects
    raw: Optional[mne.io.Raw] = None
    epochs: Optional[mne.Epochs] = None
    events: Optional[np.ndarray] = None
    first_ica: Optional[mne.preprocessing.ICA] = None
    second_ica: Optional[mne.preprocessing.ICA] = None

    # Parameters
    output_dir: str = 'output'
    session_name: str = 'session'

    # Processing state flags
    data_loaded: bool = False
    events_created: bool = False
    channels_dropped: bool = False
    raw_filtered: bool = False
    tms_removed: bool = False
    epochs_created: bool = False
    bad_channels_removed: bool = False
    bad_epochs_removed: bool = False
    avg_ref_applied: bool = False
    first_ica_done: bool = False
    muscle_cleaned: bool = False
    second_tms_removed: bool = False
    second_interpolation_done: bool = False
    epochs_filtered: bool = False
    second_ica_done: bool = False
    ssp_applied: bool = False
    downsampled: bool = False
    tep_analyzed: bool = False,
    data_saved: bool = False

    # Selected steps tracking
    selected_steps: Dict[str, bool] = field(default_factory=lambda: {
        'filter': False,
        'create_events': False,
        'tms_removed': False,
        'create_epochs': False,
        'reject_bad_channels': False,
        'reject_bad_epochs': False,
        'first_ica': False,
        'clean_muscle': False,
        'filter_epochs': False,
        'second_ica': False,
        'ssp': False,
        'downsample': False,
        'tep_analysis': False,
        'save_data': False

    })


class TMSEEGApp:
    """Main application class for TMS-EEG processing"""

    def __init__(self):
        self.processor = None
        self.data_loader = DataLoader()
        self.data_viewer = DataViewer()

        # Initialize session state
        if 'processing_state' not in st.session_state:
            st.session_state.processing_state = ProcessingState()

        # Initialize TMSEEGPreprocessor if raw data exists
        if hasattr(st.session_state.processing_state, 'raw') and st.session_state.processing_state.raw is not None:
            self.processor = TMSEEGPreprocessor(st.session_state.processing_state.raw)

    def initialize_processor(self):
        """Initialize or reinitialize the TMSEEGPreprocessor"""
        if st.session_state.processing_state.raw is not None:
            self.processor = TMSEEGPreprocessor(st.session_state.processing_state.raw)
            return True
        return False



    def render_pipeline_interface(self):
        """Render the pipeline interface with all options visible"""
        if not st.session_state.processing_state.data_loaded:
            st.warning("Please load data first")
            return


        pipeline_steps = {
            'filter': {
                'name': 'Filter Raw EEG',
                'description': 'Apply bandpass filter (should only be run once)',
                'method': self.render_filtering,
                'status_key': 'raw_filtered'
            },
            'create_events': {
                'name': 'Create Events',
                'description': 'Create events from stimulus channel or annotations',
                'method': self.render_create_events,
                'status_key': 'events_created'
            },
            'tms_removal': {  # Add your new step
                'name': 'Remove TMS Artifact',
                'description': 'Remove TMS artifacts and interpolate',
                'method': self.render_tms_removal,
                'status_key': 'tms_removed'
            },
            'create_epochs': {
                'name': 'Create Epochs',
                'description': 'Create epochs around events',
                'method': self.render_epoch_creation,
                'status_key': 'epochs_created'
            },
            'reject_bad_channels': {
                'name': 'Reject Bad Channels',
                'description': 'Requires epochs to be created first',
                'method': self.render_bad_channel_rejection,
                'status_key': 'bad_channels_removed'
            },
            'reject_bad_epochs': {
                'name': 'Reject Bad Epochs',
                'description': 'Requires epochs to be created first',
                'method': self.render_bad_epoch_rejection,
                'status_key': 'bad_epochs_removed'
            },
            'first_ica': {
                'name': 'ICA 1',
                'description': 'Run ICA for artifact removal',
                'method': self.render_first_ica,
                'status_key': 'first_ica_done'
            },
            'clean_muscle': {
                'name': 'PARAFAC Decomposition',
                'description': 'Remove muscle artifacts using automatic classification',
                'method': self.render_clean_muscle,
                'status_key': 'muscle_cleaned'
            },
            'filter_epochs': {
                'name': 'Filter Epochs',
                'description': 'Apply additional filtering to epochs',
                'method': self.render_filter_epochs,
                'status_key': 'epochs_filtered'
            },
            'second_ica': {
                'name': 'ICA 2',
                'description': 'Run second ICA with native MNE GUI',
                'method': self.render_second_ica,
                'status_key': 'second_ica_done'
            },
            'ssp': {
                'name': 'SSP',
                'description': 'Apply SSP for noise reduction',
                'method': self.render_ssp,
                'status_key': 'ssp_applied'
            },
            'downsample': {
                'name': 'Downsample',
                'description': 'Downsample epochs to final sampling rate',
                'method': self.render_downsample,
                'status_key': 'downsampled'
            },
            'tep_analysis': {
                'name': 'TEP Analysis',
                'description': 'Analyze TMS-Evoked Potentials',
                'method': self.render_tep_analysis,
                'status_key': 'tep_analyzed'
            }
        }

        # Create tabs only once
        tabs = st.tabs([step_info['name'] for step_info in pipeline_steps.values()])

        # Show all step options in their respective tabs
        with tabs[0]:
            cols = st.columns(1)
            if st.session_state.processing_state.raw_filtered:
                st.success("✅ Completed")
            elif 'filter' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['filter']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_filtering()

        with tabs[1]:
            cols = st.columns(1)
            if st.session_state.processing_state.events_created:
                st.success("✅ Completed")
            elif 'create_events' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['create_events']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_create_events()

        with tabs[2]:
            cols = st.columns(1)
            if st.session_state.processing_state.tms_removed:
                st.success("✅ Completed")
            elif 'tms_removed' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['tms_removed']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_tms_removal()

        with tabs[3]:
            cols = st.columns(1)
            if st.session_state.processing_state.epochs_created:
                st.success("✅ Completed")
            elif 'create_epochs' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['create_epochs']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_epoch_creation()

        with tabs[4]:
            cols = st.columns(1)
            if st.session_state.processing_state.bad_channels_removed:
                st.success("✅ Completed")
            elif 'reject_bad_channels' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['reject_bad_channels']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_bad_channel_rejection()

        with tabs[5]:
            cols = st.columns(1)
            if st.session_state.processing_state.bad_epochs_removed:
                st.success("✅ Completed")
            elif 'reject_bad_epochs' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['reject_bad_epochs']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_bad_epoch_rejection()

        with tabs[6]:
            cols = st.columns(1)
            if st.session_state.processing_state.first_ica_done:
                st.success("✅ Completed")
            elif 'first_ica' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['first_ica']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_first_ica()

        with tabs[7]:
            cols = st.columns(1)
            if st.session_state.processing_state.muscle_cleaned:
                st.success("✅ Completed")
            elif 'clean_muscle' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['clean_muscle']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_clean_muscle()

        with tabs[8]:
            cols = st.columns(1)
            if st.session_state.processing_state.epochs_filtered:
                st.success("✅ Completed")
            elif 'filter_epochs' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['filter_epochs']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_filter_epochs()

        with tabs[9]:
            cols = st.columns(1)
            if st.session_state.processing_state.second_ica_done:
                st.success("✅ Completed")
            elif 'second_ica' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['second_ica']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_second_ica()

        with tabs[10]:
            cols = st.columns(1)
            if st.session_state.processing_state.ssp_applied:
                st.success("✅ Completed")
            elif 'ssp' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['ssp']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_ssp()

        with tabs[11]:
            cols = st.columns(1)
            if st.session_state.processing_state.downsampled:
                st.success("✅ Completed")
            elif 'downsample' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['downsample']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_downsample()

        with tabs[12]:
            cols = st.columns(1)
            if st.session_state.processing_state.tep_analyzed:
                st.success("✅ Completed")
            elif 'tep_analysis' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['tep_analysis']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_tep_analysis()

        with tabs[13]:
            cols = st.columns(1)
            if st.session_state.processing_state.data_saved:
                st.success("✅ Completed")
            elif 'save_data' in st.session_state.processing_state.selected_steps and \
                    st.session_state.processing_state.selected_steps['save_data']:
                st.success("✅ Completed")
            else:
                st.warning("⏳ Pending")
            self.render_save_data()

        # Add run button at the bottom
        st.markdown("---")
        '''
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Selected Steps", key="run_pipeline"):
                # Run only the selected steps
                if st.session_state.get("run_filter"):
                    self.render_filtering()
                if st.session_state.get("run_events"):
                    self.render_create_events()
                if st.session_state.get("remove_tms"):
                    self.render_tms_removal()
                if st.session_state.get("run_epochs"):
                    self.render_epoch_creation()
                if st.session_state.get("run_bad_channels"):
                    self.render_bad_channel_rejection()
                if st.session_state.get("run_bad_epochs"):
                    self.render_bad_epoch_rejection()
                if st.session_state.get("run_first_ica"):
                    self.render_first_ica()
                if st.session_state.get("run_clean_muscle"):
                    self.render_clean_muscle()
                if st.session_state.get("run_filter_epochs"):
                    self.render_filter_epochs()
                if st.session_state.get("run_second_ica"):
                    self.render_second_ica()
                if st.session_state.get("run_ssp"):
                    self.render_ssp()
                if st.session_state.get("run_downsample"):
                    self.render_downsample()
                if st.session_state.get("run_tep_analysis"):
                    self.render_tep_analysis()'''
        col1, col2 = st.columns(2)
        with col2:
            if st.button("Reset Pipeline", key="reset_pipeline"):
                st.session_state.processing_state = ProcessingState()
                self.processor = None
                st.rerun()

    def render_pipeline_status(self):
        """Render the pipeline status in the sidebar"""
        state = st.session_state.processing_state

        status_items = [
            ("Data Loaded", state.data_loaded),
            ("Raw EEG Filtered", state.raw_filtered),
            ("Events Created", state.events_created),
            ("TMS Removal", state.tms_removed),
            ("Epochs Created", state.epochs_created),
            #("Channels Dropped", state.channels_dropped),
            ("Bad Channels Removed", state.bad_channels_removed),
            ("Bad Epochs Removed", state.bad_epochs_removed),
            ("First ICA", state.first_ica_done),
            ("PARAFAC", state.muscle_cleaned),
            ("Epochs Filtered", state.epochs_filtered),
            ("Second ICA", state.second_ica_done),
            ("SSP Applied", state.ssp_applied),
            ("Downsampled", state.downsampled),
            ("Data Saved", state.data_saved)
        ]

        for label, status in status_items:
            icon = "✅" if status else "⭕"
            st.text(f"{icon} {label}")

    def run(self):
        """Run the application"""
        st.title("HePoTEP (Heuristic Processing of TMS-Evoked Potentials)")

        st.info("""
        💡 **Data Visualization Available Throughout Processing**
        You can view your data at any time during the pipeline by expanding the "Data Visualization Options" section:
        - View Raw Data: Examine the raw EEG signals
        - View PSD: Check the Power Spectral Density
        - View Epochs: Inspect individual epochs (available after epoch creation)
        """)

        with st.expander("⭐️Credits and Acknowledgements from the Author", expanded=False):
            st.markdown("""
            ### Acknowledgements

            This pipeline includes various processing steps from several sources:

            - **Artifact removal** methods written by **Silvia Casarotto**
            - **Independent Component Analysis (ICA)** adapted from **Nigel Rogasch's TESA toolbox**, which served as the main inspiration and benchmark for this code
            - **Muscle artifact removal** (using Tensorly) inspired by **Tangwiriyasakul et al., 2019**

            ### Special Thanks

            - **Dr. Silvia Casarotto** for kindly sharing code and verifying the preprocessing output
            - **Dr. Nigel Rogasch** for sanctioning the adaptation of TESA in Python
            - **Dr. Mats Svantesson** (Linköping University Hospital) for many hours of assistance with code, signal processing, and EEG data verification
            - **Dr. Magnus Thordstein** (Linköping University Hospital) for providing access to TMS and TMS-EEG equipment for sample data collection
            - **Dr. Andrew Wold, PhD** for teaching me how to use the TMS equipment
            - **Gramfort et al.** for creating MNE-Python, which this program is built upon

            This project would not have been possible to complete without the support and contributions of these individuals.
            
            Author:
            Alexander Engelmark 
            """)
        try:
            # Add sidebar for global settings and status
            with st.sidebar:
                st.header("Pipeline Status")
                self.render_pipeline_status()
                if st.button("Reset Pipeline"):
                    st.session_state.processing_state = ProcessingState()
                    self.processor = None
                    st.rerun()

            # Main content area
            if not st.session_state.processing_state.data_loaded:
                self.render_load_data()
            else:
                # Debug information
                st.write("Debug: Data is loaded")

                # Data information section
                st.subheader("Data Information")
                self.data_viewer.display_data_info(st.session_state.processing_state.raw)

                with st.expander("Data Visualization Options", expanded=False):
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        if st.button("View Raw Data"):
                            self.data_viewer.view_raw(st.session_state.processing_state.raw)
                    with col2:
                        if st.button("View PSD"):
                            self.data_viewer.plot_psd(st.session_state.processing_state.raw)
                    with col3:
                        if st.button("View Epochs"):
                            self.data_viewer.view_epochs(st.session_state.processing_state.epochs)
                    with col4:
                        if st.button("View First ICA"):
                            if st.session_state.processing_state.first_ica_done:
                                self.data_viewer.view_ica_components(
                                    st.session_state.ica_state['current_ica'],
                                    st.session_state.processing_state.epochs
                                )
                            else:
                                st.warning("First ICA hasn't been run yet")
                    with col5:
                        if st.button("View Second ICA"):
                            if st.session_state.processing_state.second_ica_done:
                                self.data_viewer.view_ica_components(
                                    st.session_state.processing_state.second_ica,
                                    st.session_state.processing_state.epochs
                                )
                            else:
                                st.warning("Second ICA hasn't been run yet")

                # Force pipeline interface to show
                st.markdown("---")
                st.header("Processing Pipeline")
                #st.write("Debug: About to render pipeline interface")
                self.render_pipeline_interface()
                #st.write("Debug: Pipeline interface should be visible")

        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.error("Detailed error information:")
            import traceback
            st.code(traceback.format_exc())

    def standardize_channel_names(self, raw):
        """Standardize channel names to match common conventions"""
        rename_dict = {}
        for ch in raw.ch_names:
            # Skip non-EEG channels
            if any(ch.upper().startswith(x) for x in ['EMG', 'ECG', 'EOG', 'TRIG', 'STI']):
                continue

            # Common channel name standardizations
            if ch.upper() == 'FP1':
                rename_dict[ch] = 'Fp1'
            elif ch.upper() == 'FP2':
                rename_dict[ch] = 'Fp2'
            elif ch.upper() == 'FPZ':
                rename_dict[ch] = 'Fpz'
            elif ch.upper() == 'FPOZ':
                rename_dict[ch] = 'Fpz'
            elif ch.upper() == 'POZ':
                rename_dict[ch] = 'POz'
            elif ch.upper() == 'PZ':
                rename_dict[ch] = 'Pz'
            elif ch.upper() == 'OZ':
                rename_dict[ch] = 'Oz'
            elif ch.upper() == 'FZ':
                rename_dict[ch] = 'Fz'
            elif ch.upper() == 'CZ':
                rename_dict[ch] = 'Cz'
            elif ch.upper() == 'FCZ':
                rename_dict[ch] = 'FCz'
            elif ch.upper() == 'CPZ':
                rename_dict[ch] = 'CPz'
            # Special case for your problematic 'IZ' channel
            elif ch.upper() == 'IZ':
                rename_dict[ch] = 'Iz'  # Standard name for inion electrode

        if rename_dict:
            raw.rename_channels(rename_dict)

        return raw

    def render_load_data(self):
        """Render load data interface"""
        st.write("Load Data")
        st.info("""
        💾 **Supported Data Formats**:
        - BrainVision (.vhdr, .eeg, .vmrk)
        - NeurOne (directory containing .ses and recordings)
        - Curry (.cdt, .cef, .dat, .dap, .rs3, .cdt.dpa, .cdt.cef)
        - Other MNE formats (.fif, .set, .edf, .bdf)
        """)

        # Output directory selection
        output_dir = st.text_input(
            "Output Directory",
            value=st.session_state.processing_state.output_dir
        )
        session_name = st.text_input(
            "Session Name",
            value=st.session_state.processing_state.session_name
        )

        # Montage selection - Add this section
        st.subheader("Montage and Channel Configuration")

        # List of available montages
        montage_options = [
            'standard_1005', 'standard_1020', 'standard_alphabetic',
            'standard_postfixed', 'standard_prefixed', 'standard_primed',
            'biosemi16', 'biosemi32', 'biosemi64', 'biosemi128', 'biosemi160',
            'easycap-M1', 'easycap-M10', 'easycap-M43',
            'GSN-HydroCel-32', 'GSN-HydroCel-64_1.0', 'GSN-HydroCel-129',
            'mgh60', 'mgh70'
        ]

        selected_montage = st.selectbox(
            "EEG Montage",
            options=montage_options,
            index=montage_options.index('standard_1020'),
            help="Select the montage that matches your EEG cap setup"
        )

        # Channel renaming option
        auto_rename_channels = st.checkbox(
            "Auto-rename channels to match standard nomenclature",
            value=True,
            help="Automatically rename channel names to match standard naming conventions"
        )

        # Warning about channel numbers
        ch_warning_msg = """
        ⚠️ **Important**: Make sure your montage matches your actual electrode count.
        If using 32 channels, select a 32-channel montage (e.g., biosemi32).
        If using 64 channels, select a 64-channel montage (e.g., biosemi64).
        """
        st.info(ch_warning_msg)

        # Set missing channel handling
        missing_channel_option = st.radio(
            "Handling of missing channels",
            options=["Error", "Warn", "Ignore"],
            index=1,
            help="How to handle channels in your data that aren't in the selected montage",
            horizontal=True
        )
        missing_channel_handling = missing_channel_option.lower()

        # Data format selection
        data_format = st.radio(
            "Select Data Format",
            options=['NeurOne', 'BrainVision', 'Curry', 'Other MNE Formats'],
            horizontal=True,
            index=1
        )

        if data_format == 'NeurOne':
            # NeurOne loading code remains unchanged
            data_path = st.text_input("NeurOne Data Directory Path")

            if st.button("Load NeurOne Data") and data_path:
                try:
                    raw_list, metadata_df = self.data_loader.load_neurone_data(data_path)

                    if raw_list and metadata_df is not None:
                        # Store the first raw object
                        raw = raw_list[0]

                        # Apply channel renaming and montage
                        if auto_rename_channels:
                            raw = self.standardize_channel_names(raw)

                        # Apply selected montage
                        try:
                            montage = mne.channels.make_standard_montage(selected_montage)
                            raw.set_montage(montage, on_missing=missing_channel_handling)
                        except Exception as e:
                            st.warning(f"Montage application warning: {str(e)}")

                        # Initialize processor with the loaded raw data
                        self.processor = TMSEEGPreprocessor(raw)

                        # Update session state
                        st.session_state.processing_state.raw = raw
                        st.session_state.processing_state.data_loaded = True
                        st.session_state.processing_state.output_dir = output_dir
                        st.session_state.processing_state.session_name = session_name

                        # Display data info
                        st.success("NeurOne data loaded successfully!")
                        self.render_pipeline_interface()
                        self.data_viewer.display_data_info(raw)

                        # Add visualization options
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("View Raw Data"):
                                self.data_viewer.view_raw(raw)
                        with col2:
                            if st.button("View PSD"):
                                self.data_viewer.plot_psd(raw)

                except Exception as e:
                    st.error(f"Error loading NeurOne data: {str(e)}")

        elif data_format == 'BrainVision':
            # BrainVision loading code remains unchanged
            st.info("For BrainVision files, please enter the directory containing the .vhdr, .eeg, and .vmrk files.")

            brainvision_dir = st.text_input("BrainVision Directory Path")

            if brainvision_dir and os.path.isdir(brainvision_dir):
                # List all .vhdr files in the directory
                vhdr_files = [f for f in os.listdir(brainvision_dir) if f.lower().endswith('.vhdr')]

                if not vhdr_files:
                    st.warning(f"No BrainVision header (.vhdr) files found in the specified directory.")
                else:
                    # Let user select which .vhdr file to load
                    selected_vhdr = st.selectbox("Select BrainVision dataset to load:", vhdr_files)

                    if selected_vhdr:
                        # Check if corresponding .eeg and .vmrk files exist
                        base_name = os.path.splitext(selected_vhdr)[0]
                        eeg_file = f"{base_name}.eeg"
                        vmrk_file = f"{base_name}.vmrk"

                        vhdr_path = os.path.join(brainvision_dir, selected_vhdr)
                        eeg_path = os.path.join(brainvision_dir, eeg_file)
                        vmrk_path = os.path.join(brainvision_dir, vmrk_file)

                        missing_files = []
                        if not os.path.exists(eeg_path):
                            missing_files.append(eeg_file)
                        if not os.path.exists(vmrk_path):
                            missing_files.append(vmrk_file)

                        if missing_files:
                            st.warning(f"Missing required BrainVision files: {', '.join(missing_files)}")
                        else:
                            # All files exist, allow loading
                            if st.button("Load BrainVision Data"):
                                try:
                                    raw = mne.io.read_raw_brainvision(
                                        vhdr_path,
                                        preload=True,
                                        eog=('HEOGL', 'HEOGR', 'VEOGb'),
                                        misc='auto'
                                    )

                                    # Apply channel renaming and montage
                                    if auto_rename_channels:
                                        raw = self.standardize_channel_names(raw)

                                    # Apply selected montage
                                    try:
                                        montage = mne.channels.make_standard_montage(selected_montage)
                                        raw.set_montage(montage, on_missing=missing_channel_handling)
                                    except Exception as e:
                                        st.warning(f"Montage application warning: {str(e)}")

                                    # Initialize processor with the loaded raw data
                                    self.processor = TMSEEGPreprocessor(raw)

                                    # Update session state
                                    st.session_state.processing_state.raw = raw
                                    st.session_state.processing_state.data_loaded = True
                                    st.session_state.processing_state.output_dir = output_dir
                                    st.session_state.processing_state.session_name = session_name

                                    st.success(f"BrainVision data loaded successfully from {vhdr_path}!")
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"Error loading BrainVision data: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
            elif brainvision_dir:
                st.error(f"Directory does not exist: {brainvision_dir}")

        elif data_format == 'Curry':
            # Curry file loading section
            st.info("""
            For Curry files, please enter the directory containing Curry data files.

            Typically you should select the main data file (usually a .cdt or .rs3 file).
            Associated files (.dap, .cef, etc.) in the same directory will be automatically detected.
            """)

            curry_dir = st.text_input("Curry Data Directory Path")

            if curry_dir and os.path.isdir(curry_dir):
                # Look for main Curry data files (.cdt, .rs3) in the directory
                curry_main_files = [f for f in os.listdir(curry_dir) if
                                    any(f.lower().endswith(ext) for ext in
                                        ['.cdt', '.rs3', '.dat', '.dap', '.cdt.dpa', '.cdt.cef'])]

                if not curry_main_files:
                    st.warning("No Curry data files found in the specified directory.")
                else:
                    # Let user select which file to load
                    selected_curry = st.selectbox("Select Curry dataset to load:", curry_main_files)

                    if selected_curry:
                        curry_path = os.path.join(curry_dir, selected_curry)

                        # Allow loading
                        if st.button("Load Curry Data"):
                            try:
                                raw = mne.io.read_raw_curry(
                                    curry_path,
                                    preload=True
                                )

                                # Apply channel renaming and montage
                                if auto_rename_channels:
                                    raw = self.standardize_channel_names(raw)

                                # Apply selected montage
                                try:
                                    montage = mne.channels.make_standard_montage(selected_montage)
                                    raw.set_montage(montage, on_missing=missing_channel_handling)
                                except Exception as e:
                                    st.warning(f"Montage application warning: {str(e)}")

                                # Initialize processor with the loaded raw data
                                self.processor = TMSEEGPreprocessor(raw)

                                # Update session state
                                st.session_state.processing_state.raw = raw
                                st.session_state.processing_state.data_loaded = True
                                st.session_state.processing_state.output_dir = output_dir
                                st.session_state.processing_state.session_name = session_name

                                st.success(f"Curry data loaded successfully from {curry_path}!")
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error loading Curry data: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
            elif curry_dir:
                st.error(f"Directory does not exist: {curry_dir}")

        else:  # Other MNE Formats
            # Update the list of supported formats
            curry_exts = ['.cdt', '.cef', '.dat', '.dap', '.rs3', '.cdt.dpa', '.cdt.cef']
            other_formats = [ext for ext in self.data_loader.supported_formats['MNE']
                             if ext != '.vhdr' and ext not in curry_exts]

            uploaded_file = st.file_uploader(
                "Upload EEG data file",
                type=other_formats
            )

            if uploaded_file is not None:
                try:
                    raw = self.data_loader.load_mne_data(uploaded_file)

                    # Apply channel renaming and montage
                    if auto_rename_channels:
                        raw = self.standardize_channel_names(raw)

                    # Apply selected montage
                    try:
                        montage = mne.channels.make_standard_montage(selected_montage)
                        raw.set_montage(montage, on_missing=missing_channel_handling)
                    except Exception as e:
                        st.warning(f"Montage application warning: {str(e)}")

                    if raw is not None:
                        # Initialize processor with the loaded raw data
                        self.processor = TMSEEGPreprocessor(raw)

                        # Update session state
                        st.session_state.processing_state.raw = raw
                        st.session_state.processing_state.data_loaded = True
                        st.session_state.processing_state.output_dir = output_dir
                        st.session_state.processing_state.session_name = session_name

                        st.success("Data loaded successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")

    def render_save_data(self):
        """Render interface for saving processed data"""
        st.write("Save Processed Data")

        if not st.session_state.processing_state.data_loaded:
            st.warning("Please load and process data before saving")
            return

        # Get output directory and session name from session state
        output_dir = st.session_state.processing_state.output_dir
        session_name = st.session_state.processing_state.session_name

        # Display current output settings
        st.info(f"Current output directory: {output_dir}")
        st.info(f"Current session name: {session_name}")

        # Option to update output directory
        new_output_dir = st.text_input(
            "Update output directory (optional)",
            value=output_dir
        )

        # Option to update session name
        new_session_name = st.text_input(
            "Update session name (optional)",
            value=session_name
        )

        # Update session state if changed
        if new_output_dir != output_dir or new_session_name != session_name:
            st.session_state.processing_state.output_dir = new_output_dir
            st.session_state.processing_state.session_name = new_session_name
            output_dir = new_output_dir
            session_name = new_session_name
            st.success("Output settings updated")

        # Create the full output path
        import os
        full_output_path = os.path.join(output_dir, session_name)

        # Check what can be saved based on processing state
        can_save_raw = st.session_state.processing_state.raw is not None
        can_save_epochs = st.session_state.processing_state.epochs is not None
        can_save_evoked = can_save_epochs  # If we have epochs, we can compute evoked

        # Create checkboxes for selecting what to save
        st.subheader("Select Data to Save")

        save_raw = st.checkbox("Save preprocessed raw data", value=can_save_raw, disabled=not can_save_raw)
        save_epochs = st.checkbox("Save preprocessed epochs", value=can_save_epochs, disabled=not can_save_epochs)
        save_evoked = st.checkbox("Save evoked response", value=can_save_evoked, disabled=not can_save_evoked)

        # File format options
        st.subheader("File Format Options")

        raw_format = st.selectbox(
            "Raw data format",
            options=["FIF (.fif)", "European Data Format (.edf)", "BrainVision (.vhdr)"],
            index=0,
            disabled=not save_raw
        )

        epochs_format = st.selectbox(
            "Epochs format",
            options=["FIF (.fif)"],
            index=0,
            disabled=not save_epochs
        )

        evoked_format = st.selectbox(
            "Evoked format",
            options=["FIF (.fif)", "CSV (.csv)"],
            index=0,
            disabled=not save_evoked
        )

        # Save button
        if st.button("Save Data"):
            try:
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)

                # Counter for saved files
                saved_count = 0

                # Save raw data if selected
                if save_raw and can_save_raw:
                    if raw_format == "FIF (.fif)":
                        raw_fname = os.path.join(output_dir, f"{session_name}_raw.fif")
                        st.session_state.processing_state.raw.save(raw_fname, overwrite=True)
                    elif raw_format == "European Data Format (.edf)":
                        raw_fname = os.path.join(output_dir, f"{session_name}_raw.edf")
                        st.session_state.processing_state.raw.export(raw_fname, fmt='edf', overwrite=True)
                    elif raw_format == "BrainVision (.vhdr)":
                        raw_fname = os.path.join(output_dir, f"{session_name}_raw.vhdr")
                        st.session_state.processing_state.raw.export(raw_fname, fmt='brainvision', overwrite=True)

                    st.success(f"Raw data saved to {raw_fname}")
                    saved_count += 1

                # Save epochs if selected
                if save_epochs and can_save_epochs:
                    epochs_fname = os.path.join(output_dir, f"{session_name}_epo.fif")
                    st.session_state.processing_state.epochs.save(epochs_fname, overwrite=True)
                    st.success(f"Epochs saved to {epochs_fname}")
                    saved_count += 1

                # Save evoked if selected
                if save_evoked and can_save_evoked:
                    # Compute evoked from epochs
                    evoked = st.session_state.processing_state.epochs.average()

                    if evoked_format == "FIF (.fif)":
                        evoked_fname = os.path.join(output_dir, f"{session_name}_ave.fif")
                        evoked.save(evoked_fname, overwrite=True)
                    elif evoked_format == "CSV (.csv)":
                        evoked_fname = os.path.join(output_dir, f"{session_name}_ave.csv")
                        # Convert to pandas DataFrame and save as CSV
                        import pandas as pd
                        data_dict = {
                            'time': evoked.times
                        }
                        for i, ch_name in enumerate(evoked.ch_names):
                            data_dict[ch_name] = evoked.data[i]
                        df = pd.DataFrame(data_dict)
                        df.to_csv(evoked_fname, index=False)

                    st.success(f"Evoked response saved to {evoked_fname}")
                    saved_count += 1

                # Final success message
                if saved_count > 0:
                    st.success(f"Successfully saved {saved_count} file(s) to {output_dir}")
                    # Update the session state to mark save as completed
                    st.session_state.processing_state.data_saved = True
                    st.session_state.processing_state.selected_steps['save_data'] = True
                else:
                    st.warning("No data was selected for saving")

            except Exception as e:
                st.error(f"Error saving data: {str(e)}")
                st.error("Detailed error information:")
                import traceback
                st.code(traceback.format_exc())

    def render_create_events(self):
        """Render event creation interface"""
        st.write("Create Events")

        if not st.session_state.processing_state.data_loaded:
            st.warning("Please load data first")
            return

        if self.processor is None:
            if not self.initialize_processor():
                st.error("Could not initialize processor")
                return

        try:
            # Event creation settings
            st.subheader("Event Creation Settings")

            # Method selection
            event_method = st.radio(
                "Event Detection Method",
                options=['Threshold', 'Manual Annotation', 'From Existing Events'],
                horizontal=True
            )

            if event_method == 'Threshold':
                col1, col2 = st.columns(2)

                with col1:
                    # Stim channel selection
                    stim_channels = [ch for ch in self.processor.raw.ch_names
                                     if ch.upper().startswith(('STIM', 'TRG', 'TRIG', 'STI 014'))]
                    if not stim_channels:
                        stim_channels = self.processor.raw.ch_names

                    stim_channel = st.selectbox(
                        "Stimulus Channel",
                        options=stim_channels,
                        index=0 if stim_channels else None
                    )

                    threshold = st.number_input(
                        "Threshold Value",
                        value=0.5,
                        min_value=0.0,
                        max_value=1.0,
                        step=0.1
                    )

                with col2:
                    min_duration = st.number_input(
                        "Minimum Duration (ms)",
                        value=1,
                        min_value=1,
                        max_value=1000,
                        step=1
                    )

                    consecutive = st.checkbox(
                        "Consecutive Events",
                        value=False,
                        help="Allow consecutive events without minimum time between them"
                    )

            elif event_method == 'Manual Annotation':
                st.info("Please use MNE's raw viewer to add annotations manually (Click Data Visualization above)")
                if st.button("Open Raw Viewer for Annotation"):
                    self.data_viewer.view_raw(self.processor.raw)


            elif event_method == 'From Existing Events':

                if hasattr(self.processor.raw, 'annotations') and len(self.processor.raw.annotations) > 0:

                    st.info(f"Found {len(self.processor.raw.annotations)} existing annotations")
                    _, event_id = mne.events_from_annotations(self.processor.raw)
                    event_types = list(event_id.keys())
                    st.write("Existing event types:", event_types)
                    # Option to select specific event types
                    selected_events = st.multiselect(
                        "Select Event Types to Include",
                        options=event_types,
                        default=event_types
                    )
                else:
                    st.warning("No existing events found in the data")
                    return

            # Run event creation
            if st.button("Create Events", key="create_events_button"):
                try:
                    with st.spinner("Creating events..."):
                        if event_method == 'Threshold':
                            # Create events using threshold method
                            events = mne.find_events(
                                self.processor.raw,
                                stim_channel=stim_channel,
                                consecutive=consecutive,
                                min_duration=min_duration / 1000.0,  # Convert to seconds
                                verbose=False
                            )

                        elif event_method == 'Manual Annotation':
                            # Convert annotations to events
                            events, event_id = mne.events_from_annotations(self.processor.raw)

                        else:  # From Existing Events

                        # Filter events based on selection
                            events, event_id = mne.events_from_annotations(self.processor.raw)
                            if selected_events:
                                # Create a list of event codes to keep
                                selected_codes = [event_id[evt] for evt in selected_events]
                                mask = np.isin(events[:, 2], selected_codes)
                                events = events[mask]

                        # Store events in processor and session state
                        self.processor.events = events
                        st.session_state.processing_state.events = events
                        st.session_state.processing_state.events_created = True

                        # Display event information
                        st.success(f"Created {len(events)} events")

                        plt.figure(figsize=(10, 4))
                        # Plot events
                        fig = mne.viz.plot_events(
                            events,
                            self.processor.raw.info['sfreq'],
                            self.processor.raw.first_samp,
                            show=False,
                        )

                        st.pyplot(fig)

                        # Show event timing statistics
                        if len(events) > 1:
                            intervals = np.diff(events[:, 0]) / self.processor.raw.info['sfreq']
                            st.write("Event timing statistics (seconds):")
                            st.write(f"Mean interval: {np.mean(intervals):.3f}")
                            st.write(f"Std interval: {np.std(intervals):.3f}")
                            st.write(f"Min interval: {np.min(intervals):.3f}")
                            st.write(f"Max interval: {np.max(intervals):.3f}")

                            # Plot interval histogram
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.hist(intervals, bins='auto')
                            ax.set_xlabel('Inter-event interval (s)')
                            ax.set_ylabel('Count')
                            ax.set_title('Event Interval Distribution')
                            st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error creating events: {str(e)}")
                    st.error("Detailed error information:")
                    import traceback
                    st.code(traceback.format_exc())

            # Option to clear events
            if st.session_state.processing_state.events_created:
                if st.button("Clear Events"):
                    self.processor.events = None
                    st.session_state.processing_state.events = None
                    st.session_state.processing_state.events_created = False
                    st.success("Events cleared")

        except Exception as e:
            st.error(f"Error in event creation interface: {str(e)}")
            st.error("Detailed error information:")
            import traceback
            st.code(traceback.format_exc())

    def render_epoch_creation(self):
        """Render epoch creation interface"""
        st.write("Create Epochs")

        if not st.session_state.processing_state.events_created:
            st.warning("Please create events first")
            return

        if self.processor is None:
            if not self.initialize_processor():
                st.error("Could not initialize processor")
                return

        try:
            # Epoch creation settings
            st.subheader("Epoch Settings")

            col1, col2 = st.columns(2)
            with col1:
                tmin = st.number_input(
                    "Start time (s)",
                    value=-0.8,
                    min_value=-1.0,
                    max_value=0.0,
                    step=0.1
                )
                baseline = st.text_input(
                    "Baseline period (start,end in seconds)",
                    value="None,None",
                    help="Example: -0.1,0 or None,None"
                )

            with col2:
                tmax = st.number_input(
                    "End time (s)",
                    value=0.8,
                    min_value=0.0,
                    max_value=2.0,
                    step=0.1
                )

            # Parse baseline input
            try:
                if baseline.lower() == "none,none":
                    baseline = None
                else:
                    baseline = tuple(map(float, baseline.split(',')))
            except:
                st.error("Invalid baseline format. Use 'start,end' or 'None,None'")
                return

            # Advanced settings
            with st.expander("Advanced Settings"):
                picks = st.multiselect(
                    "Pick channels (leave empty for all)",
                    options=self.processor.raw.ch_names,
                    default=[]
                )

                detrend = st.selectbox(
                    "Detrend",
                    options=[None, 0, 1],
                    format_func=lambda x: "None" if x is None else f"Order {x}",
                    help="0: constant, 1: linear, None: no detrending"
                )

                # Add reference settings
                ref_method = st.radio(
                    "Reference method",
                    options=['None', 'Average'],
                    horizontal=True,
                    help="Choose reference method for the epochs"
                )

                # Add option for automatic channel type detection and removal
                auto_detect_channels = st.checkbox(
                    "Automatically detect and remove non-EEG channels",
                    value=True,
                    help="Automatically identify and remove channels like EMG, ECG, EOG, etc."
                )

            # Create epochs button
            if st.button("Create Epochs"):
                try:
                    with st.spinner("Creating epochs..."):
                        # Prepare picks
                        if not picks:
                            picks = 'all'

                        # Create epochs
                        epochs = mne.Epochs(
                            st.session_state.processing_state.raw,
                            self.processor.events,
                            tmin=tmin,
                            tmax=tmax,
                            baseline=baseline,
                            picks=picks,
                            preload=True,
                            detrend=detrend,
                            verbose=False
                        )

                        # Apply average reference if selected
                        if ref_method == 'Average':
                            epochs.set_eeg_reference('average')
                            st.info("Applied average reference")

                        # Automatic channel type detection and removal
                        if auto_detect_channels:
                            # Get list of all channels
                            all_channels = epochs.ch_names

                            # Define patterns for non-EEG channels
                            non_eeg_patterns = [
                                'EMG', 'ECG', 'EOG', 'HEOG', 'VEOG',
                                'Trigger', 'TRIG', 'Status', 'TIME',
                                'EMG1', 'EMG2', 'ECG1', 'ECG2'
                            ]

                            # Find channels to drop
                            channels_to_drop = []
                            for channel in all_channels:
                                if any(pattern.lower() in channel.lower() for pattern in non_eeg_patterns):
                                    channels_to_drop.append(channel)

                            if channels_to_drop:
                                epochs.drop_channels(channels_to_drop)
                                st.info(f"Dropped non-EEG channels: {', '.join(channels_to_drop)}")

                        # Store epochs in processor and session state
                        self.processor.epochs = epochs
                        st.session_state.processing_state.epochs = epochs
                        st.session_state.processing_state.epochs_created = True

                        # Display epoch information
                        st.success(f"Created {len(epochs)} epochs")
                        st.write("Epoch Information:")
                        st.write(f"Number of epochs: {len(epochs)}")
                        st.write(f"Number of channels: {len(epochs.ch_names)}")
                        st.write(f"Time range: {epochs.times[0]:.3f} to {epochs.times[-1]:.3f} seconds")

                        # Plot drop log
                        if st.button("Plot Drop Log", key="plot_drop_log_epochs"):
                            fig = epochs.plot_drop_log(show=False)
                            st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error creating epochs: {str(e)}")
                    st.error("Detailed error information:")
                    import traceback
                    st.code(traceback.format_exc())

            # Option to clear epochs
            if st.session_state.processing_state.epochs_created:
                if st.button("Clear Epochs"):
                    self.processor.epochs = None
                    st.session_state.processing_state.epochs = None
                    st.session_state.processing_state.epochs_created = False
                    st.success("Epochs cleared")

        except Exception as e:
            st.error(f"Error in epoch creation interface: {str(e)}")
            st.error("Detailed error information:")
            import traceback
            st.code(traceback.format_exc())

        # Show status if epochs have been created
        if st.session_state.processing_state.epochs_created:
            st.info("✅ Epochs have been created")

    def render_filtering(self):
        """Render filtering interface"""
        st.write("Filter Settings")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            l_freq = st.number_input("Low cutoff (Hz)", value=1.0, step=0.1)
        with col2:
            h_freq = st.number_input("High cutoff (Hz)", value=250.0, step=0.1)
        with col3:
            notch_freqs_input = st.text_input(
                "Notch frequencies",
                value="50",
                help="For IIR: Enter single frequency (e.g., '50')\nFor FIR/spectrum_fit: Enter start stop step (e.g., '50 251 50')"
            )
        with col4:
            filter_design = st.text_input(
                "Design of the filter (fir, iir or spectrum_fit)",
                value="iir",
                help="Note: IIR only accepts single frequency"
            )

        # Convert string input based on filter design
        try:
            if filter_design.lower() == 'iir':
                notch_freqs = float(notch_freqs_input)
                st.info(f"Will apply highpass filter at {l_freq}, lowpass filter at {h_freq} and notch filter at: {notch_freqs} Hz")
            else:
                start, stop, step = map(float, notch_freqs_input.split())
                notch_freqs = np.arange(start, stop, step)
                st.info(f"Will apply highpass filter at {l_freq}, lowpass filter at {h_freq} and notch filters at: {notch_freqs} Hz")
        except ValueError:
            if filter_design.lower() == 'iir':
                st.error("Please enter a single frequency for IIR filter")
            else:
                st.error("Please enter three numbers: start stop step")
            notch_freqs = []

        if st.button("Apply Filter", key="apply_filter_button"):
            try:
                if self.processor is None:
                    self.processor = TMSEEGPreprocessor(st.session_state.processing_state.raw)

                with st.spinner("Applying filters..."):
                    # First apply bandpass filter
                    self.processor.raw.filter(
                        l_freq=l_freq,
                        h_freq=h_freq,
                        method=filter_design,
                        iir_params=dict(
                            order=3,
                            ftype='butter',
                            phase='zero-double',
                            btype='bandpass'
                        ),
                        verbose=True
                    )

                    # Then apply notch filters if frequencies are specified
                    if notch_freqs is not None and (
                            isinstance(notch_freqs, (list, np.ndarray)) or isinstance(notch_freqs, (int, float))):
                        self.processor.raw.notch_filter(
                            freqs=notch_freqs,
                            picks='eeg',
                            method=filter_design,
                            verbose=True
                        )


                    # Update the raw data in session state
                    st.session_state.processing_state.raw = self.processor.raw
                    st.session_state.processing_state.raw_filtered = True

                    # Show success message and plots
                    st.success("Data filtered successfully!")

                    # Show before/after PSD comparison
                    st.subheader("Power Spectral Density After filtering")

                    # Plot filtered PSD
                    fig = self.processor.raw.compute_psd(fmax=250).plot(show=False)

                    st.pyplot(fig)
                    plt.close()

                    # Option to view filtered data
                    if st.button("View Filtered Data"):
                        self.data_viewer.view_raw(self.processor.raw)

                    # Store the step completion in session state
                    st.session_state.processing_state.selected_steps['filter'] = True

            except Exception as e:
                st.error(f"Error during filtering: {str(e)}")
                st.error("Detailed error information:")
                import traceback
                st.code(traceback.format_exc())

        # Show status if filtering has been applied
        if st.session_state.processing_state.raw_filtered:
            st.info("✅ Filtering has been applied")

    def render_tms_removal(self):
        """Render TMS artifact removal interface"""
        from tmseegpy.preproc import detect_tms_artifacts

        st.write("TMS Artifact Removal Settings")

        # Store original data for comparison
        if 'original_raw_tms' not in st.session_state:
            st.session_state.original_raw_tms = st.session_state.processing_state.raw.copy()

        # Add event selection options
        if hasattr(self.processor, 'events') and self.processor.events is not None:
            # Get unique event codes
            unique_codes = np.unique(self.processor.events[:, 2])

            # If we have event_id mapping, use it to show meaningful names
            event_names = {}
            if hasattr(self.processor, 'event_id') and self.processor.event_id is not None:
                # Invert the mapping to go from codes to names
                code_to_name = {code: name for name, code in self.processor.event_id.items()}
                event_names = {code: code_to_name.get(code, f"Event {code}") for code in unique_codes}
            else:
                event_names = {code: f"Event {code}" for code in unique_codes}

            # Create a multiselect for event types
            st.subheader("Select Event Types for TMS Removal")
            selected_event_codes = st.multiselect(
                "Event Types",
                options=list(event_names.keys()),
                default=list(event_names.keys()),
                format_func=lambda x: event_names[x]
            )
        else:
            st.warning("No events have been created. Please create events first.")
            selected_event_codes = None

        # Add configuration options
        with st.expander("Advanced Settings", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                window_start = st.number_input(
                    "Window Start (ms)",
                    value=-2,
                    min_value=-20,
                    max_value=0,
                    help="Start time of the removal window relative to TMS pulse"
                )

                smooth_window_start = st.number_input(
                    "Smooth Window Start (ms)",
                    value=-2,
                    min_value=-20,
                    max_value=0,
                    help="Start time of the smoothing window relative to TMS pulse"
                )

            with col2:
                window_end = st.number_input(
                    "Window End (ms)",
                    value=5,
                    min_value=0,
                    max_value=10,
                    help="End time of the removal window relative to TMS pulse"
                )

                smooth_window_end = st.number_input(
                    "Smooth Window End (ms)",
                    value=2,
                    min_value=0,
                    max_value=10,
                    help="End time of the smoothing window relative to TMS pulse"
                )

            span = st.slider(
                "Smoothing Span",
                min_value=1,
                max_value=5,
                value=2,
                help="Number of adjacent points to use for smoothing"
            )

            verbose = st.checkbox(
                "Verbose Output",
                value=True,
                help="Show detailed processing information"
            )

        # Add detection settings
        with st.expander("TMS Pulse Detection Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                threshold_std = st.number_input(
                    "Detection threshold (std)",
                    value=10.0,
                    min_value=1.0,
                    max_value=50.0,
                    step=0.5
                )
            with col2:
                min_distance_ms = st.number_input(
                    "Minimum distance (ms)",
                    value=50,
                    min_value=10,
                    max_value=500,
                    step=10
                )

        # Add both buttons side by side
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Remove TMS Artifact"):
                try:
                    if self.processor is None:
                        self.processor = TMSEEGPreprocessor(st.session_state.processing_state.raw)

                    with st.spinner("Removing TMS artifact..."):
                        window = (window_start / 1000, window_end / 1000)
                        smooth_window = (smooth_window_start / 1000, smooth_window_end / 1000)

                        events_to_use = None
                        if hasattr(self.processor, 'events') and self.processor.events is not None:
                            if selected_event_codes:
                                mask = np.isin(self.processor.events[:, 2], selected_event_codes)
                                events_to_use = self.processor.events[mask]
                                if verbose:
                                    st.write(
                                        f"Using {len(events_to_use)} events out of {len(self.processor.events)} total events")
                            else:
                                events_to_use = self.processor.events

                        self.processor.fix_tms_artifact(
                            window=window,
                            smooth_window=smooth_window,
                            span=span,
                            events=events_to_use,
                            verbose=verbose
                        )

                        st.session_state.processing_state.raw = self.processor.raw
                        st.session_state.processing_state.tms_removed = True
                        st.session_state.processing_state.selected_steps['tms_removed'] = True

                except Exception as e:
                    st.error(f"Error removing TMS artifact: {str(e)}")
                    st.error("Detailed error information:")
                    import traceback
                    st.code(traceback.format_exc())

        with col2:
            with st.expander("ℹ️ About TMS Pulse Detection", expanded=False):
                st.write("Creates new events for TMS pulses were there is no trigger (could be useful for paired pulse paradigms)")
            if st.button("Detect Additional TMS Pulses"):
                try:
                    with st.spinner("Detecting additional TMS pulses..."):
                        existing_events = self.processor.events if hasattr(self.processor, 'events') else None

                        additional_events = detect_tms_artifacts(
                            raw=self.processor.raw,
                            threshold_std=threshold_std,
                            min_distance_ms=min_distance_ms,
                            existing_events=existing_events
                        )

                        if additional_events is not None and len(additional_events) > 0:
                            combined_events = np.vstack((existing_events,
                                                         additional_events)) if existing_events is not None else additional_events
                            combined_events = combined_events[combined_events[:, 0].argsort()]

                            self.processor.events = combined_events
                            st.session_state.processing_state.events = combined_events
                            st.success(
                                f"Found {len(additional_events)} additional TMS pulses! Total events: {len(combined_events)}")
                        else:
                            st.info("No additional TMS pulses detected.")

                except Exception as e:
                    st.error(f"Error detecting additional TMS pulses: {str(e)}")

        # Show status if TMS artifact has been removed
        if st.session_state.processing_state.tms_removed:
            st.info("✅ TMS artifact removal has been completed")

    # Updates for bad_channel_rejection method in TMSEEGApp class

    def render_bad_channel_rejection(self):
        """Render bad channel rejection interface with manual rejection option"""
        st.write("Bad Channel Rejection")

        if not st.session_state.processing_state.epochs_created:
            st.warning("Please create epochs first (you can still plot the raw data and remove channels manually but MNE-FASTER requires epochs for automatic bad channel rejection")
            return

        if self.processor is None:
            if not self.initialize_processor():
                st.error("Could not initialize processor")
                return

        # Store original data for comparison and recovery if needed
        if 'original_channels' not in st.session_state and st.session_state.processing_state.epochs_created:
            st.session_state.original_channels = st.session_state.processing_state.epochs.ch_names.copy()

        # Track dropped channels
        if 'dropped_channels' not in st.session_state:
            st.session_state.dropped_channels = []

        # Create tabs for automatic and manual rejection
        auto_tab, manual_tab = st.tabs(["Automatic Rejection", "Manual Rejection"])

        with auto_tab:
            # Automatic rejection (existing code)
            with st.expander("Detection Settings", expanded=True):
                threshold = st.number_input(
                    "Detection threshold (Z-score)",
                    value=3.0,
                    min_value=1.0,
                    max_value=10.0,
                    help="Z-score threshold for bad channel detection",
                    key="bad_channels_threshold"
                )

                # Checkbox for interpolation
                interpolate = st.checkbox(
                    "Interpolate bad channels",
                    value=False,
                    help="If checked, bad channels will be interpolated instead of dropped using spline interpolation",
                    key="bad_channels_interpolate"
                )

            if st.button("Run Automatic Rejection", key="run_auto_rejection_bad_epochs"):
                try:
                    with st.spinner("Detecting and removing bad channels..."):
                        # Store channels before removal
                        channels_before = self.processor.epochs.ch_names.copy()

                        # Remove bad channels
                        self.processor.remove_bad_channels(
                            threshold=threshold, interpolate=interpolate)

                        # Update session state
                        st.session_state.processing_state.epochs = self.processor.epochs
                        st.session_state.processing_state.bad_channels_removed = True

                        # Track newly dropped channels
                        new_dropped = set(channels_before) - set(self.processor.epochs.ch_names)
                        st.session_state.dropped_channels.extend(list(new_dropped))

                        # Show results
                        st.success("Bad channel detection completed!")
                        if new_dropped:
                            st.write(f"Dropped channels: {list(new_dropped)}")

                        # Display channel statistics
                        st.subheader("Channel Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Original Channels",
                                len(channels_before)
                            )
                        with col3:
                            st.metric(
                                "Remaining Channels",
                                len(self.processor.epochs.ch_names)
                            )

                        # Store the step completion
                        st.session_state.processing_state.selected_steps['reject_bad_channels'] = True
                        st.session_state.processing_state.bad_channels_removed = True

                except Exception as e:
                    st.error(f"Error removing bad channels: {str(e)}")
                    st.error("Detailed error information:")
                    import traceback
                    st.code(traceback.format_exc())

        with manual_tab:
            st.subheader("Manual Channel Rejection")

            # Button to open data viewer
            if st.button("View Data for Channel Selection", key="view_data_channels"):
                if hasattr(self.processor, 'raw') and self.processor.raw is not None:
                    self.data_viewer.view_raw(self.processor.raw)
                elif hasattr(self.processor, 'epochs') and self.processor.epochs is not None:
                    self.data_viewer.view_epochs(self.processor.epochs)
                else:
                    st.warning("No data available to view")

            # Get current channels
            if hasattr(self.processor, 'epochs') and self.processor.epochs is not None:
                available_channels = self.processor.epochs.ch_names
            elif hasattr(self.processor, 'raw') and self.processor.raw is not None:
                available_channels = self.processor.raw.ch_names
            else:
                available_channels = []

            # Multi-select for manual channel rejection
            selected_channels = st.multiselect(
                "Select channels to reject",
                options=available_channels,
                key="manual_channels_to_reject"
            )

            # Add interpolation option
            interpolate_channels = st.checkbox(
                "Interpolate channels instead of dropping them",
                value=True,
                help="If checked, bad channels will be interpolated using data from neighboring channels"
            )

            # Display previously dropped channels
            if st.session_state.dropped_channels:
                st.info(f"Previously dropped channels: {st.session_state.dropped_channels}")

            # Button to apply manual rejection
            button_text = "Interpolate Selected Channels" if interpolate_channels else "Drop Selected Channels"
            if st.button(button_text):
                if not selected_channels:
                    st.warning("No channels selected")
                else:
                    try:
                        with st.spinner(
                                f"{'Interpolating' if interpolate_channels else 'Dropping'} selected channels..."):
                            # Process epochs if available
                            if hasattr(self.processor, 'epochs') and self.processor.epochs is not None:
                                if interpolate_channels:
                                    # Mark channels as bad
                                    self.processor.epochs.info['bads'].extend(selected_channels)

                                    # Interpolate bad channels
                                    self.processor.epochs.interpolate_bads(reset_bads=False)

                                    action_text = "interpolated"
                                else:
                                    # Drop channels
                                    self.processor.epochs.drop_channels(selected_channels)
                                    action_text = "dropped"

                                # Update session state
                                st.session_state.processing_state.epochs = self.processor.epochs
                                st.session_state.processing_state.bad_channels_removed = True

                                # Track affected channels
                                st.session_state.dropped_channels.extend(selected_channels)

                                st.success(f"Successfully {action_text} {len(selected_channels)} channels")
                                st.write(f"Affected channels: {selected_channels}")

                            # Process raw if epochs not available
                            elif hasattr(self.processor, 'raw') and self.processor.raw is not None:
                                if interpolate_channels:
                                    # Mark channels as bad
                                    self.processor.raw.info['bads'].extend(selected_channels)

                                    # Interpolate bad channels
                                    self.processor.raw.interpolate_bads(reset_bads=False)

                                    action_text = "interpolated"
                                else:
                                    # Drop channels
                                    self.processor.raw.drop_channels(selected_channels)
                                    action_text = "dropped"

                                # Update session state
                                st.session_state.processing_state.raw = self.processor.raw
                                st.session_state.processing_state.channels_dropped = True

                                # Track affected channels
                                st.session_state.dropped_channels.extend(selected_channels)

                                st.success(f"Successfully {action_text} {len(selected_channels)} channels")
                                st.write(f"Affected channels: {selected_channels}")
                            else:
                                st.error("No data available for channel processing")

                    except Exception as e:
                        st.error(f"Error processing channels: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            # Show status if bad channels have been removed
            if st.session_state.processing_state.bad_channels_removed:
                st.info("✅ Bad channel rejection has been completed")

    # Updates for bad_epoch_rejection method in TMSEEGApp class

    def render_bad_epoch_rejection(self):
        """Render bad epoch rejection interface with manual rejection option"""
        st.write("Bad Epoch Rejection")

        # Check if epochs exist
        if not st.session_state.processing_state.epochs_created:
            st.warning("Please create epochs first")
            return

        if self.processor is None or self.processor.epochs is None:
            st.warning("No epochs available. Please create epochs first")
            return

        # Store original epochs count only if epochs exist and count hasn't been stored
        if 'original_epochs_count' not in st.session_state and st.session_state.processing_state.epochs is not None:
            st.session_state.original_epochs_count = len(st.session_state.processing_state.epochs)

        # Track dropped epochs
        if 'dropped_epochs' not in st.session_state:
            st.session_state.dropped_epochs = []

        # Create tabs for automatic and manual rejection
        auto_tab, manual_tab = st.tabs(["Automatic Rejection", "Manual Rejection"])

        with auto_tab:
            # Advanced settings
            with st.expander("Rejection Settings", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    epoch_threshold = st.number_input(
                        "Threshold (std) for MNE-FASTER",
                        value=3.0,
                        help="Reject epochs exceeding this threshold"
                    )

                reject_by_annotation = st.checkbox(
                    "Reject by annotation",
                    value=False,
                    help="Reject epochs overlapping with annotations"
                )

            if st.button("Run Automatic Rejection"):
                try:
                    with st.spinner("Detecting and removing bad epochs..."):
                        # Store epochs count before rejection
                        epochs_before = len(self.processor.epochs)

                        # Remove bad epochs
                        self.processor.remove_bad_epochs(
                            threshold=epoch_threshold
                        )

                        # Update session state
                        st.session_state.processing_state.epochs = self.processor.epochs
                        st.session_state.processing_state.bad_epochs_removed = True

                        # Calculate rejection statistics
                        epochs_after = len(self.processor.epochs)
                        rejected_epochs = epochs_before - epochs_after
                        rejection_percentage = (rejected_epochs / epochs_before) * 100

                        # Keep track of which epochs were dropped (approximate - we just know how many)
                        st.session_state.dropped_epochs.append(f"{rejected_epochs} epochs from automatic rejection")

                        # Display results
                        st.success("Bad epoch rejection completed!")

                        # Show epoch statistics
                        st.subheader("Epoch Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Original Epochs",
                                epochs_before
                            )
                        with col2:
                            st.metric(
                                "Rejected Epochs",
                                rejected_epochs,
                                delta=f"-{rejected_epochs}"
                            )
                        with col3:
                            st.metric(
                                "Remaining Epochs",
                                epochs_after,
                                delta=f"-{rejection_percentage:.1f}%"
                            )

                        # Store the step completion
                        st.session_state.processing_state.selected_steps['bad_epoch_rejection'] = True

                except Exception as e:
                    st.error(f"Error removing bad epochs: {str(e)}")
                    st.error("Detailed error information:")
                    import traceback
                    st.code(traceback.format_exc())

        with manual_tab:
            st.subheader("Manual Epoch Rejection")

            # Button to open data viewer
            if st.button("View Epochs for Selection", key="view_epochs_manual"):
                if hasattr(self.processor, 'epochs') and self.processor.epochs is not None:
                    self.data_viewer.view_epochs(self.processor.epochs)
                else:
                    st.warning("No epochs available to view")

            # Get current epoch count
            if hasattr(self.processor, 'epochs') and self.processor.epochs is not None:
                num_epochs = len(self.processor.epochs)

                # Multi-select for manual epoch rejection
                selected_epochs = st.multiselect(
                    "Select epochs to reject (by index)",
                    options=list(range(num_epochs)),
                    key="manual_epochs_to_reject"
                )

                # Display previously dropped epochs
                if st.session_state.dropped_epochs:
                    st.info(f"Previously dropped: {', '.join(st.session_state.dropped_epochs)}")

                # Button to apply manual rejection
                if st.button("Drop Selected Epochs"):
                    if not selected_epochs:
                        st.warning("No epochs selected")
                    else:
                        try:
                            with st.spinner("Dropping selected epochs..."):
                                # Drop the selected epochs
                                self.processor.epochs.drop(selected_epochs)

                                # Update session state
                                st.session_state.processing_state.epochs = self.processor.epochs
                                st.session_state.processing_state.bad_epochs_removed = True

                                # Track dropped epochs
                                st.session_state.dropped_epochs.append(
                                    f"{len(selected_epochs)} epochs (indices: {selected_epochs})")

                                st.success(f"Successfully dropped {len(selected_epochs)} epochs")
                                st.write(f"Dropped epoch indices: {selected_epochs}")

                                # Show updated statistics
                                st.metric(
                                    "Remaining Epochs",
                                    len(self.processor.epochs),
                                    delta=f"-{len(selected_epochs)}"
                                )

                        except Exception as e:
                            st.error(f"Error dropping epochs: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            else:
                st.warning("No epochs available for rejection")

        # Show status if bad epochs have been removed
        if st.session_state.processing_state.bad_epochs_removed:
            st.info("✅ Bad epoch rejection has been completed")

    def render_first_ica(self):
        """Render first ICA interface with manual mode as default"""
        import traceback

        st.write("First ICA Settings")

        st.info("""
        Under "Data Visualization" above you can display the ICA sources as in mne.ica.plot_sources(). 
        It is RECOMMENDED to use this when selecting components since it allows you to right click on a 
        component to the left and see the corresponding FFT and topography""")

        # Initialize ICA state if not exists
        if 'ica_state' not in st.session_state:
            st.session_state.ica_state = {
                'ica_computed': False,
                'current_ica': None,
                'selected_components': []
            }

        # Check if epochs exist and are created
        if not st.session_state.processing_state.epochs_created:
            st.warning("Please create epochs first")
            return

        if self.processor is None or self.processor.epochs is None:
            st.warning("No epochs available. Please create epochs first")
            return

        # Store original epochs for comparison
        if 'pre_ica_epochs' not in st.session_state and st.session_state.processing_state.epochs is not None:
            st.session_state.pre_ica_epochs = st.session_state.processing_state.epochs.copy()

        # Calculate number of components automatically: good channels - 1
        total_channels = len(self.processor.epochs.ch_names)
        bad_channels = len(self.processor.epochs.info['bads'])
        good_channels = total_channels - bad_channels
        n_components = good_channels - 1

        # Display information about component calculation
        st.info(f"First ICA will use {n_components} components (good channels - 1)")
        st.write(f"Total channels: {total_channels}, Bad channels: {bad_channels}, Good channels: {good_channels}")

        # Advanced ICA settings
        with st.expander("ICA Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                ica_method = st.selectbox(
                    "ICA Method",
                    options=['fastica', 'picard', 'infomax'],
                    index=0,
                    help="Algorithm used for ICA computation"
                )
                # Note: n_components slider is removed and replaced with automatic calculation
            with col2:
                random_state = st.number_input(
                    "Random state",
                    value=42,
                    help="Random seed for reproducibility"
                )
                max_iter = st.number_input(
                    "Max iterations",
                    value=500,
                    min_value=100,
                    help="Maximum number of iterations"
                )

        # ICA mode selection with manual as default
        ica_mode = st.radio(
            "ICA Mode",
            options=['Manual', 'Automatic (Topography)', 'Automatic (Standard)'],
            index=0,
            help="Select method for component identification"
        )

        # Mode-specific settings
        if ica_mode == 'Automatic (Topography)':
            st.subheader("Topography Detection Settings")
            col1, col2 = st.columns(2)
            with col1:
                topo_edge_threshold = st.slider(
                    "Edge Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    help="Threshold for edge detection"
                )
                topo_zscore_threshold = st.slider(
                    "Z-score Threshold",
                    min_value=0.0,
                    max_value=5.0,
                    value=2.0,
                    help="Z-score threshold for component detection"
                )
            with col2:
                topo_peak_threshold = st.slider(
                    "Peak Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    help="Threshold for peak detection"
                )
                topo_focal_threshold = st.slider(
                    "Focal Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    help="Threshold for focal activity detection"
                )

        elif ica_mode == 'Automatic (Standard)':
            st.subheader("Automatic Detection Settings")
            col1, col2 = st.columns(2)
            with col1:
                tms_muscle_thresh = st.slider(
                    "TMS-Muscle Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    help="Threshold for TMS-muscle artifact detection"
                )
                blink_thresh = st.slider(
                    "Blink Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    help="Threshold for eye blink detection"
                )
            with col2:
                muscle_thresh = st.slider(
                    "Muscle Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    help="Threshold for muscle artifact detection"
                )
                noise_thresh = st.slider(
                    "Noise Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    help="Threshold for noise detection"
                )

        if st.button("Run First ICA"):
            try:
                if self.processor is None:
                    self.processor = TMSEEGPreprocessor(st.session_state.processing_state.raw)

                with st.spinner("Running ICA..."):
                    # Initialize ICA with automatically calculated number of components
                    ica = mne.preprocessing.ICA(
                        method=ica_method,
                        n_components=n_components,  # Now using the fixed value
                        random_state=random_state,
                        max_iter=max_iter
                    )

                    if ica_mode == 'Manual':
                        # Fit ICA
                        ica.fit(self.processor.epochs)

                        # Store ICA in both places
                        st.session_state.ica_state['current_ica'] = ica
                        st.session_state.ica_state['ica_computed'] = True
                        st.session_state.processing_state.first_ica = ica
                        st.session_state.processing_state.first_ica_done = True

                        st.success("ICA computation completed successfully!")
                        st.write(f"Number of components: {ica.n_components_}")

                        st.session_state.ica_state['ica_results'] = {  # Store ICA results in session state
                            'sources': ica.get_sources(self.processor.epochs),
                            'n_components': ica.n_components_,
                            'ica_object': ica  # Store the ICA object itself
                        }
                        st.rerun()  # Force rerun to display results

                    elif ica_mode == 'Automatic (Topography)':
                        self.processor.run_ica(
                            output_dir=st.session_state.processing_state.output_dir,
                            session_name=st.session_state.processing_state.session_name,
                            method=ica_method,
                            n_components=n_components,  # Now using the fixed value
                            use_topo=True,
                            topo_edge_threshold=topo_edge_threshold,
                            topo_zscore_threshold=topo_zscore_threshold,
                            topo_peak_threshold=topo_peak_threshold,
                            topo_focal_threshold=topo_focal_threshold
                        )
                        st.success("Automatic ICA (Topography) completed!")

                        # Store ICA references
                        st.session_state.ica_state['current_ica'] = self.processor.ica
                        st.session_state.ica_state['ica_computed'] = True
                        st.session_state.processing_state.first_ica = self.processor.ica
                        st.session_state.processing_state.first_ica_done = True

                        st.session_state.ica_state['ica_results'] = {  # Store ICA results in session state
                            'sources': self.processor.ica.get_sources(self.processor.epochs),
                            'n_components': self.processor.ica.n_components_,
                            'ica_object': self.processor.ica  # Store the ICA object itself
                        }
                        st.rerun()  # Force rerun to display results

                    else:  # Automatic (Standard)
                        self.processor.run_ica(
                            output_dir=st.session_state.processing_state.output_dir,
                            session_name=st.session_state.processing_state.session_name,
                            method=ica_method,
                            n_components=n_components,  # Now using the fixed value
                            tms_muscle_thresh=tms_muscle_thresh,
                            blink_thresh=blink_thresh,
                            muscle_thresh=muscle_thresh,
                            noise_thresh=noise_thresh,
                            manual_mode=False
                        )
                        st.success("Automatic ICA (Standard) completed!")

                        # Store ICA references
                        st.session_state.ica_state['current_ica'] = self.processor.ica
                        st.session_state.ica_state['ica_computed'] = True
                        st.session_state.processing_state.first_ica = self.processor.ica
                        st.session_state.processing_state.first_ica_done = True

                        st.session_state.ica_state['ica_results'] = {  # Store ICA results in session state
                            'sources': self.processor.ica.get_sources(self.processor.epochs),
                            'n_components': self.processor.ica.n_components_,
                            'ica_object': self.processor.ica  # Store the ICA object itself
                        }
                        st.rerun()  # Force rerun to display results

                    # Store the step completion
                    st.session_state.processing_state.selected_steps['first_ica'] = True

            except Exception as e:
                st.error(f"Error during first ICA: {str(e)}")
                st.error("Detailed error information:")
                st.code(traceback.format_exc())

        if 'ica_results' in st.session_state.ica_state:  # Display only if ICA results available
            self.display_ica_results('first_ica')

        # Show status if ICA has been completed
        if st.session_state.processing_state.first_ica_done:
            st.info("✅ First ICA has been completed")

    def render_second_ica(self):
        """Render second ICA interface with both manual and automatic modes"""
        import traceback
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import signal

        st.write("Second ICA Settings")

        st.info("""
        Under "Data Visulaization" above you can display the ICA sources as in mne.ica.plot_sources(). 
        It is RECOMMENDED to use this when selecting components since it allows you to right click on a 
        component to the left and see the corresponding FFT and topography""")

        # Initialize second ICA state if not exists
        if 'second_ica_state' not in st.session_state:
            st.session_state.second_ica_state = {
                'ica_computed': False,
                'current_ica': None,
                'selected_components': []
            }

        # Check if first ICA is completed
        if not st.session_state.processing_state.first_ica_done:
            st.warning("Please complete first ICA before proceeding")
            return

        if self.processor is None or self.processor.epochs is None:
            st.warning("No epochs available. Please complete first ICA")
            return

        # Store original epochs for comparison
        if 'pre_second_ica_epochs' not in st.session_state and st.session_state.processing_state.epochs is not None:
            st.session_state.pre_second_ica_epochs = st.session_state.processing_state.epochs.copy()

        # Advanced ICA settings
        with st.expander("ICA Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                ica_method = st.selectbox(
                    "ICA Method",
                    options=['fastica', 'picard', 'infomax'],
                    index=0,
                    help="Algorithm used for ICA computation",
                    key="second_ica_method"
                )
                n_components = st.slider(
                    "Number of components",
                    min_value=0,
                    max_value=len(st.session_state.processing_state.epochs.ch_names),
                    value=min(len(st.session_state.processing_state.epochs.ch_names), 20),
                    help="Number of ICA components to compute (0 = automatic)",
                    key="second_ica_components"
                )
            with col2:
                random_state = st.number_input(
                    "Random state",
                    value=42,
                    help="Random seed for reproducibility",
                    key="second_ica_random"
                )
                max_iter = st.number_input(
                    "Max iterations",
                    value=500,
                    min_value=100,
                    help="Maximum number of iterations",
                    key="second_ica_iter"
                )

        # ICA mode selection with manual as default
        ica_mode = st.radio(
            "ICA Mode",
            options=['Manual', 'Automatic (Topography)', 'Automatic (Standard)'],
            index=0,
            help="Select method for component identification",
            key="second_ica_mode"
        )

        # Mode-specific settings
        if ica_mode == 'Automatic (Topography)':
            st.subheader("Topography Detection Settings")
            col1, col2 = st.columns(2)
            with col1:
                topo_edge_threshold = st.slider(
                    "Edge Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    help="Threshold for edge detection",
                    key="second_topo_edge"
                )
                topo_zscore_threshold = st.slider(
                    "Z-score Threshold",
                    min_value=0.0,
                    max_value=5.0,
                    value=2.0,
                    help="Z-score threshold for component detection",
                    key="second_topo_zscore"
                )
            with col2:
                topo_peak_threshold = st.slider(
                    "Peak Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    help="Threshold for peak detection",
                    key="second_topo_peak"
                )
                topo_focal_threshold = st.slider(
                    "Focal Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    help="Threshold for focal activity detection",
                    key="second_topo_focal"
                )

        elif ica_mode == 'Automatic (Standard)':
            st.subheader("Automatic Detection Settings")
            col1, col2 = st.columns(2)
            with col1:
                tms_muscle_thresh = st.slider(
                    "TMS-Muscle Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    help="Threshold for TMS-muscle artifact detection",
                    key="second_tms_muscle"
                )
                blink_thresh = st.slider(
                    "Blink Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    help="Threshold for eye blink detection",
                    key="second_blink"
                )
            with col2:
                muscle_thresh = st.slider(
                    "Muscle Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    help="Threshold for muscle artifact detection",
                    key="second_muscle"
                )
                noise_thresh = st.slider(
                    "Noise Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    help="Threshold for noise detection",
                    key="second_noise"
                )

        if st.button("Run Second ICA", key="run_second_ica_button"):
            try:
                if self.processor is None:
                    self.processor = TMSEEGPreprocessor(st.session_state.processing_state.raw)

                with st.spinner("Running Second ICA..."):
                    # Initialize ICA
                    ica = mne.preprocessing.ICA(
                        method=ica_method,
                        n_components=n_components if n_components > 0 else None,
                        random_state=random_state,
                        max_iter=max_iter
                    )

                    if ica_mode == 'Manual':
                        # Fit ICA
                        ica.fit(self.processor.epochs)

                        # Store ICA results
                        st.session_state.second_ica_state['current_ica'] = ica
                        st.session_state.second_ica_state['ica_computed'] = True
                        st.session_state.processing_state.second_ica = ica
                        st.session_state.processing_state.second_ica_done = True

                        st.success("Second ICA computation completed successfully!")
                        st.write(f"Number of components: {ica.n_components_}")

                        st.session_state.second_ica_state['ica_results'] = {  # Store ICA results in session state
                            'sources': ica.get_sources(self.processor.epochs),
                            'n_components': ica.n_components_,
                            'ica_object': ica  # Store the ICA object itself
                        }
                        st.rerun()  # Force rerun to display results

                    elif ica_mode == 'Automatic (Topography)':
                        self.processor.run_ica(
                            output_dir=st.session_state.processing_state.output_dir,
                            session_name=st.session_state.processing_state.session_name,
                            method=ica_method,
                            n_components=n_components if n_components > 0 else None,
                            use_topo=True,
                            topo_edge_threshold=topo_edge_threshold,
                            topo_zscore_threshold=topo_zscore_threshold,
                            topo_peak_threshold=topo_peak_threshold,
                            topo_focal_threshold=topo_focal_threshold
                        )
                        st.success("Automatic Second ICA (Topography) completed!")

                        # Store ICA references
                        st.session_state.second_ica_state['current_ica'] = self.processor.ica
                        st.session_state.second_ica_state['ica_computed'] = True
                        st.session_state.processing_state.second_ica = self.processor.ica
                        st.session_state.processing_state.second_ica_done = True

                        st.session_state.second_ica_state['ica_results'] = {  # Store ICA results in session state
                            'sources': self.processor.ica.get_sources(self.processor.epochs),
                            'n_components': self.processor.ica.n_components_,
                            'ica_object': self.processor.ica  # Store the ICA object itself
                        }
                        st.rerun()  # Force rerun to display results

                    else:  # Automatic (Standard)
                        self.processor.run_ica(
                            output_dir=st.session_state.processing_state.output_dir,
                            session_name=st.session_state.processing_state.session_name,
                            method=ica_method,
                            n_components=n_components if n_components > 0 else None,
                            tms_muscle_thresh=tms_muscle_thresh,
                            blink_thresh=blink_thresh,
                            muscle_thresh=muscle_thresh,
                            noise_thresh=noise_thresh,
                            manual_mode=False
                        )
                        st.success("Automatic Second ICA (Standard) completed!")

                        # Store ICA references
                        st.session_state.second_ica_state['current_ica'] = self.processor.ica
                        st.session_state.second_ica_state['ica_computed'] = True
                        st.session_state.processing_state.second_ica = self.processor.ica
                        st.session_state.processing_state.second_ica_done = True

                        st.session_state.second_ica_state['ica_results'] = {  # Store ICA results in session state
                            'sources': self.processor.ica.get_sources(self.processor.epochs),
                            'n_components': self.processor.ica.n_components_,
                            'ica_object': self.processor.ica  # Store the ICA object itself
                        }
                        st.rerun()  # Force rerun to display results

                    # Store the step completion
                    st.session_state.processing_state.selected_steps['second_ica'] = True

            except Exception as e:
                st.error(f"Error during second ICA: {str(e)}")
                st.error("Detailed error information:")
                st.code(traceback.format_exc())

        if 'ica_results' in st.session_state.second_ica_state:  # Display only if ICA results available
            self.display_ica_results('second_ica')

        # Show status if Second ICA has been completed
        if st.session_state.processing_state.second_ica_done:
            st.info("✅ Second ICA has been completed")

    def display_ica_results(self, ica_key):
        """Displays ICA results and handles component selection."""
        import traceback
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import signal

        ica_state = st.session_state.ica_state if ica_key == 'first_ica' else st.session_state.second_ica_state
        results = ica_state.get('ica_results')  # Use .get() to avoid KeyError if 'ica_results' is not present

        if results is None:  # Handle the case where 'ica_results' is not in the session state
            st.warning(f"No ICA results found for {ica_key}. Please run ICA first.")
            return

        sources = results['sources']
        n_components = results['n_components']
        ica = results['ica_object']

        try:

            # Plot topomaps using matplotlib
            st.subheader("Component Topographies")

            ica_object = ica_state['ica_results']['ica_object']


            if ica_key == 'first_ica':
                epochs_for_topo = st.session_state.pre_ica_epochs
            else:
                epochs_for_topo = st.session_state.pre_second_ica_epochs

            if epochs_for_topo is None:
                st.warning(f"Epochs for topomap plotting not found for {ica_key}.")
                return

            # Use a copy of the info to avoid modifying the original
            info_for_topo = ica_object.info.copy()

            # Ensure montage is set for correct plotting
            montage = mne.channels.make_standard_montage('standard_1020')
            info_for_topo.set_montage(montage)

            components = ica_object.get_components()  # Get components from the ICA object
            n_cols = 5
            n_rows = int(np.ceil(n_components / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

            # Handle cases with only one component
            if n_components == 1:
                axes = [axes]
            else:
                axes = axes.ravel()

            for i in range(n_components):
                mne.viz.plot_topomap(components[:, i], info_for_topo, axes=axes[i], show=False)  # Use info_for_topo
                axes[i].set_title(f'IC{i:02d}')

            # Hide unused subplots
            for i in range(n_components, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Component selection section - MODIFIED
            selected_key = f"{ica_key}_selected_components"
            if selected_key not in st.session_state:
                st.session_state[selected_key] = []

            # Display current selections
            current_selections = st.session_state[selected_key]
            if current_selections:
                st.write(f"Currently selected components for exclusion: {', '.join(map(str, current_selections))}")
            else:
                st.write("No components currently selected")

            # Temporarily store selections without triggering reruns
            temp_key = f"{ica_key}_temp_selections"
            if temp_key not in st.session_state:
                st.session_state[temp_key] = []

            # Use a form to prevent automatic reruns - this is key
            with st.form(key=f"{ica_key}_selection_form"):
                # Component selection UI
                component_options = [f"Component {i}" for i in range(n_components)]
                selected_options = st.multiselect(
                    "Select components to exclude",
                    options=component_options,
                    default=[component_options[i] for i in current_selections if i < len(component_options)]
                )

                # Only process when form is submitted
                submit_button = st.form_submit_button("Update Selection")

            # Process form submission (this runs only after explicit submission)
            if submit_button:
                # Convert selection back to component indices
                selected_indices = [int(option.split()[1]) for option in selected_options]
                st.session_state[selected_key] = selected_indices
                st.write(f"Selection updated: {', '.join(map(str, selected_indices))}")

            # Apply button (outside the form)
            if st.button(f"Apply Selected Components ({ica_key})"):
                selected_components = st.session_state[selected_key]
                ica.exclude = selected_components
                ica.apply(self.processor.epochs)
                st.success(f"Successfully excluded components: {selected_components}")

                # Show before/after comparison
                st.subheader("Before vs After ICA")
                compare_fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                # Before ICA
                pre_ica_epochs = st.session_state.get(
                    'pre_ica_epochs') if ica_key == 'first_ica' else st.session_state.get('pre_second_ica_epochs')
                if pre_ica_epochs is not None:
                    evoked_before = pre_ica_epochs.average()
                    times = evoked_before.times
                    mean_before = evoked_before.data.mean(axis=0)
                    std_before = evoked_before.data.std(axis=0) / np.sqrt(len(evoked_before.data))

                    ax1.plot(times, mean_before, 'b-', label='Before')
                    ax1.fill_between(times,
                                     mean_before - 1.96 * std_before,
                                     mean_before + 1.96 * std_before,
                                     color='b', alpha=0.2)
                    ax1.set_title(f'Before {ica_key.replace("_", " ").title()}')
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('Amplitude (µV)')
                    ax1.grid(True)
                    ax1.legend()
                else:
                    st.warning(f"No 'pre_ica_epochs' found for {ica_key} comparison.")

                # After ICA
                evoked_after = self.processor.epochs.average()
                mean_after = evoked_after.data.mean(axis=0)
                std_after = evoked_after.data.std(axis=0) / np.sqrt(len(evoked_after.data))

                ax2.plot(times, mean_after, 'r-', label='After')
                ax2.fill_between(times,
                                 mean_after - 1.96 * std_after,
                                 mean_after + 1.96 * std_after,
                                 color='r', alpha=0.2)
                ax2.set_title(f'After {ica_key.replace("_", " ").title()}')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Amplitude (µV)')
                ax2.grid(True)
                ax2.legend()

                plt.tight_layout()
                st.pyplot(compare_fig)
                plt.close(compare_fig)

                # Update epochs in session state and processor
                if ica_key == 'first_ica':
                    st.session_state.processing_state.epochs = self.processor.epochs
                    st.session_state.processing_state.first_ica = ica
                else:
                    st.session_state.processing_state.epochs = self.processor.epochs
                    st.session_state.processing_state.second_ica = ica


        except Exception as e:
            st.error(f"Error in visualization: {str(e)}")
            st.error("Detailed error information:")
            st.code(traceback.format_exc())

    def render_clean_muscle(self):
        """Render muscle artifact cleaning interface using PARAFAC decomposition"""
        st.write("PARAFAC")

        if not st.session_state.processing_state.epochs_created:
            st.warning("Please create epochs first")
            return

        # Initialize processor if needed
        if self.processor is None:
            if not self.initialize_processor():
                st.error("Could not initialize processor")
                return

        # Store original epochs for comparison if not already stored
        if 'original_muscle_epochs' not in st.session_state:
            st.session_state.original_muscle_epochs = st.session_state.processing_state.epochs.copy()

        try:
            st.subheader("PARAFAC Settings")

            # PARAFAC settings
            with st.expander("PARAFAC Settings", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    muscle_window_start = st.number_input(
                        "Muscle Window Start (ms)",
                        value=0,
                        min_value=-500,
                        max_value=500,
                        help="Start time of the muscle artifact window in milliseconds"
                    )

                    threshold_factor = st.number_input(
                        "Threshold Factor",
                        value=1.5,
                        min_value=0.1,
                        max_value=5.0,
                        step=0.1,
                        help="Factor to multiply the median absolute deviation for threshold"
                    )

                with col2:
                    muscle_window_end = st.number_input(
                        "Muscle Window End (ms)",
                        value=50,
                        min_value=-500,
                        max_value=500,
                        help="End time of the muscle artifact window in milliseconds"
                    )

                    n_components = st.number_input(
                        "Number of Components",
                        value=20,
                        min_value=1,
                        max_value=100,
                        help="Number of components for PARAFAC decomposition"
                    )

            # Advanced options
            with st.expander("Advanced Options", expanded=False):
                show_decomposition = st.checkbox(
                    "Show Decomposition Results",
                    value=True,
                    help="Display component plots after decomposition"
                )

                save_components = st.checkbox(
                    "Save Component Data",
                    value=False,
                    help="Save component information for later analysis"
                )

            # Run muscle cleaning
            if st.button("PARAFAC Decomposition"):
                try:
                    with st.spinner("Trying PARAFAC Decomposition..."):
                        # Convert window times from ms to seconds
                        muscle_window = (muscle_window_start / 1000, muscle_window_end / 1000)

                        # Run muscle artifact cleaning
                        components = self.processor.clean_muscle_artifacts(
                            muscle_window=muscle_window,
                            threshold_factor=threshold_factor,
                            n_components=n_components,
                            verbose=True
                        )

                        # Update session state
                        st.session_state.processing_state.epochs = self.processor.epochs
                        st.session_state.processing_state.muscle_cleaned = True
                        if save_components:
                            st.session_state.muscle_components = components

                        # Success message
                        st.success("PARAFAC completed")

                        # Compare before and after
                        st.subheader("Before vs After PARAFAC Decomposition")

                        # Plotting options
                        plot_type = st.radio(
                            "Plot type",
                            ["Average", "Single Epoch", "Butterfly"],
                            horizontal=True
                        )

                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                        if plot_type == "Average":
                            st.session_state.original_muscle_epochs.average().plot(axes=ax1, show=False)
                            self.processor.epochs.average().plot(axes=ax2, show=False)
                        elif plot_type == "Single Epoch":
                            epoch_idx = st.slider("Epoch index", 0, len(self.processor.epochs) - 1, 0)
                            st.session_state.original_muscle_epochs[epoch_idx].plot(axes=ax1, show=False)
                            self.processor.epochs[epoch_idx].plot(axes=ax2, show=False)
                        else:  # Butterfly
                            st.session_state.original_muscle_epochs.plot(axes=ax1, show=False)
                            self.processor.epochs.plot(axes=ax2, show=False)

                        ax1.set_title('Evoked before epoch rejection and PARAFAC')
                        ax2.set_title('Evoked after epoch rejection and PARAFAC')

                        st.pyplot(fig)
                        plt.close()

                        # Quick stats
                        st.subheader("Cleaning Statistics")
                        col1, col2 = st.columns(2)
                        with col1:
                            original_var = np.var(st.session_state.original_muscle_epochs.get_data())
                            cleaned_var = np.var(self.processor.epochs.get_data())
                            var_reduction = (original_var - cleaned_var) / original_var
                            st.metric("Variance Reduction", f"{var_reduction:.2%}")

                        # Option to revert changes
                        if st.button("Revert Changes"):
                            self.processor.epochs = st.session_state.original_muscle_epochs.copy()
                            st.session_state.processing_state.muscle_cleaned = False
                            if 'muscle_components' in st.session_state:
                                del st.session_state.muscle_components
                            st.success("Reverted to original epochs")

                        # Store the step completion
                        st.session_state.processing_state.selected_steps['clean_muscle'] = True

                except Exception as e:
                    st.error(f"Error cleaning muscle artifacts: {str(e)}")
                    st.error("Detailed error information:")
                    import traceback
                    st.code(traceback.format_exc())

            # Show status if muscle cleaning has been completed
            if st.session_state.processing_state.muscle_cleaned:
                st.info("✅ Muscle artifact cleaning has been completed")

        except Exception as e:
            st.error(f"Error in muscle cleaning interface: {str(e)}")
            st.error("Detailed error information:")
            import traceback
            st.code(traceback.format_exc())

    def render_filter_epochs(self):
        """Render epoch filtering interface"""
        st.write("Filter Epochs")

        if not st.session_state.processing_state.epochs_created:
            st.warning("Please create epochs first")
            return

        if self.processor is None:
            if not self.initialize_processor():
                st.error("Could not initialize processor")
                return

        # Store original epochs for comparison if not already stored
        if 'original_filtered_epochs' not in st.session_state:
            st.session_state.original_filtered_epochs = st.session_state.processing_state.epochs.copy()

        try:
            st.subheader("Epoch Filtering Settings")


            # Filter settings
            col1, col2 = st.columns(2)

            with col1:
                l_freq = st.number_input(
                    "Low-cut frequency (Hz)",
                    value=1.0,
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    help="High-pass filter cutoff frequency. Set to 0 to disable."
                )

                notch_freq = st.number_input(
                    "Notch frequency (Hz)",
                    value=50.0,
                    min_value=0.0,
                    max_value=500.0,
                    step=0.1,
                    help="Frequency to notch filter (usually power line frequency). Set to 0 to disable."
                )

            with col2:
                h_freq = st.number_input(
                    "High-cut frequency (Hz)",
                    value=100.0,
                    min_value=0.0,
                    max_value=500.0,
                    step=0.1,
                    help="Low-pass filter cutoff frequency. Set to 0 to disable."
                )

            # Advanced settings
            with st.expander("Advanced Settings"):

                show_filter_response = st.checkbox(
                    "Show filter response",
                    value=True,
                    help="Display the frequency response of the filter"
                )

                show_before_after = st.checkbox(
                    "Show before/after comparison",
                    value=True,
                    help="Display comparison plots before and after filtering"
                )

            if st.button("Apply Filters"):
                try:
                    with st.spinner("Applying filters..."):
                        # Apply filtering based on selected method

                        self.processor.epochs.filter(
                            l_freq=l_freq if l_freq > 0 else None,
                            h_freq=h_freq if h_freq > 0 else None,
                            method='iir',
                            iir_params=dict(
                                order=3,
                                ftype='butter',
                                phase='zero-double',
                                btype='bandpass'
                            ),
                            verbose=True
                        )


                        # Update session state
                        st.session_state.processing_state.epochs = self.processor.epochs
                        st.session_state.processing_state.epochs_filtered = True

                        st.success("Filtering completed successfully!")

                        # Show results
                        st.subheader("Filtering Results")


                        # Option to revert changes
                        if st.button("Revert Changes"):
                            self.processor.epochs = st.session_state.original_filtered_epochs.copy()
                            st.session_state.processing_state.epochs_filtered = False
                            st.success("Reverted to original epochs")

                        # Store the step completion
                        st.session_state.processing_state.selected_steps['filter_epochs'] = True

                except Exception as e:
                    st.error(f"Error applying filters: {str(e)}")
                    st.error("Detailed error information:")
                    import traceback
                    st.code(traceback.format_exc())

            # Show status if filtering has been completed
            if st.session_state.processing_state.epochs_filtered:
                st.info("✅ Epoch filtering has been completed")

        except Exception as e:
            st.error(f"Error in filtering interface: {str(e)}")
            st.error("Detailed error information:")
            import traceback
            st.code(traceback.format_exc())



    def render_ssp(self):
        """Render SSP interface"""
        st.write("Signal Subspace Projection (SSP) Settings")

        if not st.session_state.processing_state.epochs_created:
            st.warning("Please create epochs first")
            return

        try:
            st.subheader("SSP Configuration")
            n_eeg = st.number_input(
                "Number of SSP components for EEG",
                value=2,
                min_value=1,
                max_value=10,
                help="Number of SSP components to compute for EEG"
            )

            if st.button("Apply SSP"):
                with st.spinner("Applying SSP..."):
                    # Store original epochs for comparison
                    if 'pre_ssp_epochs' not in st.session_state:
                        st.session_state.pre_ssp_epochs = self.processor.epochs.copy()

                    # Apply SSP
                    self.processor.apply_ssp(n_eeg=n_eeg)

                    # Update session state
                    st.session_state.processing_state.epochs = self.processor.epochs
                    st.session_state.processing_state.ssp_applied = True

                    # Show before/after comparison
                    st.subheader("Before vs After SSP")
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                    st.session_state.pre_ssp_epochs.average().plot(axes=ax1, show=False)
                    self.processor.epochs.average().plot(axes=ax2, show=False)
                    ax1.set_title('Before SSP')
                    ax2.set_title('After SSP')
                    st.pyplot(fig)
                    plt.close()

                    st.success("SSP applied successfully!")

        except Exception as e:
            st.error(f"Error applying SSP: {str(e)}")
            st.error("Detailed error information:")
            import traceback
            st.code(traceback.format_exc())

    def render_downsample(self):
        """Render downsampling interface"""
        st.write("Downsample Settings")

        if not st.session_state.processing_state.epochs_created:
            st.warning("Please create epochs first")
            return

        try:
            st.subheader("Downsampling Configuration")
            current_sfreq = self.processor.epochs.info['sfreq']
            st.write(f"Current sampling frequency: {current_sfreq} Hz")

            target_sfreq = st.number_input(
                "Target sampling frequency (Hz)",
                value=725,
                min_value=100,
                max_value=int(current_sfreq),
                help="Target sampling frequency in Hz"
            )

            if st.button("Apply Downsampling"):
                with st.spinner("Downsampling..."):
                    # Store original epochs for comparison
                    if 'pre_downsample_epochs' not in st.session_state:
                        st.session_state.pre_downsample_epochs = self.processor.epochs.copy()

                    # Apply downsampling
                    self.processor.final_sfreq = target_sfreq
                    self.processor.final_downsample()

                    # Update session state
                    st.session_state.processing_state.epochs = self.processor.epochs
                    st.session_state.processing_state.downsampled = True

                    # Show success message
                    st.success(f"Successfully downsampled to {target_sfreq} Hz")

                    # Show before/after comparison
                    st.subheader("Before vs After Downsampling")
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                    st.session_state.pre_downsample_epochs.average().plot(axes=ax1, show=False)
                    self.processor.epochs.average().plot(axes=ax2, show=False)
                    ax1.set_title(f'Before downsampling ({current_sfreq} Hz)')
                    ax2.set_title(f'After downsampling ({target_sfreq} Hz)')
                    st.pyplot(fig)
                    plt.close()

        except Exception as e:
            st.error(f"Error during downsampling: {str(e)}")
            st.error("Detailed error information:")
            import traceback
            st.code(traceback.format_exc())


    def render_tep_analysis(self):
        """Render TEP analysis interface with interactive Plotly visualization"""

        import plotly.graph_objects as go
        from scipy import signal

        st.write("TEP Analysis")

        if not st.session_state.processing_state.epochs_created:
            st.warning("Please create epochs first")
            return

        if self.processor is None or self.processor.epochs is None:
            st.warning("No epochs available for analysis")
            return

        try:
            # Get evoked response
            evoked = self.processor.epochs.average()

            # Channel selection
            st.subheader("Channel Selection")
            all_channels = evoked.ch_names

            # Add "Select All" and "Clear All" buttons in the same row
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select All Channels"):
                    st.session_state.selected_channels = all_channels
            with col2:
                if st.button("Clear All Channels"):
                    st.session_state.selected_channels = []

            # Initialize selected channels in session state if not exists
            if 'selected_channels' not in st.session_state:
                st.session_state.selected_channels = ['Cz']  # Default to Cz

            # Multiselect for channels
            selected_channels = st.multiselect(
                "Select channels to display",
                options=all_channels,
                default=st.session_state.selected_channels
            )
            st.session_state.selected_channels = selected_channels

            # Time window selection
            st.subheader("Time Window")
            col1, col2 = st.columns(2)
            with col1:
                tmin = st.number_input(
                    "Start time (ms)",
                    value=-300,
                    min_value=int(evoked.times[0] * 1000),
                    max_value=int(evoked.times[-1] * 1000)
                )
            with col2:
                tmax = st.number_input(
                    "End time (ms)",
                    value=600,
                    min_value=int(evoked.times[0] * 1000),
                    max_value=int(evoked.times[-1] * 1000)
                )

            # Create Plotly figure
            if selected_channels:
                # Get channel indices
                ch_idx = [evoked.ch_names.index(ch) for ch in selected_channels]

                # Time window indices
                time_mask = (evoked.times * 1000 >= tmin) & (evoked.times * 1000 <= tmax)
                times = evoked.times[time_mask] * 1000  # Convert to ms

                # Create traces for selected channels
                traces = []
                for idx, ch in zip(ch_idx, selected_channels):
                    trace = go.Scatter(
                        x=times,
                        y=evoked.data[idx, time_mask] * 1e6,  # Convert to µV
                        mode='lines',
                        name=ch,
                        hovertemplate='Amplitude: %{y:.2f} µV<br>Time: %{x:.2f} ms'
                    )
                    traces.append(trace)

                # Create layout
                layout = go.Layout(
                    title='TMS-Evoked Potentials',
                    xaxis=dict(
                        title='Time (ms)',
                        showgrid=True,
                        zeroline=True,
                        zerolinecolor='lightgray'
                    ),
                    yaxis=dict(
                        title='Amplitude (µV)',
                        showgrid=True,
                        zeroline=True,
                        zerolinecolor='lightgray'
                    ),
                    hovermode='closest',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    ),
                    width=None,  # Let Streamlit control the width
                    height=600
                )

                # Create and display figure
                fig = go.Figure(data=traces, layout=layout)

                # Add a horizontal line at y=0
                fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")

                # Add vertical line at t=0 (TMS pulse)
                fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="red")

                # Show the plot
                st.plotly_chart(fig, use_container_width=True)

                # Display peak information
                if st.checkbox("Show Peak Information"):
                    st.subheader("Peak Analysis")
                    for ch in selected_channels:
                        ch_idx = evoked.ch_names.index(ch)
                        data = evoked.data[ch_idx, time_mask] * 1e6
                        times = evoked.times[time_mask] * 1000

                        # Find peaks
                        peak_times = []
                        peak_amplitudes = []

                        # Find positive peaks
                        pos_peaks = signal.find_peaks(data)[0]
                        for peak in pos_peaks:
                            peak_times.append(times[peak])
                            peak_amplitudes.append(data[peak])

                        # Find negative peaks
                        neg_peaks = signal.find_peaks(-data)[0]
                        for peak in neg_peaks:
                            peak_times.append(times[peak])
                            peak_amplitudes.append(data[peak])

                        # Sort by time
                        peaks = sorted(zip(peak_times, peak_amplitudes))

                        # Display peaks
                        st.write(f"Channel: {ch}")
                        peak_df = pd.DataFrame(peaks, columns=['Latency (ms)', 'Amplitude (µV)'])
                        st.dataframe(peak_df)

            else:
                st.warning("Please select at least one channel to display")

        except Exception as e:
            st.error(f"Error in TEP analysis: {str(e)}")
            st.error("Detailed error information:")
            import traceback
            st.code(traceback.format_exc())

