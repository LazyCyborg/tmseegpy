import mne
import os
import pandas as pd
from typing import List, Dict, Optional, Tuple
from tmseegpy.neurone_loader_fix import Recording, Session, Phase
import streamlit as st
import tempfile
from pathlib import Path

from streamlit.runtime.uploaded_file_manager import UploadedFile

class DataLoader:
    """Handles loading of both NeurOne and standard MNE-supported data formats."""

    def __init__(self):
        self.supported_formats = {
            'NeurOne': ['.dat', '.xml'],
            'MNE': ['.fif', '.set', '.edf', '.bdf', '.vhdr']
        }

    def detect_format(self, path: str) -> str:
        """
        Detect if the path contains NeurOne or standard MNE data.

        Parameters
        ----------
        path : str
            Path to the data file

        Returns
        -------
        str
            Format type ('NeurOne' or 'MNE')
        """
        if os.path.isdir(path) and any(f.endswith('.xml') for f in os.listdir(path)):
            return 'NeurOne'
        elif isinstance(path, str):  # Handle string paths
            file_extension = Path(path).suffix.lower()
            if file_extension in self.supported_formats['MNE']:
                return 'MNE'
        elif hasattr(path, 'name'):  # Handle UploadedFile objects
            file_extension = Path(path.name).suffix.lower()
            if file_extension in self.supported_formats['MNE']:
                return 'MNE'

        supported_extensions = (
                self.supported_formats['NeurOne'] +
                self.supported_formats['MNE']
        )
        raise ValueError(
            f"Unsupported data format. Supported formats are: "
            f"{', '.join(supported_extensions)}"
        )

    def load_neurone_data(self, data_path: str) -> Tuple[List[mne.io.Raw], pd.DataFrame]:
        """
        Load NeurOne data and return list of raw objects and metadata.ß
        """
        try:
            rec = Recording(data_path)
            phase_info = []

            # Default notes dictionary - can be customized through UI
            notes = {}

            # Allow user to add notes through UI
            st.subheader("Session Notes")
            for session_idx, session in enumerate(rec.sessions):
                for phase_idx, phase in enumerate(session.phases):
                    session_num = session_idx + 1
                    phase_num = phase_idx + 1
                    key = (session_num, phase_num)

                    # Create input field for notes
                    note = st.text_input(
                        f"Notes for Session {session_num}, Phase {phase_num}",
                        value=notes.get(key, ""),
                        key=f"note_{session_num}_{phase_num}"
                    )
                    notes[key] = note

            # Process each phase
            for session_idx, session in enumerate(rec.sessions):
                for phase_idx, phase in enumerate(session.phases):
                    session_num = session_idx + 1
                    phase_num = phase_idx + 1

                    metadata = {
                        'session': session_num,
                        'phase': phase_num,
                        'start_time': phase.time_start,
                        'note': notes.get((session_num, phase_num), ""),
                        'duration': phase.n_samples / phase.sampling_rate,
                        'sampling_rate': phase.sampling_rate
                    }

                    # Convert to MNE with progress indicator
                    progress_text = st.empty()
                    progress_text.text(f"Converting Session {session_num}, Phase {phase_num}...")
                    raw = phase.to_mne(substitute_zero_events_with=10)
                    progress_text.text(f"Converted Session {session_num}, Phase {phase_num}")

                    phase_info.append({
                        'metadata': metadata,
                        'raw': raw
                    })

            # Create metadata DataFrame
            metadata_df = pd.DataFrame([info['metadata'] for info in phase_info])
            raw_list = [info['raw'] for info in phase_info]

            return raw_list, metadata_df

        except Exception as e:
            st.error(f"Error loading NeurOne data: {str(e)}")
            return None, None

    def load_mne_data(self, uploaded_file: UploadedFile, additional_files=None) -> mne.io.Raw:
        """
        Load standard MNE-supported data formats with Streamlit upload support.

        Parameters
        ----------
        uploaded_file : streamlit.uploaded_file_button
            File uploaded through Streamlit interface
        additional_files : list of uploaded_file
            Additional files needed for formats like BrainVision

        Returns
        -------
        mne.io.Raw
            Loaded raw data object
        """
        try:
            file_extension = Path(uploaded_file.name).suffix.lower()
            temp_dir = None
            tmp_path = None

            # For files with compound extensions like .cdt.dpa
            if uploaded_file.name.lower().endswith(('.cdt.dpa', '.cdt.cef')):
                file_extension = '.'.join(Path(uploaded_file.name).name.lower().split('.')[-2:])

            # Special handling for BrainVision files which require multiple files
            if file_extension == '.vhdr' and additional_files:
                # Create a temporary directory for all BrainVision files
                temp_dir = tempfile.mkdtemp()
                base_name = Path(uploaded_file.name).stem

                # Save all three files (.vhdr, .eeg, .vmrk) to the temp directory
                tmp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(tmp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                # Save additional files
                for add_file in additional_files:
                    add_file_path = os.path.join(temp_dir, add_file.name)
                    with open(add_file_path, 'wb') as f:
                        f.write(add_file.getvalue())

                st.info(f"Saved all BrainVision files to temporary directory: {temp_dir}")
            else:
                # For other formats, use a single temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    # Write the uploaded file content to the temporary file
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

            try:
                if file_extension == '.fif':
                    st.info("Loading FIFF format file...")
                    raw = mne.io.read_raw_fif(tmp_path, preload=True)

                elif file_extension == '.set':
                    st.info("Loading EEGLAB SET format file...")
                    raw = mne.io.read_raw_eeglab(tmp_path, preload=True)

                elif file_extension == '.edf':
                    st.info("Loading EDF format file...")
                    raw = mne.io.read_raw_edf(
                        tmp_path,
                        preload=True,
                        stim_channel='auto',
                        misc=['AUTO']
                    )

                elif file_extension == '.bdf':
                    st.info("Loading BDF format file...")
                    raw = mne.io.read_raw_bdf(
                        tmp_path,
                        preload=True,
                        stim_channel='auto',
                        misc=['AUTO']
                    )

                elif file_extension == '.vhdr':
                    st.info("Loading BrainVision format file...")
                    raw = mne.io.read_raw_brainvision(
                        tmp_path,
                        preload=True,
                        eog=('HEOGL', 'HEOGR', 'VEOGb'),
                        misc='auto'
                    )

                # Add Curry file format handling
                elif file_extension in ['.cdt', '.cef', '.dat', '.dap', '.rs3', '.cdt.dpa', '.cdt.cef']:
                    st.info("Loading Curry format file...")
                    raw = mne.io.read_raw_curry(
                        tmp_path,
                        preload=True
                    )

                else:
                    raise ValueError(f"Unsupported file format: {file_extension}")

                # Print information about the loaded data
                st.info(f"Successfully loaded file with {len(raw.ch_names)} channels")
                st.info(f"Sampling frequency: {raw.info['sfreq']} Hz")

                # Check for and handle stim channels
                stim_channels = mne.pick_types(raw.info, stim=True, exclude=[])
                if len(stim_channels) > 0:
                    st.info(
                        f"Found {len(stim_channels)} stimulus channels: {[raw.ch_names[ch] for ch in stim_channels]}")
                else:
                    st.warning("No stimulus channels detected in the file")

                # Check and handle channel types
                ch_types = raw.get_channel_types()
                if 'eeg' not in ch_types:
                    st.warning("No EEG channels detected. Setting appropriate channel types...")
                    # Set channel types for EEG channels (modify the pattern as needed)
                    eeg_picks = [idx for idx, name in enumerate(raw.ch_names)
                                 if any(pattern in name.upper()
                                        for pattern in ['EEG', 'C', 'F', 'P', 'O', 'T'])]
                    if eeg_picks:
                        raw.set_channel_types({raw.ch_names[idx]: 'eeg' for idx in eeg_picks})
                        st.info(f"Set {len(eeg_picks)} channels as EEG type")

                # Display channel types summary
                channel_type_counts = {}
                for ch_type in ch_types:
                    if ch_type not in channel_type_counts:
                        channel_type_counts[ch_type] = 1
                    else:
                        channel_type_counts[ch_type] += 1
                st.info("Channel types found:")
                for ch_type, count in channel_type_counts.items():
                    st.info(f"- {ch_type}: {count} channels")

                # For BrainVision files, check for impedance values
                if file_extension == '.vhdr' and hasattr(raw, 'impedances'):
                    st.info("Impedance data found in BrainVision file")
                    impedances = raw.impedances
                    if impedances:
                        st.write("Channel impedances (kΩ):")
                        imp_df = pd.DataFrame({
                            'Channel': list(impedances.keys()),
                            'Impedance (kΩ)': [imp / 1000 for imp in impedances.values()]  # Convert to kΩ
                        })
                        st.dataframe(imp_df)

                return raw

            finally:
                # Clean up temporary files
                if temp_dir:
                    import shutil
                    try:
                        shutil.rmtree(temp_dir)
                        st.info(f"Cleaned up temporary directory: {temp_dir}")
                    except Exception as e:
                        st.warning(f"Could not remove temporary directory: {str(e)}")
                elif tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        st.warning(f"Could not remove temporary file: {str(e)}")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.error("Detailed error information:")
            import traceback
            st.code(traceback.format_exc())
            return None



    def save_data(self, raw_list: List[mne.io.Raw], metadata_df: pd.DataFrame,
                  output_dir: str) -> None:
        """
        Save converted data to MNE format.
        """
        os.makedirs(output_dir, exist_ok=True)

        for idx, raw in enumerate(raw_list):
            metadata = metadata_df.iloc[idx]
            filename = (f"{metadata['note']}_session{metadata['session']}_"
                        f"phase{metadata['phase']}-raw.fif")
            filepath = os.path.join(output_dir, filename)
            raw.save(filepath, overwrite=True)