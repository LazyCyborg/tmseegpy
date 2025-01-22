from pathlib import Path
import mne
from typing import Optional, Union, Dict, List, Any
from .neurone_loader import Recording


class TMSEEGLoader:
    """Flexible data loader for TMS-EEG data supporting multiple formats"""

    SUPPORTED_FORMATS = {
        'neurone': ('.ses',),
        'brainvision': ('.vhdr', '.vmrk', '.eeg'),
        'edf': ('.edf',),
        'cnt': ('.cnt',),
        'eeglab': ('.set', '.fdt'),
        'auto': ('.ses', '.vhdr', '.edf', '.cnt', '.set')
    }

    def __init__(self,
                 data_path: Union[str, Path],
                 format: str = 'auto',
                 substitute_zero_events_with: int = 10,
                 eeglab_montage_units: str = 'auto',
                 verbose: bool = False):
        """Initialize loader."""
        self.data_path = Path(data_path)
        self.format = format.lower()
        self.substitute_zero_events_with = substitute_zero_events_with
        self.eeglab_montage_units = eeglab_montage_units
        self.verbose = verbose

        if not self.data_path.exists():
            raise ValueError(f"Path does not exist: {self.data_path}")

        # Initialize containers
        self.sessions = []
        self.raw_list = []
        self.session_info = []

        # Validate format
        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. "
                           f"Supported formats are: {list(self.SUPPORTED_FORMATS.keys())}")

    def detect_format(self) -> str:
        """Detect data format based on file extensions"""
        if self.data_path.is_file():
            ext = self.data_path.suffix.lower()
            for fmt, extensions in self.SUPPORTED_FORMATS.items():
                if fmt != 'auto' and ext in extensions:
                    return fmt
        else:
            # Look for known file types in directory
            for fmt, extensions in self.SUPPORTED_FORMATS.items():
                if fmt != 'auto':
                    for ext in extensions:
                        if any(self.data_path.glob(f"**/*{ext}")):
                            return fmt

        raise ValueError(f"Could not detect format for {self.data_path}")

    def _find_data_files(self) -> List[Path]:
        """Find all relevant data files"""
        if self.format == 'auto':
            self.format = self.detect_format()
            if self.verbose:
                print(f"Detected format: {self.format}")

        extensions = self.SUPPORTED_FORMATS[self.format]

        if self.data_path.is_file():
            return [self.data_path]

        files = []
        for ext in extensions:
            files.extend(self.data_path.glob(f"**/*{ext}"))

        # Filter out secondary files
        if self.format == 'eeglab':
            files = [f for f in files if f.suffix != '.fdt']
        elif self.format == 'brainvision':
            files = [f for f in files if f.suffix == '.vhdr']

        return sorted(files)

    def load_data(self) -> List[mne.io.Raw]:
        """Load all data files"""
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")

        if self.format == 'auto':
            self.format = self.detect_format()
            if self.verbose:
                print(f"Detected format: {self.format}")

        # Special handling for NeurOne format
        if self.format == 'neurone':
            raw_list = self._load_neurone()
            if not raw_list:
                print(f"No data loaded from {self.data_path}")
                return []
            return raw_list

        # Handle other formats
        data_files = self._find_data_files()
        if not data_files:
            print(f"No {self.format} files found in {self.data_path}")
            return []

        self.raw_list = []
        self.session_info = []

        for file_path in data_files:
            try:
                if self.verbose:
                    print(f"Loading {file_path}")
                raw = self._load_single_file(file_path)
                if raw is not None:
                    if not isinstance(raw, mne.io.Raw):
                        print(f"Warning: Loaded data from {file_path} is not an MNE Raw object")
                        continue
                    self.raw_list.append(raw)
                    self.session_info.append({
                        'name': file_path.stem,
                        'format': self.format,
                        'path': str(file_path)
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

        if not self.raw_list:
            print("Warning: No data was successfully loaded")

        return self.raw_list

    def _load_single_file(self, file_path: Path) -> Optional[mne.io.Raw]:
        """Load a single data file"""
        try:
            if file_path.suffix.lower() in ('.vhdr', '.eeg', '.vmrk'):
                return mne.io.read_raw_brainvision(file_path, preload=True)
            elif file_path.suffix.lower() == '.edf':
                return mne.io.read_raw_edf(file_path, preload=True)
            elif file_path.suffix.lower() == '.cnt':
                return mne.io.read_raw_cnt(file_path, preload=True)
            elif file_path.suffix.lower() == '.set':
                return mne.io.read_raw_eeglab(
                    file_path,
                    preload=True,
                    montage_units=self.eeglab_montage_units
                )
            elif file_path.suffix.lower() == '.ses':
                rec = Recording(str(file_path))
                if rec.sessions:
                    return rec.sessions[0].to_mne(
                        substitute_zero_events_with=self.substitute_zero_events_with
                    )
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None

    def _load_neurone(self) -> List[mne.io.Raw]:
        """
        Load NeurOne .ses files.

        Returns
        -------
        List[mne.io.Raw]
            List of loaded Raw objects
        """
        rec = Recording(str(self.data_path))
        self.sessions = rec.sessions
        self.raw_list = [
            session.to_mne(substitute_zero_events_with=self.substitute_zero_events_with)
            for session in rec.sessions
        ]

        # Modified session info creation
        self.session_info = []
        for session in rec.sessions:
            # Get session path as Path object and extract name
            session_path = Path(session.path)
            session_name = session_path.name if isinstance(session_path, Path) else Path(session.path).name

            self.session_info.append({
                'name': session_name,
                'format': 'neurone',
                'path': str(session.path)
            })

        return self.raw_list

    def get_session_names(self) -> List[str]:
        """Get list of session names"""
        return [info['name'] for info in self.session_info]

    def get_session_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about sessions"""
        return self.session_info

    def print_summary(self) -> None:
        """Print summary of loaded data"""
        print(f"\nData Format: {self.format}")
        print(f"Number of sessions: {len(self.raw_list)}")
        print("\nSession Details:")
        for i, info in enumerate(self.session_info):
            print(f"\nSession {i + 1}:")
            print(f"  Name: {info['name']}")
            print(f"  Format: {info['format']}")
            print(f"  Path: {info['path']}")
            if self.raw_list[i] is not None:
                print(f"  Channels: {len(self.raw_list[i].ch_names)}")
                print(f"  Duration: {self.raw_list[i].times[-1]:.1f} seconds")
                print(f"  Sampling Rate: {self.raw_list[i].info['sfreq']} Hz")