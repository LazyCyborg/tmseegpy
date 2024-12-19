# data_loader.py
from pathlib import Path
import mne
from typing import Optional, Union, Dict, List, Tuple
from .neurone_loader import Recording  # Keep NeurOne support

class TMSEEGLoader:
    """
    A flexible data loader for TMS-EEG data that supports multiple formats.
    Default support for NeurOne (.ses) files, with additional support for
    other common EEG formats.
    
    Parameters
    ----------
    data_path : str or Path
        Path to the data directory or file
    format : str, optional
        Data format to load ('neurone', 'brainvision', 'edf', 'cnt', 'eeglab', or 'auto')
        Default is 'neurone'
    substitute_zero_events_with : int, optional 
        Value to substitute zero events with in NeurOne data
    eeglab_montage_units : str, optional
        Units for EEGLAB channel positions ('auto', 'mm', 'm', etc.)
    """
    
    SUPPORTED_FORMATS = {
        'neurone': ('.ses',),
        'brainvision': ('.vhdr', '.eeg', '.vmrk'),
        'edf': ('.edf',),
        'cnt': ('.cnt',),
        'eeglab': ('.set',),
    }
    
    def __init__(self, 
                 data_path: Union[str, Path],
                 format: str = 'neurone',
                 substitute_zero_events_with: int = 10,
                 eeglab_montage_units: str = 'auto',
                 verbose: Optional[Union[bool, str, int]] = None):
        self.data_path = Path(data_path)
        self.format = format.lower()
        self.substitute_zero_events_with = substitute_zero_events_with
        self.eeglab_montage_units = eeglab_montage_units
        self.verbose = verbose
        
        if self.format not in self.SUPPORTED_FORMATS and self.format != 'auto':
            raise ValueError(f"Unsupported format: {format}. "
                           f"Supported formats are: {list(self.SUPPORTED_FORMATS.keys())}")
        
        # For storing loaded sessions
        self.sessions = []
        self.raw_list = []
        self.session_info = []
        
    def detect_format(self) -> str:
        """
        Detect the data format based on file extensions in the directory.
        
        Returns
        -------
        str
            Detected format name
        """
        if self.data_path.is_file():
            ext = self.data_path.suffix.lower()
            for fmt, extensions in self.SUPPORTED_FORMATS.items():
                if ext in extensions:
                    return fmt
        else:
            # Check directory contents for each format
            for fmt, extensions in self.SUPPORTED_FORMATS.items():
                if any(self.data_path.glob(f"**/*{extensions[0]}")):
                    return fmt
        
        raise ValueError("Could not detect data format")

    def load_data(self) -> List[mne.io.Raw]:
        """
        Load TMS-EEG data in the specified format.
        
        Returns
        -------
        List[mne.io.Raw]
            List of loaded Raw objects
        """
        if self.format == 'auto':
            self.format = self.detect_format()
            print(f"Detected format: {self.format}")
            
        # Dictionary mapping formats to their loading methods
        format_loaders = {
            'neurone': self._load_neurone,
            'brainvision': self._load_brainvision,
            'edf': self._load_edf,
            'cnt': self._load_cnt,
            'eeglab': self._load_eeglab
        }
        
        if self.format not in format_loaders:
            raise ValueError(f"Unsupported format: {self.format}")
            
        return format_loaders[self.format]()

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
        
        # Store session info
        self.session_info = [
            {'name': str(session.path.name), 'format': 'neurone', 'path': str(session.path)}
            for session in rec.sessions
        ]
        
        return self.raw_list

    def _load_brainvision(self) -> List[mne.io.Raw]:
        """
        Load BrainVision files.
        
        Returns
        -------
        List[mne.io.Raw]
            List of loaded Raw objects
        """
        if self.data_path.is_file():
            # Single file
            raw = mne.io.read_raw_brainvision(self.data_path, preload=True, verbose=self.verbose)
            self.raw_list = [raw]
            self.session_info = [{
                'name': self.data_path.stem,
                'format': 'brainvision',
                'path': str(self.data_path)
            }]
        else:
            # Directory with multiple files
            vhdr_files = list(self.data_path.glob("**/*.vhdr"))
            self.raw_list = []
            self.session_info = []
            
            for f in vhdr_files:
                try:
                    raw = mne.io.read_raw_brainvision(f, preload=True, verbose=self.verbose)
                    self.raw_list.append(raw)
                    self.session_info.append({
                        'name': f.stem,
                        'format': 'brainvision',
                        'path': str(f)
                    })
                except Exception as e:
                    print(f"Warning: Could not load {f}: {str(e)}")
                    
        return self.raw_list

    def _load_edf(self) -> List[mne.io.Raw]:
        """
        Load EDF files.
        
        Returns
        -------
        List[mne.io.Raw]
            List of loaded Raw objects
        """
        if self.data_path.is_file():
            raw = mne.io.read_raw_edf(self.data_path, preload=True, verbose=self.verbose)
            self.raw_list = [raw]
            self.session_info = [{
                'name': self.data_path.stem,
                'format': 'edf',
                'path': str(self.data_path)
            }]
        else:
            edf_files = list(self.data_path.glob("**/*.edf"))
            self.raw_list = []
            self.session_info = []
            
            for f in edf_files:
                try:
                    raw = mne.io.read_raw_edf(f, preload=True, verbose=self.verbose)
                    self.raw_list.append(raw)
                    self.session_info.append({
                        'name': f.stem,
                        'format': 'edf',
                        'path': str(f)
                    })
                except Exception as e:
                    print(f"Warning: Could not load {f}: {str(e)}")
                    
        return self.raw_list

    def _load_cnt(self) -> List[mne.io.Raw]:
        """
        Load Neuroscan .cnt files.
        
        Returns
        -------
        List[mne.io.Raw]
            List of loaded Raw objects
        """
        if self.data_path.is_file():
            raw = mne.io.read_raw_cnt(self.data_path, preload=True, verbose=self.verbose)
            self.raw_list = [raw]
            self.session_info = [{
                'name': self.data_path.stem,
                'format': 'cnt',
                'path': str(self.data_path)
            }]
        else:
            cnt_files = list(self.data_path.glob("**/*.cnt"))
            self.raw_list = []
            self.session_info = []
            
            for f in cnt_files:
                try:
                    raw = mne.io.read_raw_cnt(f, preload=True, verbose=self.verbose)
                    self.raw_list.append(raw)
                    self.session_info.append({
                        'name': f.stem,
                        'format': 'cnt',
                        'path': str(f)
                    })
                except Exception as e:
                    print(f"Warning: Could not load {f}: {str(e)}")
                    
        return self.raw_list

    def _load_eeglab(self) -> List[mne.io.Raw]:
        """
        Load EEGLAB .set files.
        
        Returns
        -------
        List[mne.io.Raw]
            List of loaded Raw objects
        """
        if self.data_path.is_file():
            raw = mne.io.read_raw_eeglab(
                self.data_path,
                preload=True,
                montage_units=self.eeglab_montage_units,
                eog='auto',
                verbose=self.verbose
            )
            self.raw_list = [raw]
            self.session_info = [{
                'name': self.data_path.stem,
                'format': 'eeglab',
                'path': str(self.data_path)
            }]
        else:
            set_files = list(self.data_path.glob("**/*.set"))
            self.raw_list = []
            self.session_info = []
            
            for f in set_files:
                try:
                    raw = mne.io.read_raw_eeglab(
                        f,
                        preload=True,
                        montage_units=self.eeglab_montage_units,
                        eog='auto',
                        verbose=self.verbose
                    )
                    self.raw_list.append(raw)
                    self.session_info.append({
                        'name': f.stem,
                        'format': 'eeglab',
                        'path': str(f)
                    })
                except Exception as e:
                    print(f"Warning: Could not load {f}: {str(e)}")
                    
        return self.raw_list

    def get_session_names(self) -> List[str]:
        """
        Get list of session names.
        
        Returns
        -------
        List[str]
            List of session names
        """
        return [info['name'] for info in self.session_info] if self.session_info else []
    
    def get_session_info(self) -> List[Dict]:
        """
        Get detailed information about loaded sessions.
        
        Returns
        -------
        List[Dict]
            List of dictionaries containing session information
        """
        return self.session_info
    
    def print_summary(self) -> None:
        """Print a summary of loaded data."""
        print(f"\nData Format: {self.format}")
        print(f"Number of sessions: {len(self.raw_list)}")
        print("\nSession Details:")
        for i, info in enumerate(self.session_info):
            print(f"\nSession {i+1}:")
            print(f"  Name: {info['name']}")
            print(f"  Format: {info['format']}")
            print(f"  Path: {info['path']}")
            if self.raw_list[i] is not None:
                print(f"  Channels: {len(self.raw_list[i].ch_names)}")
                print(f"  Duration: {self.raw_list[i].times[-1]:.1f} seconds")
                print(f"  Sampling Rate: {self.raw_list[i].info['sfreq']} Hz")