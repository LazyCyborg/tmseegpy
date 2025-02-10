import mne


def filter_raw(self, l_freq=0.1, h_freq=45, notch_freq=50, notch_width=2):
    """
    Filter raw data using a zero-phase Butterworth filter with improved stability.

    Parameters
    ----------
    l_freq : float
        Lower frequency cutoff for bandpass filter (default: 0.1 Hz)
    h_freq : float
        Upper frequency cutoff for bandpass filter (default: 45 Hz)
    notch_freq : float
        Frequency for notch filter (default: 50 Hz)
    notch_width : float
        Width of notch filter (default: 2 Hz)
    """
    from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch
    import numpy as np

    print(f"Applying SciPy filters to raw data with frequency {l_freq}Hz and frequency {h_freq}Hz")

    # Create a copy of the raw data
    filtered_raw = self.raw.copy()

    # Get data and scale it up for better numerical precision
    data = filtered_raw.get_data()
    scale_factor = 1e6  # Convert to microvolts
    data = data * scale_factor

    print(f"Data shape: {data.shape}")
    print(f"Scaled data range: [{np.min(data)}, {np.max(data)}] µV")

    # Ensure data is float64
    data = data.astype(np.float64)

    sfreq = filtered_raw.info['sfreq']
    nyquist = sfreq / 2

    try:
        # High-pass filter
        sos_high = butter(3, l_freq / nyquist, btype='high', output='sos')
        data = sosfiltfilt(sos_high, data, axis=-1)
        print(f"After high-pass - Data range: [{np.min(data)}, {np.max(data)}] µV")

        # Low-pass filter
        sos_low = butter(5, h_freq / nyquist, btype='low', output='sos')
        data = sosfiltfilt(sos_low, data, axis=-1)
        print(f"After low-pass - Data range: [{np.min(data)}, {np.max(data)}] µV")

        # Multiple notch filters for harmonics
        for freq in [notch_freq, notch_freq * 2]:  # 50 Hz and 100 Hz
            # Using iirnotch for sharper notch characteristics
            b, a = iirnotch(freq / nyquist, 35)  # Q=35 for very narrow notch
            data = filtfilt(b, a, data, axis=-1)
        print(f"After notch - Data range: [{np.min(data)}, {np.max(data)}] µV")

        # Scale back
        data = data / scale_factor
        filtered_raw._data = data

    except Exception as e:
        print(f"Error during filtering: {str(e)}")
        raise

    print("Filtering complete")
    self.raw = filtered_raw


def mne_filter_epochs(self, l_freq=0.1, h_freq=45, notch_freq=50, notch_width=2):
    """
    Filter epoched data using MNE's built-in filtering plus custom notch.

    Parameters
    ----------
    l_freq : float
        Lower frequency bound for bandpass filter
    h_freq : float
        Upper frequency bound for bandpass filter
    notch_freq : float
        Frequency to notch filter (usually power line frequency)
    notch_width : float
        Width of the notch filter

    Returns
    -------
    None
        Updates self.epochs in place
    """
    from scipy.signal import iirnotch, filtfilt
    import numpy as np
    from mne.time_frequency import psd_array_welch

    if self.epochs is None:
        raise ValueError("Must create epochs before filtering")

    # Store original epochs for potential recovery
    original_epochs = self.epochs
    try:
        # Create a deep copy to work with
        filtered_epochs = self.epochs.copy()

        # Get data and sampling frequency
        data = filtered_epochs.get_data()
        sfreq = filtered_epochs.info['sfreq']
        nyquist = sfreq / 2.0

        # Diagnostic before filtering
        psds, freqs = psd_array_welch(data.reshape(-1, data.shape[-1]),
                                      sfreq=sfreq,
                                      fmin=0,
                                      fmax=200,
                                      n_per_seg=256,
                                      n_overlap=128)

        print(f"\nBefore filtering:")
        print(f"Peak frequency: {freqs[np.argmax(psds.mean(0))]} Hz")
        print(f"Frequency range with significant power: {freqs[psds.mean(0) > psds.mean(0).max() * 0.1][0]:.1f} - "
              f"{freqs[psds.mean(0) > psds.mean(0).max() * 0.1][-1]:.1f} Hz")

        # Apply filters in sequence
        print("\nApplying low-pass filter...")
        filtered_epochs.filter(
            l_freq=None,
            h_freq=h_freq,
            picks='eeg',
            filter_length='auto',
            h_trans_bandwidth=10,
            method='fir',
            fir_window='hamming',
            fir_design='firwin',
            phase='zero',
            verbose=True
        )

        print("\nApplying high-pass filter...")
        filtered_epochs.filter(
            l_freq=l_freq,
            h_freq=None,
            picks='eeg',
            filter_length='auto',
            l_trans_bandwidth=l_freq / 2,
            method='fir',
            fir_window='hamming',
            fir_design='firwin',
            phase='zero',
            verbose=True
        )

        # Get the filtered data for notch filtering
        data = filtered_epochs.get_data()

        print("\nApplying notch filters...")
        for freq in [notch_freq, notch_freq * 2]:
            print(f"Processing {freq} Hz notch...")
            Q = 30.0  # Quality factor
            w0 = freq / nyquist
            b, a = iirnotch(w0, Q)

            # Apply to each epoch and channel
            for epoch_idx in range(data.shape[0]):
                for ch_idx in range(data.shape[1]):
                    data[epoch_idx, ch_idx, :] = filtfilt(b, a, data[epoch_idx, ch_idx, :])

        # Update the filtered epochs with notch-filtered data
        filtered_epochs._data = data

        # Diagnostic after filtering
        data_filtered = filtered_epochs.get_data()
        psds, freqs = psd_array_welch(data_filtered.reshape(-1, data_filtered.shape[-1]),
                                      sfreq=sfreq,
                                      fmin=0,
                                      fmax=200,
                                      n_per_seg=256,
                                      n_overlap=128)

        print(f"\nAfter filtering:")
        print(f"Peak frequency: {freqs[np.argmax(psds.mean(0))]} Hz")
        print(f"Frequency range with significant power: {freqs[psds.mean(0) > psds.mean(0).max() * 0.1][0]:.1f} - "
              f"{freqs[psds.mean(0) > psds.mean(0).max() * 0.1][-1]:.1f} Hz")

        # Verify the filtered data
        if np.any(np.isnan(filtered_epochs._data)):
            raise ValueError("Filtering produced NaN values")

        if np.any(np.isinf(filtered_epochs._data)):
            raise ValueError("Filtering produced infinite values")

        # Update the instance's epochs with the filtered version
        self.epochs = filtered_epochs
        print("\nFiltering completed successfully")

    except Exception as e:
        print(f"Error during filtering: {str(e)}")
        print("Reverting to original epochs")
        self.epochs = original_epochs
        raise


def scipy_filter_epochs(self, l_freq=0.1, h_freq=45, notch_freq=50, notch_width=2):
    """
    Filter epoched data using a zero-phase Butterworth filter with improved stability.

    Parameters
    ----------
    l_freq : float
        Lower frequency cutoff for bandpass filter (default: 0.1 Hz)
    h_freq : float
        Upper frequency cutoff for bandpass filter (default: 45 Hz)
    notch_freq : float
        Frequency for notch filter (default: 50 Hz)
    notch_width : float
        Width of notch filter (default: 2 Hz)
    """
    from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch
    import numpy as np

    if self.epochs is None:
        raise ValueError("Must create epochs before filtering")

    # Create a copy of the epochs object
    filtered_epochs = self.epochs.copy()

    # Get data and scale it up for better numerical precision
    data = filtered_epochs.get_data()
    scale_factor = 1e6  # Convert to microvolts
    data = data * scale_factor

    print(f"Data shape: {data.shape}")
    print(f"Scaled data range: [{np.min(data)}, {np.max(data)}] µV")

    # Ensure data is float64
    data = data.astype(np.float64)

    sfreq = filtered_epochs.info['sfreq']
    nyquist = sfreq / 2

    try:
        # High-pass filter
        sos_high = butter(3, l_freq / nyquist, btype='high', output='sos')
        data = sosfiltfilt(sos_high, data, axis=-1)
        print(f"After high-pass - Data range: [{np.min(data)}, {np.max(data)}] µV")

        # Low-pass filter
        sos_low = butter(5, h_freq / nyquist, btype='low', output='sos')
        data = sosfiltfilt(sos_low, data, axis=-1)
        print(f"After low-pass - Data range: [{np.min(data)}, {np.max(data)}] µV")

        # Multiple notch filters for harmonics
        for freq in [notch_freq, notch_freq * 2]:  # 50 Hz and 100 Hz
            # Using iirnotch for sharper notch characteristics
            b, a = iirnotch(freq / nyquist, 35)  # Q=35 for very narrow notch
            data = filtfilt(b, a, data, axis=-1)
        print(f"After notch - Data range: [{np.min(data)}, {np.max(data)}] µV")

        # Scale back
        data = data / scale_factor
        filtered_epochs._data = data

    except Exception as e:
        print(f"Error during filtering: {str(e)}")
        raise

    print("Filtering complete")
    self.epochs = filtered_epochs