from typing import Tuple, Optional

import numpy as np

def remove_tms_artifact(self,
                        cut_times_tms: Tuple[float, float] = (-2, 10),
                        replace_times: Optional[Tuple[float, float]] = None,
                        verbose: bool = True) -> None:
    """
    Remove TMS artifacts following TESA implementation.

    Parameters
    ----------
    cut_times_tms : tuple
        Time window to cut around TMS pulse in ms [start, end]
        Default is [-2, 10] following TESA
    replace_times : tuple, optional
        Time window for calculating average to replace removed data in ms [start, end]
        If None (default), data will be replaced with 0s
    """
    raw_out = self.raw.copy()
    data = raw_out.get_data()
    sfreq = raw_out.info['sfreq']

    # Store original info about cut (like TESA's EEG.tmscut)
    if not hasattr(self, 'tmscut'):
        self.tmscut = []

    tmscut_info = {
        'cut_times_tms': cut_times_tms,
        'replace_times': replace_times,
        'sfreq': sfreq,
        'interpolated': 'no'
    }

    cut_samples = np.round(np.array(cut_times_tms) * sfreq / 1000).astype(int)

    # Get TMS annotations
    tms_annotations = [ann for ann in raw_out.annotations
                       if ann['description'] == 'Stimulation']

    print(f"\nFound {len(tms_annotations)} TMS events to process")
    print(f"Removing artifact in window {cut_times_tms} ms")

    processed_count = 0
    skipped_count = 0

    for ann in tms_annotations:
        event_sample = int(ann['onset'] * sfreq)
        start = event_sample + cut_samples[0]
        end = event_sample + cut_samples[1]

        if start < 0 or end >= data.shape[1]:
            skipped_count += 1
            continue

        if replace_times is None:
            data[:, start:end] = 0
        else:
            # Calculate average from replace_times window
            replace_samples = np.round(np.array(replace_times) * sfreq / 1000).astype(int)
            baseline_start = event_sample + replace_samples[0]
            baseline_end = event_sample + replace_samples[1]
            if baseline_start >= 0 and baseline_end < data.shape[1]:
                baseline_mean = np.mean(data[:, baseline_start:baseline_end], axis=1)
                data[:, start:end] = baseline_mean[:, np.newaxis]
        processed_count += 1

    print(f"Successfully removed artifacts from {processed_count} events")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} events due to window constraints")

    raw_out._data = data
    raw_out.set_annotations(raw_out.annotations)
    self.raw = raw_out
    self.tmscut.append(tmscut_info)


def interpolate_tms_artifact(self,
                             method: str = 'cubic',
                             interp_window: float = 1.0,
                             cut_times_tms: Tuple[float, float] = (-2, 10),  # Add this back
                             verbose: bool = True) -> None:
    """
    Interpolate TMS artifacts following TESA implementation.
    Uses polynomial interpolation rather than spline interpolation.

    Parameters
    ----------
    method : str
        Interpolation method: must be 'cubic' for TESA compatibility
    interp_window : float
        Time window (in ms) before and after artifact for fitting cubic function
        Default is 1.0 ms following TESA
    cut_times_tms : tuple
        Time window where TMS artifact was removed in ms [start, end]
        Default is (-2, 10) following TESA
    verbose : bool
        Whether to print progress information
    """
    if not hasattr(self, 'tmscut') or not self.tmscut:
        raise ValueError("Must run remove_tms_artifact first")

    print(f"\nStarting interpolation with {method} method")
    print(f"Using interpolation window of {interp_window} ms")
    print(f"Processing cut window {cut_times_tms} ms")

    interpolated_count = 0
    warning_count = 0

    raw_out = self.raw.copy()
    data = raw_out.get_data()
    sfreq = raw_out.info['sfreq']

    cut_samples = np.round(np.array(cut_times_tms) * sfreq / 1000).astype(int)
    interp_samples = int(round(interp_window * sfreq / 1000))

    for tmscut in self.tmscut:
        if tmscut['interpolated'] == 'no':
            cut_times = tmscut['cut_times_tms']
            cut_samples = np.round(np.array(cut_times) * sfreq / 1000).astype(int)
            interp_samples = int(round(interp_window * sfreq / 1000))

            # Process annotations
            tms_annotations = [ann for ann in raw_out.annotations
                               if ann['description'] == 'Stimulation']

            for ann in tms_annotations:
                event_sample = int(ann['onset'] * sfreq)
                start = event_sample + cut_samples[0]
                end = event_sample + cut_samples[1]

                # Calculate fitting windows
                window_start = start - interp_samples
                window_end = end + interp_samples

                if window_start < 0 or window_end >= data.shape[1]:
                    warning_count += 1
                    continue

                # Get time points for fitting
                x = np.arange(window_end - window_start + 1)
                x_fit = np.concatenate([
                    x[:interp_samples],
                    x[-interp_samples:]
                ])

                # Center x values at 0 to avoid badly conditioned warnings (TESA approach)
                x_fit = x_fit - x_fit[0]
                if len(x) <= 2 * interp_samples:
                    print(f"Warning: Window too small for interpolation at sample {event_sample}")
                    warning_count += 1
                    continue

                x_interp = x[interp_samples:-interp_samples] - x_fit[0]

                # Interpolate each channel using polynomial fit
                for ch in range(data.shape[0]):
                    y_full = data[ch, window_start:window_end + 1]
                    y_fit = np.concatenate([
                        y_full[:interp_samples],
                        y_full[-interp_samples:]
                    ])

                    # Using polynomial fit (like TESA) instead of spline which I used before and got worse results I think
                    p = np.polyfit(x_fit, y_fit, 3)
                    data[ch, start:end + 1] = np.polyval(p, x_interp)

                interpolated_count += 1

            tmscut['interpolated'] = 'yes'

    print(f"\nSuccessfully interpolated {interpolated_count} events")
    if warning_count > 0:
        print(f"Encountered {warning_count} warnings during interpolation")
    print("TMS artifact interpolation complete")
    raw_out._data = data
    raw_out.set_annotations(raw_out.annotations)
    self.raw = raw_out