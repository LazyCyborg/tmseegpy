# pcist.py
import numpy as np
from numpy import linalg
import scipy.signal
from typing import Optional, Union, Dict, List, Tuple
import matplotlib.pyplot as plt
import mne
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d import proj3d


class PCIst:
    """
    Class implementation of PCIst (Perturbational Complexity Index based on State transitions).
    
    This implementation is a copy paste from:
    Comolatti R et al., "A fast and general method to empirically estimate the complexity of brain responses
    to transcranial and intracranial stimulations" Brain Stimulation (2019):
    https://github.com/renzocom/PCIst/tree/master
    
    Parameters
    ----------
    epochs : mne.Epochs
        MNE epochs object containing TMS-evoked potentials. 
        Should be preprocessed before passing to this class.
    """
    
    def __init__(self, epochs: mne.epochs.Epochs):
        if not isinstance(epochs, (mne.epochs.Epochs, mne.epochs.EpochsArray)):
            print(f"Type of epochs: {type(epochs)}")
            raise ValueError("Input must be an MNE Epochs or EpochsArray object")
        self.epochs = epochs
        self.PCI = None

    def calc_PCIst(self, full_return=False, return_details=True, **par):
        ''' Calculates PCIst (Perturbational Complexity Index based on State transitions) of a signal.
        Parameters
        ----------
        epochs : mne.epochs object
            Preprocessed mne.epochs object.
        times : ndarray
            1D array (time,) containing timepoints (negative values are baseline).
        full_return : bool
            Returns multiple variables involved in PCI computation.
        **pars : dictionary
            Dictionary containing parameters (see dimensionality_reduction(),
            state_transition_quantification()and preprocess_signal() documentation).
            Example:
            >> par = {'baseline_window':(-400,-50), 'response_window':(0,300), 'k':1.2, 'min_snr':1.1,
            'max_var':99, 'embed':False,'n_steps':100}
            >> PCIst, PCIst_bydim = calc_PCIst(signal_evoked, times, **par)

        Returns
        -------
        float
            PCIst value
        OR (if full_return==True)
        dict
            Dictionary containing all variables from calculation including array 'dNSTn' with PCIst decomposition.
        '''
        # Get evoked data
        self.evoked = self.epochs.average()    
        # Get data and times and reshape
        self.signal_evk = self.evoked.get_data()
        self.times = self.evoked.times * 1000
        
        t_data = self.evoked.data
        print(f"Input data shape: {t_data.shape}")  # Should be (channels, timepoints)
        print(f"Number of channels: {len(self.epochs.ch_names)}")
        print(f"Number of timepoints: {len(self.evoked.times)}")

        if np.any(np.isnan(self.signal_evk)):
            print('Data contains nan values.')
            return 0

        #signal_evk, times = preprocess_signal(signal_evk, times, (par['baseline_window'][0],
        #                                                        par['response_window'][1]), **par)
        
        self.baseline_window = par.get('baseline_window')
        self.response_window = par.get('response_window')
        
        self.signal_svd, self.var_exp, self.eigenvalues, self.snrs = self.dimensionality_reduction(
            signal=self.signal_evk,
            times=self.times,
            response_window=self.response_window,
            baseline_window=self.baseline_window,
            max_var=par.get('max_var', 99),
            min_snr=par.get('min_snr', 1.1),
            n_components=par.get('n_components', None)
        )
        print(f"Number of components retained: {self.signal_svd.shape[0]}")
        print(f"Variance explained: {np.sum(self.var_exp):.2f}%")
        print(f"SNRs of components: {self.snrs}")

        STQ = self.state_transition_quantification(
            signal=self.signal_svd,
            times=self.times,
            k=par.get('k', 1.2),
            baseline_window=self.baseline_window,
            response_window=self.response_window,
            embed=par.get('embed', False),
            L=par.get('L', None),
            tau=par.get('tau', None),
            n_steps=par.get('n_steps', 100),
            max_thr_p=par.get('max_thr_p', 1.0)
        )
        print(f"NST base range: {np.min(STQ['NST_base']):.3f} to {np.max(STQ['NST_base']):.3f}")
        print(f"NST resp range: {np.min(STQ['NST_resp']):.3f} to {np.max(STQ['NST_resp']):.3f}")
        print(f"dNST values: {STQ['dNST']}")

        PCI = np.sum(STQ['dNST'])
        self.PCI = PCI 

        if full_return:
            return {'PCI':PCI, **STQ, 'signal_evk':self.signal_evk, 'times':self.times, 'signal_svd':self.signal_svd,
                    'eigenvalues':self.eigenvalues, 'var_exp':self.var_exp, 'snrs':self.snrs}
        
        if return_details:
            details = {
                'components': self.signal_svd,
                'times': self.times,
                'evoked_data': self.signal_evk,
                'eigenvalues': self.eigenvalues,
                'var_exp': self.var_exp,
                'snr_values': self.snrs,
                'distance_matrices': [(STQ['D_base'][0], STQ['D_resp'][0])],
                'transition_matrices': [(STQ['T_base'][0], STQ['T_resp'][0])],
                'optimal_thresholds': [STQ['max_thresholds'][0]],
                'thresholds': [STQ['thresholds'][:,0]],
                'nst_base_values': [STQ['NST_base'][:,0]],
                'nst_resp_values': [STQ['NST_resp'][:,0]],
                'diff_values': [STQ['NST_diff'][:,0]],
                'delta_nst_values': [STQ['dNST'][0]],
                'baseline_window': self.baseline_window,
                'response_window': self.response_window 
            }
            return PCI, details
            
        return PCI

    ## DIMENSIONALITY REDUCTION
    def dimensionality_reduction(self, signal, times, response_window, baseline_window, max_var=99, min_snr=1.1, n_components=None):
        print(f"Dimensionality Reduction Parameters:")
        print(f"  response_window: {response_window}")
        print(f"  baseline_window: {baseline_window}")
        print(f"  max_var: {max_var}")
        print(f"  min_snr: {min_snr}")
        print(f"  n_components: {n_components}")
        '''Returns principal components of signal according to SVD of the response.

        Calculates SVD at a given time interval (response_window) and uses the new basis to transform
        the whole signal yielding `n_components` principal components. The principal components are
        then selected to account for at least `max_var`% of the variance basesent in the signal's
        response.

        Parameters
        ----------
        signal : ndarray
            2D array (ch,time) containing signal.
        times : ndarray
            1D array (time,) containing timepoints
        response_window : tuple
            Signal's response time interval (ini,end).
        max_var: 0 < float <= 100
            Percentage of variance accounted for by the selected principal components.
        min_snr : float, optional
            Selects principal components with a signal-to-noise ratio (SNR) > min_snr.
        n_components : int, optional
            Number of principal components calculated (before selection).


        Returns
        -------
        np.ndarray
            2D array (ch,time) with selected principal components.
        np.ndarray
            1D array (n_components,) with `n_components` SVD eigenvalues of the signal's response.
        '''
        print(f"Dimensionality Reduction Parameters:")
        print(f"  response_window: {response_window}")
        print(f"  baseline_window: {baseline_window}")
        print(f"  max_var: {max_var}")
        print(f"  min_snr: {min_snr}")
        print(f"  n_components: {n_components}")

        if not n_components:
            n_components = signal.shape[0]

        Vk, eigenvalues = self.get_svd(signal, times, response_window, n_components)
        var_exp = 100 * eigenvalues**2/np.sum(eigenvalues**2)

        signal_svd = self.apply_svd(signal, Vk)

        max_dim = self.calc_maxdim(eigenvalues, max_var)

        signal_svd = signal_svd[:max_dim, :]

        snrs = self.calc_snr(signal_svd, times, baseline_window, response_window)
        signal_svd = signal_svd[snrs > min_snr, :]
        snrs = snrs[snrs > min_snr]

        Nc = signal_svd.shape[0]

        # if min_snr:
            # base_ini_ix = get_time_index(times, kwargs['baseline_window'][0])
            # base_end_ix = get_time_index(times, kwargs['baseline_window'][1])
            # resp_ini_ix = get_time_index(times, response_window[0])
            # resp_end_ix = get_time_index(times, response_window[1])
            # n_dims = np.size(signal_svd, 0)
            # snrs = np.zeros(n_dims)
            # for c in range(n_dims):
            #     resp_power = np.mean(np.square(signal_svd[c, resp_ini_ix:resp_end_ix]))
            #     base_power = np.mean(np.square(signal_svd[c, base_ini_ix:base_end_ix]))
            #     snrs[c] = np.sqrt(np.divide(resp_power, base_power))

        return signal_svd, var_exp[:Nc], eigenvalues, snrs

    def calc_snr(self, signal_svd, times, baseline_window, response_window):

        base_ini_ix = self.get_time_index(times, baseline_window[0])
        base_end_ix = self.get_time_index(times, baseline_window[1])
        resp_ini_ix = self.get_time_index(times, response_window[0])
        resp_end_ix = self.get_time_index(times, response_window[1])

        resp_power = np.mean(np.square(signal_svd[:,resp_ini_ix:resp_end_ix]), axis=1)
        base_power = np.mean(np.square(signal_svd[:,base_ini_ix:base_end_ix]), axis=1)
        snrs = np.sqrt(resp_power / base_power)
        return snrs
    

    def get_svd(self, signal_evk, times, response_window, n_components):
        ini_t, end_t = response_window
        ini_ix = self.get_time_index(times, onset=ini_t)
        end_ix = self.get_time_index(times, onset=end_t)
        signal_resp = signal_evk[:, ini_ix:end_ix].T
        print(f"Signal shape before SVD: {signal_resp.shape}")
        U, S, V = linalg.svd(signal_resp, full_matrices=False)
        print(f"All singular values: {S}")
        print(f"Number of non-zero singular values: {np.sum(S > 1e-10)}")
        V = V.T
        Vk = V[:, :n_components]
        eigenvalues = S[:n_components]
        return Vk, eigenvalues

    def apply_svd(self, signal, V):
        '''Transforms signal according to SVD basis.'''
        return signal.T.dot(V).T

    ## STATE TRANSITION QUANTIFICATION
    def state_transition_quantification(self, signal, times, k, baseline_window, response_window, embed=False,
                                        L=None, tau=None, n_steps=100, max_thr_p=1.0, **kwargs):
        ''' Receives selected principal components of perturbational signal and
        performs state transition quantification.

        Parameters
        ----------
        signal : ndarray
            2D array (component,time) containing signal (typically, the selected
            principal components).
        times : ndarray
            1D array (time,) containing timepoints
        k : float > 1
            Noise control parameter.
        baseline_window : tuple
            Signal's baseline time interval (ini,end).
        response_window : tuple
            Signal's response time interval (ini,end).
        embed : bool, optional
            Perform time-delay embedding.
        L : int
            Number of embedding dimensions.
        tau : int
            Number of timesamples of embedding delay
        n_steps : int, optional
            Number of steps used to search for the threshold that maximizes ∆NST.
            Search is performed by partitioning  the interval (defined from the median
            of the baseline’s distance matrix to the maximum of the response’s
            distance matrix) into ‘n_steps’ equal lengths.

        Returns
        -------
        float
            PCIst value.
        ndarray
            List containing component wise PCIst value (∆NSTn).
        '''
        print(f"State Transition Quantification Parameters:")
        print(f"  k: {k}")
        print(f"  baseline_window: {baseline_window}")
        print(f"  response_window: {response_window}")
        print(f"  embed: {embed}")
        print(f"  L: {L}")
        print(f"  tau: {tau}")
        print(f"  n_steps: {n_steps}")
        print(f"  max_thr_p: {max_thr_p}")
        n_dims = signal.shape[0]
        if n_dims == 0:
            print('No components --> PCIst=0')
            return {'dNST':np.array([]), 'n_dims':0}

        # EMBEDDING
        if embed:
            cut = (L-1)*tau
            times = times[cut:]
            temp_signal = np.zeros((n_dims, L, len(times)))
            for i in range(n_dims):
                temp_signal[i, :, :] = self.dimension_embedding(signal[i, :], L, tau)
            signal = temp_signal

        else:
            signal = signal[:, np.newaxis, :]

        # BASELINE AND RESPONSE DEFINITION
        base_ini_ix = self.get_time_index(times, baseline_window[0])
        base_end_ix = self.get_time_index(times, baseline_window[1])
        resp_ini_ix = self.get_time_index(times, response_window[0])
        resp_end_ix = self.get_time_index(times, response_window[1])
        n_baseline = len(times[base_ini_ix:base_end_ix])
        n_response = len(times[resp_ini_ix:resp_end_ix])

        if n_response <= 1 or n_baseline <= 1:
            print('Warning: Bad time interval defined.')

        baseline = signal[:, :, base_ini_ix:base_end_ix]
        response = signal[:, :, resp_ini_ix:resp_end_ix]

        # NST CALCULATION
            # Distance matrix
        D_base = np.zeros((n_dims, n_baseline, n_baseline))
        D_resp = np.zeros((n_dims, n_response, n_response))
            # Transition matrix
        T_base = np.zeros((n_steps, n_dims, n_baseline, n_baseline))
        T_resp = np.zeros((n_steps, n_dims, n_response, n_response))
            # Number of state transitions
        NST_base = np.zeros((n_steps, n_dims))
        NST_resp = np.zeros((n_steps, n_dims))
        thresholds = np.zeros((n_steps, n_dims))
        for i in range(n_dims):
            D_base[i, :, :] = self.recurrence_matrix(baseline[i, :, :], thr=None, mode='distance')
            D_resp[i, :, :] = self.recurrence_matrix(response[i, :, :], thr=None, mode='distance')
            min_thr = np.median(D_base[i, :, :].flatten())
            max_thr = np.max(D_resp[i, :, :].flatten()) * max_thr_p
            thresholds[:, i] = np.linspace(min_thr, max_thr, n_steps)
        for i in range(n_steps):
            for j in range(n_dims):
                T_base[i, j, :, :] = self.distance2transition(D_base[j, :, :], thresholds[i, j])
                T_resp[i, j, :, :] = self.distance2transition(D_resp[j, :, :], thresholds[i, j])

                NST_base[i, j] = np.sum(T_base[i, j, :, :])/n_baseline**2
                NST_resp[i, j] = np.sum(T_resp[i, j, :, :])/n_response**2

        # PCIST
        NST_diff = NST_resp - k * NST_base
        ixs = np.argmax(NST_diff, axis=0)
        max_thresholds = np.array([thresholds[ix, i] for ix, i in zip(ixs, range(n_dims))])
        dNST = np.array([NST_diff[ix, i] for ix, i in zip(ixs, range(n_dims))]) * n_response
        dNST = [x if x>0 else 0 for x in dNST]

        temp = np.zeros((n_dims, n_response, n_response))
        temp2 = np.zeros((n_dims, n_baseline, n_baseline))
        for i in range(n_dims):
            temp[i, :, :] = T_resp[ixs[i], i, :, :]
            temp2[i, :, :] = T_base[ixs[i], i, :, :]
        T_resp = temp
        T_base = temp2

        return {'dNST':dNST, 'n_dims':n_dims,
        'D_base':D_base, 'D_resp':D_resp, 'T_base':T_base,'T_resp':T_resp,
        'thresholds':thresholds, 'NST_diff':NST_diff, 'NST_resp':NST_resp, 'NST_base':NST_base,'max_thresholds':max_thresholds}


    def recurrence_matrix(self, signal, mode, thr=None):
        ''' Calculates distance, recurrence or transition matrix. Signal can be
        embedded (m, n_times) or not (, n_times).

        Parameters
        ----------
        signal : ndarray
            Time-series; may be a 1D (time,) or a m-dimensional array (m, time) for
            time-delay embeddeding.
        mode : str
            Specifies calculated matrix: 'distance', 'recurrence' or 'transition'
        thr : float, optional
            If transition matrix is chosen (`mode`=='transition'), specifies threshold value.

        Returns
        -------
        ndarray
            2D array containing specified matrix.
        '''
        if len(signal.shape) == 1:
            signal = signal[np.newaxis, :]
        n_dims = signal.shape[0]
        n_times = signal.shape[1]

        R = np.zeros((n_dims, n_times, n_times))
        for i in range(n_dims):
            D = np.tile(signal[i, :], (n_times, 1))
            D = D - D.T
            R[i, :, :] = D
        R = np.linalg.norm(R, ord=2, axis=0)

        mask = (R <= thr) if thr else np.zeros(R.shape).astype(bool)
        if mode == 'distance':
            R[mask] = 0
            return R
        if mode == 'recurrence':
            return mask.astype(int)
        if mode == 'transition':
            return self.diff_matrix(mask.astype(int), symmetric=False)
        return 0

    def distance2transition(self, dist_R, thr):
        ''' Receives 2D distance matrix and calculates transition matrix. '''
        mask = dist_R <= thr
        R = self.diff_matrix(mask.astype(int), symmetric=False)
        return R

    def distance2recurrence(self, dist_R, thr):
        ''' Receives 2D distance matrix and calculates recurrence matrix. '''
        mask = dist_R <= thr
        return mask.astype(int)

    def diff_matrix(self, A, symmetric=False):
        B = np.abs(np.diff(A))
        if B.shape[1] != B.shape[0]:
            B2 = np.zeros((B.shape[0], B.shape[1]+1))
            B2[:, :-1] = B
            B = B2
        if symmetric:
            B = (B + B.T)
            B[B > 0] = 1
        return B

    def calc_maxdim(self, eigenvalues, max_var):
        ''' Get number of dimensions that accumulates at least `max_var`% of total variance'''
        if max_var == 100:
            return len(eigenvalues)
        eigenvalues = np.sort(eigenvalues)[::-1] # Sort in descending order
        var = eigenvalues ** 2
        var_p = 100 * var/np.sum(var)
        var_cum = np.cumsum(var_p)
        max_dim = len(eigenvalues) - np.sum(var_cum >= max_var) + 1
        return max_dim

    def dimension_embedding(self, x, L, tau):
        '''
        Returns time-delay embedding of vector.
        Parameters
        ----------
        x : ndarray
            1D array time series.
        L : int
            Number of dimensions in the embedding.
        tau : int
            Number of samples in delay.
        Returns
        -------
        ndarray
            2D array containing embedded signal (L, time)

        '''
        assert len(x.shape) == 1, "x must be one-dimensional array (n_times,)"
        n_times = x.shape[0]
        s = np.zeros((L, n_times - (L-1) * tau))
        ini = (L-1) * tau if L > 1 else None
        s[0, :] = x[ini:]
        for i in range(1, L):
            ini = (L-i-1) * tau
            end = -i * tau
            s[i, :] = x[ini:end]
        return s

    ## PREPROCESS
    def preprocess_signal(self, signal_evk, times, time_window, baseline_corr=False, resample=None,
                        avgref=False, **kwargs):
        assert signal_evk.shape[1] == len(times), 'Signal and Time arrays must be of the same size.'
        if avgref:
            signal_evk = self.avgreference(signal_evk)
        if baseline_corr:
            signal_evk = self.baseline_correct(signal_evk, times, delta=-50)
        t_ini, t_end = time_window
        ini_ix = self.get_time_index(times, t_ini)
        end_ix = self.get_time_index(times, t_end)
        signal_evk = signal_evk[:, ini_ix:end_ix]
        times = times[ini_ix:end_ix]
        if resample:
            signal_evk, times = self.undersample_signal(signal_evk, times, new_fs=resample)
        return signal_evk, times

    def avgreference(self, signal):
        ''' Performs average reference to signal. '''
        new_signal = np.zeros(signal.shape)
        channels_mean = np.mean(signal, axis=0)[np.newaxis]
        new_signal = signal - channels_mean
        return new_signal

    def undersample_signal(self, signal, times, new_fs):
        '''
        signal : (ch x times)
        times : (times,) [ms]
        new_fs : [hz]
        '''
        n_samples = int((times[-1]-times[0])/1000 * new_fs)
        new_signal_evk, new_times = scipy.signal.resample(signal, n_samples, t=times, axis=1)
        return new_signal_evk, new_times

    def baseline_correct(self, Y, times, delta=0):
        ''' Baseline correct signal using times < delta '''
        newY = np.zeros(Y.shape)
        onset_ix = self.get_time_index(times, delta)
        baseline_mean = np.mean(Y[:, :onset_ix], axis=1)[np.newaxis]
        newY = Y - baseline_mean.T
        close_enough = np.all(np.isclose(np.mean(newY[:, :onset_ix], axis=1), 0, atol=1e-08))
        assert close_enough, "Baseline mean is not zero"
        return newY

    def get_time_index(self, times, onset=0):
        ''' Returns index of first time greater then delta. For delta=0 gets index of
        first non-negative time.
        '''
        return np.sum(times < onset)
    
    def plot_analysis(self, details: Dict, session_name: str = None) -> plt.Figure:
        """
        Plot PCIst analysis steps with parallel plots for baseline/response periods.
        Includes dual thresholds, enhanced projections, and visual connections.
        """
        # Input validation
        required_keys = ['components', 'times', 'evoked_data', 'eigenvalues', 'var_exp']
        if not all(key in details for key in required_keys):
            raise ValueError("Missing required keys in details dictionary")

        # Time window limits
        time_min, time_max = -300, 300  # in milliseconds

        # Filter times and data within our window
        mask = (details['times'] >= time_min) & (details['times'] <= time_max)
        plot_times = details['times'][mask]
        plot_evoked = details['evoked_data'][:, mask]
        plot_components = details['components'][:, mask]

        # Get number of available components
        n_available_components = details['components'].shape[0]

        # Create figure
        fig = plt.figure(figsize=(12, 20))
        gs = plt.GridSpec(8, 2, height_ratios=[0.5, 1.5, 0.75, 0.75, 1.5, 2.5, 2, 1])

        def add_connecting_lines(ax_matrix, ax_surface, fig):
            xlim = ax_matrix.get_xlim()
            ylim = ax_matrix.get_ylim()

            # Define corners in the correct order
            matrix_corners = [
                [xlim[0], ylim[0]],  # Lower-left corner
                [xlim[1], ylim[0]],  # Lower-right corner
                [xlim[1], ylim[1]],  # Upper-right corner
                [xlim[0], ylim[1]],  # Upper-left corner
            ]
            # Transform 2D data coordinates to display coordinates
            matrix_corners_display = [ax_matrix.transData.transform(c) for c in matrix_corners]

            # Project 3D corners to 2D display coordinates
            surface_corners_display = []
            for x3d, y3d in matrix_corners:
                z3d = ax_surface.get_zlim()[0]  # Use the bottom z-limit
                x2d, y2d, _ = proj3d.proj_transform(x3d, y3d, z3d, ax_surface.get_proj())
                x_disp, y_disp = ax_surface.transData.transform((x2d, y2d))
                surface_corners_display.append([x_disp, y_disp])

            # Draw lines between corresponding corners
            for (mx, my), (sx, sy) in zip(matrix_corners_display, surface_corners_display):
                mx_fig, my_fig = fig.transFigure.inverted().transform((mx, my))
                sx_fig, sy_fig = fig.transFigure.inverted().transform((sx, sy))
                line = plt.Line2D([mx_fig, sx_fig],
                                [my_fig, sy_fig],
                                transform=fig.transFigure, color='gray', linestyle='--', alpha=0.5)
                fig.lines.append(line)


        # Add text box with analysis details at the top
        ax_text = fig.add_subplot(gs[0, :])
        analysis_text = (
            f"Session name: {session_name}\n"
            f"Number of non-zero singular values: {np.sum(details['eigenvalues'] > 1e-10)}\n"
            f"Number of components retained: {details['components'].shape[0]}\n"
            f"Variance explained: {np.sum(details['var_exp']):.2f}%\n"
            f"PCI: {self.PCI:.6f}"
        )
        ax_text.text(0.05, 0.05, analysis_text, fontsize=10, family='monospace',
                    verticalalignment='bottom', transform=ax_text.transAxes)
        ax_text.axis('off')
        
        # A: TMS-evoked potentials
        ax1 = fig.add_subplot(gs[1, :])
        ax1.plot(plot_times, plot_evoked.T, 'b-', alpha=0.3, linewidth=0.5)
        ax1.set_title('A) TMS-Evoked Potentials')
        ax1.axvline(x=0, color='k', linestyle='--')
        ax1.set_xlim(time_min, time_max)
        ax1.text(0, ax1.get_ylim()[1], 'TMS', ha='center', va='bottom')
        ax1.text(-350, ax1.get_ylim()[1], 'TEPs', va='center', ha='right')

        max_pcs_to_plot = min(2, n_available_components)  # Plot up to 2 PCs if available
        for i in range(max_pcs_to_plot):
            ax_pc = fig.add_subplot(gs[2 + i, :])
            ax_pc.set_xlim(time_min, time_max)
            if i == 0:
                ax_pc.plot(details['times'], details['components'][i], 'k-')
                ax_pc.text(-350, ax_pc.get_ylim()[1], 'Principal\nComponents', 
                        va='center', ha='right')
            else:
                ax_pc.plot(details['times'], details['components'][i], 'b-')
            ax_pc.text(-300, ax_pc.get_ylim()[1], f'PC{i+1}', va='top')
            ax_pc.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        # If we have fewer than 2 components, fill remaining subplot with informative text
        if n_available_components < 2:
            for i in range(n_available_components, 2):
                ax_empty = fig.add_subplot(gs[2 + i, :])
                ax_empty.text(0.5, 0.5, f'PC{i+1} not available\n(only {n_available_components} component{"s" if n_available_components != 1 else ""} present)',
                            ha='center', va='center')
                ax_empty.axis('off')
        
        # Get common scale for distance matrices and surfaces
        baseline_dist, response_dist = details['distance_matrices'][0]
        vmin = min(baseline_dist.min(), response_dist.min())
        vmax = max(baseline_dist.max(), response_dist.max())
        
        # Get thresholds
        optimal_threshold = details['optimal_thresholds'][0]
        lower_threshold = optimal_threshold * 0.6
        
        # B: Distance matrices
        # Baseline distance matrix
        # Baseline distance matrix
        ax3 = fig.add_subplot(gs[4, 0])
        im1 = ax3.imshow(baseline_dist, aspect='equal', cmap='viridis',
                        extent=[-300, 0, -300, 0], vmin=vmin, vmax=vmax, origin='lower')
        ax3.set_xlim(-300, 0)
        ax3.set_ylim(-300, 0)
        ax3.set_aspect('equal')
        ax3.set_title('B) Baseline Distance Matrix')
        plt.colorbar(im1, ax=ax3)

        # Response distance matrix
        ax4 = fig.add_subplot(gs[4, 1])
        im2 = ax4.imshow(response_dist, aspect='equal', cmap='viridis',
                        extent=[0, 300, 0, 300], vmin=vmin, vmax=vmax, origin='lower')
        ax4.set_xlim(0, 300)
        ax4.set_ylim(0, 300)
        ax4.set_aspect('equal')
        ax4.set_title('B) Response Distance Matrix')
        plt.colorbar(im2, ax=ax4)

        
        # Add connecting boxes
        for ax in [ax3, ax4]:
            box = plt.Rectangle(
                (ax.get_xlim()[0], ax.get_ylim()[0]),
                ax.get_xlim()[1] - ax.get_xlim()[0],
                ax.get_ylim()[1] - ax.get_ylim()[0],
                fill=False, color='pink', linewidth=1.5
            )
            ax.add_patch(box)
        


        def add_projection_shadow(ax_3d, dist_matrix, is_baseline=True):
            """Create visual connection between matrix and surface."""
            # Get the matrix corner points
            if is_baseline:
                x = np.linspace(-300, 0, dist_matrix.shape[0])
            else:
                x = np.linspace(0, 300, dist_matrix.shape[0])
            X, Y = np.meshgrid(x, x)
            vmin = dist_matrix.min()
            vmax = dist_matrix.max()
            
            # Project on walls and bottom
            ax_3d.contourf(X, Y, dist_matrix, 
                        zdir='z', offset=vmin,
                        levels=np.linspace(vmin, vmax, 20), 
                        cmap='viridis', alpha=0.3)
            
            if is_baseline:
                ax_3d.contourf(X, np.full_like(Y, Y.min()), dist_matrix,
                            zdir='y', offset=Y.min(),
                            levels=np.linspace(vmin, vmax, 20),
                            cmap='viridis', alpha=0.3)
                ax_3d.contourf(np.full_like(X, X.min()), Y, dist_matrix,
                            zdir='x', offset=X.min(),
                            levels=np.linspace(vmin, vmax, 20),
                            cmap='viridis', alpha=0.3)
            else:
                ax_3d.contourf(X, np.full_like(Y, Y.max()), dist_matrix,
                            zdir='y', offset=Y.max(),
                            levels=np.linspace(vmin, vmax, 20),
                            cmap='viridis', alpha=0.3)
                ax_3d.contourf(np.full_like(X, X.max()), Y, dist_matrix,
                            zdir='x', offset=X.max(),
                            levels=np.linspace(vmin, vmax, 20),
                            cmap='viridis', alpha=0.3)

        
        # C: 3D representation
        # Baseline surface
        ax5 = fig.add_subplot(gs[5, 0], projection='3d')
        x = np.linspace(-300, 0, baseline_dist.shape[0])
        X, Y = np.meshgrid(x, x)
        # Main surface plot
        surf1 = ax5.plot_surface(X, Y, baseline_dist, cmap='viridis',
                                vmin=vmin, vmax=vmax, alpha=0.8)
        
        
        # Add threshold plane
        xx, yy = np.meshgrid([-300, 0], [-300, 0])
        zz = np.full_like(X, optimal_threshold)
        ax5.plot_surface(X, Y, zz, alpha=0.2, color='gray')
        
        # Enhanced shadow projections
        ax5.contourf(X, Y, baseline_dist, zdir='z', offset=optimal_threshold,
                    levels=np.linspace(vmin, vmax, 20), cmap='viridis', alpha=0.5)
        
        ax5.set_title('C) Baseline Threshold Surface')
        ax5.view_init(elev=20, azim=-45)
        ax5.set_zlim(vmin, vmax)
        ax5.set_xlim(-300, 0)
        ax5.set_ylim(-300, 0)

        # Response surface
        ax6 = fig.add_subplot(gs[5, 1], projection='3d')
        x = np.linspace(0, 300, response_dist.shape[0])
        X, Y = np.meshgrid(x, x)
        
        # Main surface plot
        surf2 = ax6.plot_surface(X, Y, response_dist, cmap='viridis',
                            vmin=vmin, vmax=vmax, alpha=0.8)
        
        # Add threshold plane
        xx, yy = np.meshgrid([0, 300], [0, 300])
        zz = np.full_like(xx, optimal_threshold)
        ax6.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
        ax6.contourf(X, Y, response_dist, zdir='z', offset=optimal_threshold,
                    levels=np.linspace(vmin, vmax, 20), cmap='viridis', alpha=0.5)
        
        ax6.set_title('C) Response Threshold Surface')
        ax6.view_init(elev=20, azim=-45)
        ax6.set_zlim(vmin, vmax)
        ax6.set_xlim(0, 300)
        ax6.set_ylim(0, 300)
        
        fig.canvas.draw()
        
        # Add connecting lines
        #add_connecting_lines(ax3, ax5, fig)
        #add_connecting_lines(ax4, ax6, fig)

        add_projection_shadow(ax5, baseline_dist, is_baseline=True)
        add_projection_shadow(ax6, response_dist, is_baseline=False)

        
        # D: Transition matrices
        # Calculate transition matrices for both thresholds
        baseline_trans_low = self.distance2transition(baseline_dist, lower_threshold)
        response_trans_low = self.distance2transition(response_dist, lower_threshold)
        baseline_trans_opt = self.distance2transition(baseline_dist, optimal_threshold)
        response_trans_opt = self.distance2transition(response_dist, optimal_threshold)

        # Get maximum size to pad all matrices to same size
        max_size = max(baseline_trans_low.shape[0], 
                    response_trans_low.shape[0],
                    baseline_trans_opt.shape[0], 
                    response_trans_opt.shape[0])

        # Function to pad matrix to target size
        def pad_matrix(matrix, target_size):
            pad_width = target_size - matrix.shape[0]
            if pad_width > 0:
                return np.pad(matrix, ((0, pad_width), (0, pad_width)), mode='constant')
            return matrix

        # Pad all matrices to same size
        baseline_trans_low = pad_matrix(baseline_trans_low, max_size)
        response_trans_low = pad_matrix(response_trans_low, max_size)
        baseline_trans_opt = pad_matrix(baseline_trans_opt, max_size)
        response_trans_opt = pad_matrix(response_trans_opt, max_size)

        # Calculate sizes for the grid
        n = max_size
        full_size = n * 2  # Total size for 2x2 grid

        # Create empty array for full grid
        full_trans = np.zeros((full_size, full_size))

        # Fill in the quadrants
        full_trans[:n, :n] = baseline_trans_low
        full_trans[:n, n:] = baseline_trans_opt
        full_trans[n:, :n] = response_trans_low
        full_trans[n:, n:] = response_trans_opt
        
        # Create a new subplot for the consolidated transition matrices
        ax_trans = fig.add_subplot(gs[6, :])  # Use one row instead of two
        
        # Plot the consolidated matrix
        im = ax_trans.imshow(full_trans, cmap='binary')
        
        # Add grid lines to separate quadrants
        ax_trans.axhline(y=n-0.5, color='r', linestyle='-', linewidth=0.5)
        ax_trans.axvline(x=n-0.5, color='r', linestyle='-', linewidth=0.5)
        
        # Add labels
        ax_trans.text(-n/4, n/2, f'ε\' = {lower_threshold:.3f}', 
                    rotation=90, ha='center', va='center')
        ax_trans.text(-n/4, n*1.5, f'ε* = {optimal_threshold:.3f}', 
                    rotation=90, ha='center', va='center')
        ax_trans.text(n/2, -n/4, 'Baseline', ha='center', va='center')
        ax_trans.text(n*1.5, -n/4, 'Response', ha='center', va='center')
        ax_trans.set_title('D) Transition Matrices')
        
        # E: NST analysis
        ax11 = fig.add_subplot(gs[7, :])
        ax11.plot(details['thresholds'][0], details['nst_resp_values'][0],
                'b-', label='NST Response (NSTres)')
        ax11.plot(details['thresholds'][0], details['nst_base_values'][0],
                'k-', label='NST Baseline (NSTbase)')
        ax11.plot(details['thresholds'][0], details['diff_values'][0],
                'r-', label='ΔNST = NSTres - k × NSTbase')
        
        # Add both threshold lines
        ax11.axvline(x=lower_threshold, color='gray', linestyle='--', label='ε\'')
        ax11.axvline(x=optimal_threshold, color='r', linestyle='--', label='ε*')
        
        ax11.set_xlabel('Threshold (ε)')
        ax11.set_ylabel('Number of State Transitions')
        ax11.legend()
        ax11.set_title('E) State Transitions')
        
        # Add PCIst equation
        ax11.text(0.95, 0.95, 'PCIst = ∑ΔNSTn', transform=ax11.transAxes,
                ha='right', va='top', fontsize=10)
        
        plt.subplots_adjust(hspace=0.4)
        plt.tight_layout()
        return fig
            

