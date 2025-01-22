# gui_app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import sys
import io
import os

from tmseegpy.run import process_subjects


class ToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.id = None  # For scheduled showing
        self.widget.bind("<Enter>", self.schedule)
        self.widget.bind("<Leave>", self.leave)

    def schedule(self, event=None):
        self.unschedule()
        self.id = self.widget.after(500, self.enter)  # 500ms delay

    def unschedule(self):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def enter(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(self.tooltip, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1)
        label.pack()

    def leave(self, event=None):
        self.unschedule()
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class TMSEEG_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TMS-EEG Preprocessing and PCIst")

        import queue
        
        # Set initial window size (width x height)
        self.root.geometry("1000x1000")
        
        # Initialize state variables first
        self.show_advanced = tk.BooleanVar(value=False)
        self.running = False
        
        # Initialize validation rules and tooltips
        self.init_validation_rules()
        self.init_tooltips()

        # Create scroll canvas
        self.create_scroll_canvas()

        # Create GUI elements
        self.create_gui_elements()

    def init_validation_rules(self):
        self.validation_rules = {
            'initial_sfreq': {'min': 100, 'max': 10000, 'type': float},
            'final_sfreq': {'min': 100, 'max': 10000, 'type': float},
            'random_seed': {'min': 0, 'max': 1000000, 'type': int},
            'bad_channels_threshold': {'min': 0, 'max': 10, 'type': float},
            'bad_epochs_threshold': {'min': 0, 'max': 10, 'type': float},
            #'amplitude_threshold': {'min': 50, 'max': 100000, 'type': float},
            'tms_muscle_thresh': {'min': 0, 'max': 10, 'type': float},
            'n_components': {'min': 1, 'max': 100, 'type': int},
            'ssp_n_eeg': {'min': 1, 'max': 10, 'type': int},
            'l_freq': {'min': 0, 'max': 100, 'type': float},
            'h_freq': {'min': 1, 'max': 1000, 'type': float},
            'notch_freq': {'min': 0, 'max': 1000, 'type': float},
            'notch_width': {'min': 0, 'max': 10, 'type': float},
            'initial_cut_start': {'min': -100, 'max': 0, 'type': float},
            'initial_cut_end': {'min': 0, 'max': 100, 'type': float},
            'initial_interp_window': {'min': 0.1, 'max': 10.0, 'type': float},
            'extended_cut_start': {'min': -100, 'max': 0, 'type': float},
            'extended_cut_end': {'min': 0, 'max': 100, 'type': float},
            'extended_interp_window': {'min': 0.1, 'max': 10.0, 'type': float},
            'interpolation_method': {'type': 'str', 'choices': ['cubic']},
            'epochs_tmin': {'min': -2, 'max': 0, 'type': float},
            'epochs_tmax': {'min': 0, 'max': 2, 'type': float},
            'baseline_start': {'min': -1000, 'max': 0, 'type': int},
            'baseline_end': {'min': -1000, 'max': 0, 'type': int},
            'k': {'min': 0, 'max': 10, 'type': float},
            'min_snr': {'min': 0, 'max': 10, 'type': float},
            'max_var': {'min': 0, 'max': 100, 'type': float},
            'n_steps': {'min': 10, 'max': 1000, 'type': int},
            'substitute_zero_events_with': {'min': 1, 'max': 100, 'type': int},
            'threshold_factor': {'min': 0.1, 'max': 10.0, 'type': float},
            'lambda2': {'min': 1e-6, 'max': 1e-2, 'type': float},
            'stiffness': {'min': 1, 'max': 10, 'type': int},
            'ica_method': {'type': 'str', 'choices': ['fastica', 'infomax']},
            'second_ica_method': {'type': 'str', 'choices': ['fastica', 'infomax']},
            'first_ica_manual': {'type': 'bool'},
            'second_ica_manual': {'type': 'bool'},
            'response_start': {'min': -1000, 'max': 1000, 'type': int},
            'response_end': {'min': -1000, 'max': 1000, 'type': int},
            'n_samples': {'min': 1, 'max': 20, 'type': int},
            'min_peak_distance': {'min': 1, 'max': 100, 'type': int},
            'prominence': {'min': 0.001, 'max': 1.0, 'type': float},
            'analysis_start': {'min': -1000, 'max': 0, 'type': float},
            'analysis_end': {'min': 0, 'max': 1000, 'type': float}
            
        }

    def init_tooltips(self):
        self.tooltips = {
            'initial_sfreq': 'Initial downsampling frequency (Hz) for raw data processing (default: 1000 Hz)',
            'final_sfreq': 'Final downsampling frequency (Hz) for the processed epochs (default: 725 Hz)',
            'random_seed': 'Random seed for reproducibility',
            'bad_channels_threshold': 'Threshold for detecting bad channels',
            'bad_epochs_threshold': 'Threshold for detecting bad epochs',
           # 'amplitude_threshold': 'Threshold for epoch rejection based on peak-to-peak amplitude in µV',
            'tms_muscle_thresh': 'Threshold for detecting muscle artifacts',
            'n_components': 'Number of components for PARAFAC',
            'ssp_n_eeg': 'Number of EEG components for SSP',
            'l_freq': 'Lower frequency cutoff for filtering',
            'h_freq': 'Upper frequency cutoff for filtering',
            'notch_freq': 'Frequency for notch filter',
            'notch_width': 'Width of the notch filter',
            'initial_cut_start': 'Start time for initial TMS artifact removal (TESA default: -2)',
            'initial_cut_end': 'End time for initial TMS artifact removal (TESA default: 10)',
            'initial_interp_window': 'Initial interpolation window (TESA default: 1.0)',
            'extended_cut_start': 'Start time for extended TMS artifact removal (TESA default: -2)',
            'extended_cut_end': 'End time for extended TMS artifact removal (TESA default: 15)',
            'extended_interp_window': 'Extended interpolation window (TESA default: 5.0)',
            'interpolation_method': 'Interpolation method (TESA requires cubic)',
            'epochs_tmin': 'Start time for epochs (s)',
            'epochs_tmax': 'End time for epochs (s)',
            'baseline_start': 'Start time for baseline period (ms)',
            'baseline_end': 'End time for baseline period (ms)',
            'k': 'PCIst parameter k',
            'min_snr': 'Minimum signal-to-noise ratio',
            'max_var': 'Maximum variance explained (%)',
            'n_steps': 'Number of steps for PCIst calculation',
            'min_peak_distance': 'Minimum distance between peaks',
            'substitute_zero_events_with': 'Value to substitute zero events with in the data',
            'threshold_factor': 'Threshold factor for muscle artifact cleaning',
            'muscle_window_start': 'Start time for muscle artifact window (s)',
            'muscle_window_end': 'End time for muscle artifact window (s)',
            'ica_method': 'Method for first ICA (fastica or infomax)',
            'first_ica_manual': 'Enable manual selection of components for first ICA pass',
            'second_ica_manual': 'Enable manual selection of components for second ICA pass',
            'second_ica_method': 'Method for second ICA (fastica or infomax)',
            'lambda2': 'Regularization parameter for CSD transformation',
            'stiffness': 'Stiffness parameter for CSD transformation',
            'response_start': 'Start of response window in ms',
            'response_end': 'End of response window in ms',
            'n_samples': 'Number of samples to check on each side for peak detection',
            'min_peak_distance': 'Minimum distance between peaks in milliseconds',
            'prominence': 'Minimum prominence for peak detection as fraction of max amplitude',
            'analysis_start': 'Start time for TEP analysis in milliseconds',
            'analysis_end': 'End time for TEP analysis in milliseconds'
        }
        
    def create_scroll_canvas(self):
        self.main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.main_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mouse wheel scrolling
        self.root.bind("<MouseWheel>", self._on_mousewheel)

    def create_gui_elements(self):
        main_frame = ttk.Frame(self.scrollable_frame, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Directory selectors (always visible)
        self.create_directory_selectors(main_frame)

        # Create a frame for controls (checkbox, buttons, and progress bar)
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        # Left side: Advanced options checkbox
        ttk.Checkbutton(controls_frame,
                        text="Show Advanced Options",
                        variable=self.show_advanced,
                        command=self.toggle_advanced_options).grid(
            row=0, column=0, sticky=tk.W)

        # Right side: Run and Stop buttons with progress bar
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.grid(row=0, column=1, sticky=tk.E)

        ttk.Button(buttons_frame, text="Run Epoched",
                   command=lambda: self.run_analysis(continuous=False)).grid(row=0, column=0, padx=5)
        ttk.Button(buttons_frame, text="Run Continuous",
                   command=lambda: self.run_analysis(continuous=True)).grid(row=0, column=1, padx=5)
        ttk.Button(buttons_frame, text="Stop",
                   command=self.stop_analysis).grid(row=0, column=2, padx=5)

        # Progress bar below the buttons in the same frame
        self.progress = ttk.Progressbar(controls_frame, length=300, mode='indeterminate')
        self.progress.grid(row=1, column=0, columnspan=2, pady=5)

        # Create frames for options (hidden by default)
        self.options_frame = ttk.Frame(main_frame)
        self.options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        self.options_frame.grid_remove()  # Hide initially

        # Create basic and advanced options in the hidden frame
        self.create_basic_options(self.options_frame)
        self.create_advanced_options(self.options_frame)

        # Console at bottom
        self.create_console(main_frame)

    def toggle_advanced_options(self):
        if self.show_advanced.get():
            self.options_frame.grid()
        else:
            self.options_frame.grid_remove()

    def create_directory_selectors(self, parent):
        dir_frame = ttk.LabelFrame(parent, text="Data Selection (data directory must contain a folder named 'TMSEEG)' ", padding="10")
        dir_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Make directory selectors more prominent
        ttk.Label(dir_frame, text="Data Directory:", font=('TkDefaultFont', 10, 'bold')).grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.data_dir = tk.StringVar(value=str(Path.cwd() / 'data'))
        ttk.Entry(dir_frame, textvariable=self.data_dir, width=60).grid(
            row=0, column=1, padx=5, pady=5)
        ttk.Button(dir_frame, text="Browse", command=self.browse_data_dir).grid(
            row=0, column=2, padx=5, pady=5)
        
        ttk.Label(dir_frame, text="Output Directory:", font=('TkDefaultFont', 10, 'bold')).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_dir = tk.StringVar(value=str(Path.cwd() / 'output'))
        ttk.Entry(dir_frame, textvariable=self.output_dir, width=60).grid(
            row=1, column=1, padx=5, pady=5)
        ttk.Button(dir_frame, text="Browse", command=self.browse_output_dir).grid(
            row=1, column=2, padx=5, pady=5)
        
    def validate_parameter(self, param, value):
        rules = self.validation_rules.get(param)
        if not rules:
            return True
            
        if rules['type'] == 'str':
            return value in rules['choices']
            
        try:
            # First convert to float to handle all numeric inputs
            val = float(value)
            
            # Then convert to int if needed, using round()
            if rules['type'] == int:
                val = round(val)
                
            # Check bounds
            if 'min' in rules and val < rules['min']:
                return False
            if 'max' in rules and val > rules['max']:
                return False
                
            return True
        except ValueError:
            return False

    def add_parameter_with_validation(self, parent, label, param, default, row):
        ttk.Label(parent, text=label + ":").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        var = tk.StringVar(value=str(default))
        entry = ttk.Entry(parent, textvariable=var, width=10)
        entry.grid(row=row, column=1, padx=5, pady=2)
        
        # Add tooltip
        if param in self.tooltips:
            ToolTip(entry, self.tooltips[param])
        
        # Add validation
        def validate(*args):
            if not self.validate_parameter(param, var.get()):
                rules = self.validation_rules[param]
                messagebox.showwarning("Invalid Value", 
                    f"Invalid value for {param}.\n"
                    f"Expected {rules['type'].__name__} between {rules['min']} and {rules['max']}")
                var.set(str(default))
        
        var.trace_add('write', validate)  # Updated from trace() to trace_add()
        return var

    def run_analysis(self, continuous=False):
        if self.running:
            return

        self.continuous_mode = continuous
            
        # Validate all parameters before running
        for param, var in self.params.items():
            if not self.validate_parameter(param, var.get()):
                rules = self.validation_rules[param]
                messagebox.showerror("Validation Error", 
                    f"Invalid value for {param}.\n"
                    f"Expected {rules['type'].__name__} between {rules['min']} and {rules['max']}")
                return
                
        # Convert parameters to correct types for args
        args_dict = {
            # Directory and basic options
            'data_dir': self.data_dir.get(),
            'output_dir': self.output_dir.get(),
            'data_format': self.data_format.get(),
            'eeglab_montage_units': self.eeglab_units.get(),
            'plot_preproc': self.plot_preproc.get(),
            'clean_muscle_artifacts': self.clean_muscle.get(),
            'save_evoked': self.save_evoked.get(),
            'research': self.research_stats.get(),
            'validate_teps': self.validate_teps.get(),
            'save_validation': self.save_validation.get(),
            'apply_ssp': self.apply_ssp.get(),
            'apply_csd': self.apply_csd.get(),
            'skip_second_artifact_removal': self.skip_second_artifact_removal.get(),
            'stim_channel': self.stim_channel.get(),
            'filter_raw': self.filter_raw.get(),
            'plot_raw': self.plot_raw.get(),
            'mne_filter_epochs': self.mne_filter_epochs.get(),
            'first_ica_manual': self.first_ica_manual.get(),
            'initial_sfreq': float(self.params['initial_sfreq'].get()),
            'final_sfreq': float(self.params['final_sfreq'].get()),
            'second_ica_manual': self.second_ica_manual.get(),
            'muscle_window_start': float(self.params['muscle_window_start'].get()),
            'muscle_window_end': float(self.params['muscle_window_end'].get()),
            'threshold_factor': float(self.params['threshold_factor'].get()),
            'n_components': int(self.params['n_components'].get()),

            # Set default values
            'continuous_mode': continuous,
            'substitute_zero_events_with': int(self.params['substitute_zero_events_with'].get()),
            'interpolation_method': 'cubic',
            'interp_window': 1.0,
            'no_second_ICA': False,
            'embed': False,
            'fix_artifact_window_start': -0.005,
            'fix_artifact_window_end': 0.015,
            'response_start': int(self.params['response_start'].get()),
            'response_end': int(self.params['response_end'].get())
        }

        # Add ALL parameters from self.params
        for param, var in self.params.items():
            try:
                if param in self.validation_rules:
                    if self.validation_rules[param]['type'] == 'str':
                        args_dict[param] = var.get()
                    else:
                        value = float(var.get())
                        if self.validation_rules[param]['type'] == int:
                            value = round(value)
                        args_dict[param] = value
            except ValueError:
                messagebox.showerror("Error", f"Invalid value for {param}")
                self.progress.stop()
                self.running = False
                return

        # Create Args object and set all attributes
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        args = Args(**args_dict)
        for key, value in args_dict.items():
            setattr(args, key, value)

        self.console.delete(1.0, tk.END)
        self.progress.start()
        self.running = True

        # Run analysis in separate thread
        thread = threading.Thread(target=self.run_analysis_thread, args=(args,))
        thread.daemon = True
        thread.start()
        
    def create_parameter_group(self, parent, params, start_row=0):
        vars = {}
        for i, (label, (param, default)) in enumerate(params.items()):
            vars[param] = self.add_parameter_with_validation(parent, label, param, default, start_row + i)
        return vars
    
    def add_parameter_group(self, parent, params):
        for i, (label, (param, default)) in enumerate(params.items()):
            ttk.Label(parent, text=label + ":").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            var = self.add_parameter_with_validation(parent, label, param, default, i)
            self.params[param] = var
    
    def _on_mousewheel(self, event):
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_basic_options(self, parent):
        options_frame = ttk.LabelFrame(parent, text="Basic Options", padding="5")
        options_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Create two columns for better organization
        left_frame = ttk.Frame(options_frame)
        left_frame.grid(row=0, column=0, padx=10, sticky=(tk.N, tk.W))

        right_frame = ttk.Frame(options_frame)
        right_frame.grid(row=0, column=1, padx=10, sticky=(tk.N, tk.W))

        # Left column - Processing Options
        processing_frame = ttk.LabelFrame(left_frame, text="Processing Options", padding="5")
        processing_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        self.clean_muscle = tk.BooleanVar()
        muscle_btn = ttk.Checkbutton(processing_frame,
                                     text="Clean Muscle Artifacts (PARAFAC)",
                                     variable=self.clean_muscle)
        muscle_btn.grid(row=0, column=0, sticky=tk.W)
        ToolTip(muscle_btn, "Use PARAFAC decomposition to clean muscle artifacts")

        self.apply_ssp = tk.BooleanVar(value=False)
        ssp_btn = ttk.Checkbutton(processing_frame,
                                  text="Apply SSP",
                                  variable=self.apply_ssp)
        ssp_btn.grid(row=1, column=0, sticky=tk.W)
        ToolTip(ssp_btn, "Apply Signal Space Projection for artifact removal")

        self.apply_csd = tk.BooleanVar(value=False)
        csd_btn = ttk.Checkbutton(processing_frame,
                                  text="Apply CSD",
                                  variable=self.apply_csd)
        csd_btn.grid(row=2, column=0, sticky=tk.W)
        ToolTip(csd_btn, "Apply Current Source Density transformation")

        # Right column - Visualization & Analysis Options
        viz_frame = ttk.LabelFrame(right_frame, text="Visualization & Analysis", padding="5")
        viz_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        self.plot_preproc = tk.BooleanVar()
        plot_btn = ttk.Checkbutton(viz_frame,
                                   text="Save preprocessing plots",
                                   variable=self.plot_preproc)
        plot_btn.grid(row=0, column=0, sticky=tk.W)
        ToolTip(plot_btn, "Show preprocessing visualization steps")

        self.plot_raw = tk.BooleanVar(value=False)
        raw_btn = ttk.Checkbutton(viz_frame,
                                  text="Plot raw data",
                                  variable=self.plot_raw)
        raw_btn.grid(row=1, column=0, sticky=tk.W)
        ToolTip(raw_btn, "Plot raw data (may take time)")

        self.save_evoked = tk.BooleanVar()
        evoked_btn = ttk.Checkbutton(viz_frame,
                                     text="Save Evoked Plot",
                                     variable=self.save_evoked)
        evoked_btn.grid(row=2, column=0, sticky=tk.W)
        ToolTip(evoked_btn, "Display evoked response plots")

        # Bottom frame - Data Format Settings
        format_frame = ttk.LabelFrame(options_frame, text="Data Settings", padding="5")
        format_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Data Format
        ttk.Label(format_frame, text="Data Format:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.data_format = tk.StringVar(value='neurone')
        format_combo = ttk.Combobox(format_frame, textvariable=self.data_format, width=15)
        format_combo['values'] = ('neurone', 'brainvision', 'edf', 'cnt', 'eeglab', 'auto')
        format_combo.grid(row=0, column=1, padx=5, pady=2)
        format_combo.state(['readonly'])
        ToolTip(format_combo, "Format of input data files")

        # EEGLAB Units
        ttk.Label(format_frame, text="EEGLAB Units:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.eeglab_units = tk.StringVar(value='auto')
        units_combo = ttk.Combobox(format_frame, textvariable=self.eeglab_units, width=15)
        units_combo['values'] = ('auto', 'mm', 'm', 'cm')
        units_combo.grid(row=0, column=3, padx=5, pady=2)
        units_combo.state(['readonly'])
        ToolTip(units_combo, "Units for EEGLAB channel positions")

        # Stim Channel
        ttk.Label(format_frame, text="Stim Channel:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.stim_channel = tk.StringVar(value='STI 014')
        stim_entry = ttk.Entry(format_frame, textvariable=self.stim_channel, width=15)
        stim_entry.grid(row=1, column=1, padx=5, pady=2)
        ToolTip(stim_entry, "Name of the stimulus channel in your data")

        # Validation Options
        validation_frame = ttk.LabelFrame(options_frame, text="Validation Options", padding="5")
        validation_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.validate_teps = tk.BooleanVar()
        validate_btn = ttk.Checkbutton(validation_frame,
                                       text="Validate TEPs",
                                       variable=self.validate_teps)
        validate_btn.grid(row=0, column=0, sticky=tk.W, padx=5)
        ToolTip(validate_btn, "Perform TEP validation and generate reports")

        self.save_validation = tk.BooleanVar()
        save_btn = ttk.Checkbutton(validation_frame,
                                   text="Save Validation Reports",
                                   variable=self.save_validation)
        save_btn.grid(row=0, column=1, sticky=tk.W, padx=5)
        ToolTip(save_btn, "Save validation reports and plots to output directory")

        self.research_stats = tk.BooleanVar()
        stats_btn = ttk.Checkbutton(validation_frame,
                                    text="Generate Research Statistics",
                                    variable=self.research_stats)
        stats_btn.grid(row=0, column=2, sticky=tk.W, padx=5)
        ToolTip(stats_btn, "Generate additional research statistics")

    def create_advanced_options(self, parent):
        # Notebook for advanced parameters
        notebook = ttk.Notebook(parent)
        notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Initialize params dictionary
        self.params = {}

        # 1. General Settings Tab (basic setup params)
        preproc_frame = ttk.Frame(notebook, padding="5")
        notebook.add(preproc_frame, text="General")

        # Create frame specifically for sampling rates
        sampling_frame = ttk.LabelFrame(preproc_frame, text="Downsampling", padding="5")
        sampling_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)

        self.add_parameter_group(sampling_frame, {
            'Initial Sampling Rate (Hz)': ('initial_sfreq', 1000),
            'Final Sampling Rate (Hz)': ('final_sfreq', 725)
        })

        ttk.Label(sampling_frame,
                  text="Data will first be downsampled to initial rate,\n" +
                       "then to final rate after preprocessing",
                  wraplength=250).grid(row=2, column=0, columnspan=2, pady=5)

        other_frame = ttk.LabelFrame(preproc_frame, text="Other Settings", padding="5")
        other_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        self.add_parameter_group(other_frame, {
            'Random Seed': ('random_seed', 42),
        })

        # 2. TMS Artifact Removal Tab
        tms_frame = ttk.Frame(notebook, padding="5")
        notebook.add(tms_frame, text="TMS Artifact")

        stage1_frame = ttk.LabelFrame(tms_frame, text="Stage 1 (Initial Removal)", padding="5")
        stage1_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        self.add_parameter_group(stage1_frame, {
            'Initial Cut Start (ms)': ('initial_cut_start', -2),
            'Initial Cut End (ms)': ('initial_cut_end', 10),
            'Initial Interp Window (ms)': ('initial_interp_window', 1.0),
        })

        stage2_frame = ttk.LabelFrame(tms_frame, text="Stage 2 (Extended Removal)", padding="5")
        stage2_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        self.add_parameter_group(stage2_frame, {
            'Extended Cut Start (ms)': ('extended_cut_start', -2),
            'Extended Cut End (ms)': ('extended_cut_end', 15),
            'Extended Interp Window (ms)': ('extended_interp_window', 5.0),
        })

        common_frame = ttk.LabelFrame(tms_frame, text="Common Settings", padding="5")
        common_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        self.add_parameter_group(common_frame, {
            'Interpolation Method': ('interpolation_method', 'cubic')
        })

        self.skip_second_artifact_removal = tk.BooleanVar()
        ttk.Checkbutton(stage2_frame, text="Skip Second TMS Artifact Removal",
                        variable=self.skip_second_artifact_removal).grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)

        # 3. Epoching Tab
        epoch_frame = ttk.Frame(notebook, padding="5")
        notebook.add(epoch_frame, text="Epoching")
        self.add_parameter_group(epoch_frame, {
            'Epoch Start Time (s)': ('epochs_tmin', -0.41),
            'Epoch End Time (s)': ('epochs_tmax', 0.41),
            'Baseline Start (ms)': ('baseline_start', -400),
            'Baseline End (ms)': ('baseline_end', -50),
            #'Amplitude Threshold (µV)': ('amplitude_threshold', 4500),
            'Bad Channels Threshold': ('bad_channels_threshold', 1),
            'Bad Epochs Threshold': ('bad_epochs_threshold', 2),
            'Substitute Zero Events With': ('substitute_zero_events_with', 10),
        })

        # 4. ICA and Muscle Artifacts Tab
        ica_frame = ttk.Frame(notebook, padding="5")
        notebook.add(ica_frame, text="ICA, filter & PARAFAC")

        # Configure grid weights to allow expansion
        ica_frame.columnconfigure(0, weight=1)
        ica_frame.columnconfigure(1, weight=1)

        # First ICA Frame
        first_ica_frame = ttk.LabelFrame(ica_frame, text="First ICA (TMS-Muscle)", padding="5")
        first_ica_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)

        self.first_ica_manual = tk.BooleanVar(value=True)
        first_manual_check = ttk.Checkbutton(first_ica_frame,
                                             text="Enable Manual Component Selection",
                                             variable=self.first_ica_manual)
        first_manual_check.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ToolTip(first_manual_check, "Enable interactive selection of ICA components to remove")

        ttk.Label(first_ica_frame, text="ICA Method:").grid(row=1, column=0, sticky=tk.W)
        self.params['ica_method'] = tk.StringVar(value='fastica')
        method_combo = ttk.Combobox(first_ica_frame,
                                    textvariable=self.params['ica_method'],
                                    values=['fastica', 'infomax'],
                                    state='readonly',
                                    width=10)
        method_combo.grid(row=1, column=1, padx=5, pady=2)

        # Filtering Parameters Frame
        filter_frame = ttk.LabelFrame(ica_frame, text="Filtering Parameters", padding="5")
        filter_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)

        filter_params = {
            'Low Frequency (Hz)': ('l_freq', 0.1),
            'High Frequency (Hz)': ('h_freq', 45),
            'Notch Frequency (Hz)': ('notch_freq', 50),
            'Notch Width': ('notch_width', 2),
        }

        row = 0
        for label, (param_name, default_val) in filter_params.items():
            ttk.Label(filter_frame, text=label).grid(row=row, column=0, sticky=tk.W)
            self.params[param_name] = tk.StringVar(value=str(default_val))
            ttk.Entry(filter_frame,
                      textvariable=self.params[param_name],
                      width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1

        # Filter Checkboxes
        self.filter_raw = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_frame, text="Filter raw data (Not after ICA 1)",
                        variable=self.filter_raw).grid(row=row, column=0, columnspan=2, padx=5, pady=2, sticky=tk.W)
        row += 1

        self.mne_filter_epochs = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_frame, text="Use built in filters in mne",
                        variable=self.mne_filter_epochs).grid(row=row, column=0, columnspan=2, padx=5, pady=2,
                                                              sticky=tk.W)

        # Muscle Artifact Frame
        muscle_frame = ttk.LabelFrame(ica_frame, text="Muscle Artifacts (PARAFAC)", padding="5")
        muscle_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)

        muscle_params = {
            'Muscle Window Start (s)': ('muscle_window_start', 0.005),
            'Muscle Window End (s)': ('muscle_window_end', 0.030),
            'Threshold Factor': ('threshold_factor', 1.0),
            'Number of PARAFAC Components': ('n_components', 5),
        }

        row = 0
        for label, (param_name, default_val) in muscle_params.items():
            ttk.Label(muscle_frame, text=label).grid(row=row, column=0, sticky=tk.W)
            self.params[param_name] = tk.StringVar(value=str(default_val))
            ttk.Entry(muscle_frame,
                      textvariable=self.params[param_name],
                      width=10).grid(row=row, column=1, padx=5, pady=2)
            row += 1

        # Second ICA Frame
        second_ica_frame = ttk.LabelFrame(ica_frame, text="Second ICA (Other Artifacts)", padding="5")
        second_ica_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)

        self.second_ica_manual = tk.BooleanVar(value=True)
        second_manual_check = ttk.Checkbutton(second_ica_frame,
                                              text="Enable Manual Component Selection",
                                              variable=self.second_ica_manual)
        second_manual_check.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        ttk.Label(second_ica_frame, text="ICA Method:").grid(row=1, column=0, sticky=tk.W)
        self.params['second_ica_method'] = tk.StringVar(value='fastica')
        second_method_combo = ttk.Combobox(second_ica_frame,
                                           textvariable=self.params['second_ica_method'],
                                           values=['fastica', 'infomax'],
                                           state='readonly',
                                           width=10)
        second_method_combo.grid(row=1, column=1, padx=5, pady=2)


        # 5. Filtering Tab
        #filter_frame = ttk.Frame(notebook, padding="5")
        #notebook.add(filter_frame, text="Filtering")


        # 5. SSP & CSD Tab
        transform_frame = ttk.Frame(notebook, padding="5")
        notebook.add(transform_frame, text="SSP & CSD")
        self.add_parameter_group(transform_frame, {
            'SSP EEG Components': ('ssp_n_eeg', 2),
            'Lambda2 (CSD)': ('lambda2', 1e-3),
            'Stiffness (CSD)': ('stiffness', 3),
        })

        # 7. TEP Analysis Tab
        tep_frame = ttk.Frame(notebook, padding="5")
        notebook.add(tep_frame, text="TEP Analysis")

        analysis_type_frame = ttk.LabelFrame(tep_frame, text="Analysis Type", padding="5")
        analysis_type_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        self.params['tep_analysis_type'] = tk.StringVar(value='gmfa')
        ttk.Radiobutton(analysis_type_frame, text="GMFA",
                        variable=self.params['tep_analysis_type'],
                        value='gmfa').grid(row=0, column=0, padx=5)
        ttk.Radiobutton(analysis_type_frame, text="ROI",
                        variable=self.params['tep_analysis_type'],
                        value='roi').grid(row=0, column=1, padx=5)

        roi_frame = ttk.LabelFrame(tep_frame, text="ROI Channels", padding="5")
        roi_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        self.params['roi_channels'] = tk.StringVar(value='C3, C4')
        ttk.Label(roi_frame, text="Channels:").grid(row=0, column=0, sticky=tk.W)
        roi_entry = ttk.Entry(roi_frame, textvariable=self.params['roi_channels'], width=30)
        roi_entry.grid(row=0, column=1, padx=5)

        peak_frame = ttk.LabelFrame(tep_frame, text="Peak Detection", padding="5")
        peak_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)

        peak_params = {
            'Number of Samples': ('n_samples', 5),
            'Minimum Peak Distance (ms)': ('min_peak_distance', 10),
            'Prominence': ('prominence', 0.01)
        }
        self.add_parameter_group(peak_frame, peak_params)

        ttk.Label(peak_frame, text="Peak Selection Method:").grid(row=len(peak_params), column=0, sticky=tk.W)
        self.params['peak_method'] = tk.StringVar(value='largest')
        method_combo = ttk.Combobox(peak_frame,
                                    textvariable=self.params['peak_method'],
                                    values=['largest', 'centre'],
                                    state='readonly',
                                    width=10)
        method_combo.grid(row=len(peak_params), column=1, padx=5)

        comp_frame = ttk.LabelFrame(tep_frame, text="TEP Components", padding="5")
        comp_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)

        components_text = ("Default components:\n"
                           "N15 (12-18ms), P30 (25-35ms), N45 (36-57ms),\n"
                           "P60 (58-80ms), N100 (81-144ms), P180 (145-250ms)")
        ttk.Label(comp_frame, text=components_text).pack(pady=5)

        time_frame = ttk.LabelFrame(tep_frame, text="Analysis Time Windows", padding="5")
        time_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)

        window_params = {
            'Analysis Start (ms)': ('analysis_start', -100),
            'Analysis End (ms)': ('analysis_end', 400)
        }
        self.add_parameter_group(time_frame, window_params)

        # 8. PCIst Tab (Final analysis)
        pcist_frame = ttk.Frame(notebook, padding="5")
        notebook.add(pcist_frame, text="PCIst")
        self.add_parameter_group(pcist_frame, {
            'k Parameter': ('k', 1.2),
            'Min SNR': ('min_snr', 1.1),
            'Max Variance (%)': ('max_var', 99.0),
            'Number of Steps': ('n_steps', 100),
            'Response Start (ms)': ('response_start', 0),
            'Response End (ms)': ('response_end', 299)
        })

    def add_parameter_group(self, parent, params):
        for i, (label, (param, default)) in enumerate(params.items()):
            ttk.Label(parent, text=label + ":").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            self.params[param] = tk.StringVar(value=str(default))
            ttk.Entry(parent, textvariable=self.params[param], width=10).grid(row=i, column=1, padx=5, pady=2)
            
    def create_console(self, parent):
        # Console frame
        console_frame = ttk.LabelFrame(parent, text="Console Output", padding="5")
        console_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.console = tk.Text(console_frame, height=15, width=120)
        self.console.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self.console.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.console.configure(yscrollcommand=scrollbar.set)
        
    def create_control_buttons(self, parent):
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Run tmseegpy", command=self.run_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_analysis).pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(parent, length=300, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=3, pady=5)
        
    def browse_data_dir(self):
        directory = filedialog.askdirectory(initialdir=self.data_dir.get())
        if directory:
            self.data_dir.set(directory)
            
    def browse_output_dir(self):
        directory = filedialog.askdirectory(initialdir=self.output_dir.get())
        if directory:
            self.output_dir.set(directory)
            
    def redirect_output(self):
        class StdoutRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
            def write(self, string):
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
            def flush(self):
                pass
        sys.stdout = StdoutRedirector(self.console)

    def update_gui(self):
        """Ensure GUI remains responsive during long operations."""
        if self.running:
            # Process pending events
            self.root.update_idletasks()
            # Schedule next update
            self.root.after(100, self.update_gui)
    '''
    def run_analysis_thread(self, args):
        """Modified thread function to handle both continuous and epoched processing"""
        try:
            # Redirect output to console
            self.redirect_output()

            # Configure matplotlib for thread-safe operation
            import matplotlib
            matplotlib.use('Agg')

            # Import needed modules
            from ..run import process_subjects, process_continuous_data
            import builtins
            import queue

            # Make sure GUI root is available
            if not hasattr(builtins, 'GUI_MAIN_ROOT'):
                builtins.GUI_MAIN_ROOT = self.root

            # Create and start GUI queue processing
            gui_queue = queue.Queue()

            def check_gui_queue():
                try:
                    while True:
                        #callback, args = gui_queue.get_nowait()
                        callback(*args)
                except queue.Empty:
                    pass
                if self.running:
                    self.root.after(100, check_gui_queue)

            def schedule_on_gui(callback, *args):
                gui_queue.put((callback, args))

            builtins.schedule_on_gui = schedule_on_gui
            self.root.after(0, check_gui_queue)

            # Reset stop flag
            setattr(builtins, 'STOP_PROCESSING', False)

            # Run appropriate processing based on mode
            if args.continuous_mode:
                results = process_continuous_data(args)
                #schedule_on_gui(self.continuous_analysis_complete, results)
            else:
                pcists = process_subjects(args)
                if not getattr(builtins, 'STOP_PROCESSING', False):
                    schedule_on_gui(self.analysis_complete, pcists)

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            gui_queue.put((self.analysis_error, (error_msg,)))
            '''

            
    def analysis_complete(self, pcists):
        self.progress.stop()
        self.running = False
        
        # Build completion message
        message = f"Analysis completed successfully!\nPCIst values: {pcists}"
        
        if self.save_validation.get():
            output_dir = self.output_dir.get()
            message += f"\n\nValidation reports and plots saved to:\n{output_dir}"
            
            # Optional: List saved files
            try:
                validation_files = [f for f in os.listdir(output_dir) 
                                if f.startswith(('tep_validation_', 'pcist_'))]
                if validation_files:
                    message += "\n\nSaved files:"
                    for f in validation_files[:5]:  # Show first 5 files
                        message += f"\n- {f}"
                    if len(validation_files) > 5:
                        message += f"\n... and {len(validation_files)-5} more files"
            except Exception:
                pass
                
        messagebox.showinfo("Complete", message)

    def continuous_analysis_complete(self, raw_list):
        """Completion handler for continuous processing mode"""
        self.progress.stop()
        self.running = False

        # Build completion message
        n_sessions = len(raw_list) if raw_list else 0
        message = f"Continuous processing completed successfully!\n"
        message += f"Processed {n_sessions} session{'s' if n_sessions != 1 else ''}"

        output_dir = self.output_dir.get()
        try:
            processed_files = [f for f in os.listdir(output_dir)
                               if f.endswith('_processed_continuous_raw.fif')]
            if processed_files:
                message += "\n\nProcessed files:"
                for f in processed_files[:5]:
                    message += f"\n- {f}"
                if len(processed_files) > 5:
                    message += f"\n... and {len(processed_files) - 5} more files"
        except Exception:
            pass

        messagebox.showinfo("Complete", message)
        
    def analysis_error(self, error_msg):
        """Enhanced error handling with full traceback"""
        import traceback
        import sys
        
        self.progress.stop()
        self.running = False
        
        # Get the full traceback
        exc_info = sys.exc_info()
        
        # Format error message with traceback
        error_detail = "Error Traceback:\n"
        
        if exc_info[0] is not None:  # If we have exception info
            exc_type, exc_value, exc_traceback = exc_info
            tb_list = traceback.extract_tb(exc_traceback)
            
            for filename, line, func, text in tb_list:
                error_detail += f"  File '{filename}', line {line}, in {func}\n"
                if text:
                    error_detail += f"    {text}\n"
            error_detail += f"\nError Type: {exc_type.__name__}\n"
        
        # Always include the error message
        error_detail += f"Error Message: {str(error_msg)}"
        
        # Show error in GUI
        messagebox.showerror("Error", error_detail)
        
        # Also print to console for debugging
        print("\nFull Error Details:")
        print(error_detail)
        
    def cleanup_after_stop(self):
        """Clean up after stopping the analysis"""
        import matplotlib.pyplot as plt
        plt.close('all')  # Close all matplotlib figures
        
        # Reset the stop flag
        import builtins
        setattr(builtins, 'STOP_PROCESSING', False)
        
        # Clear any remaining progress indication
        self.progress.stop()
        self.running = False
        print("\nAnalysis stopped and cleaned up")

    def stop_analysis(self):
        if self.running:
            print("\nStopping analysis (this may take a moment)...")
            self.running = False
            import builtins
            setattr(builtins, 'STOP_PROCESSING', True)
            self.root.after(1000, self.cleanup_after_stop)  # Cleanup after 1 second

import builtins

if __name__ == "__main__":
    root = tk.Tk()
    builtins.GUI_MAIN_ROOT = root  # <-- new line
    app = TMSEEG_GUI(root)
    root.mainloop()
