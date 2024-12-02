import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import sys
import io

class ToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

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
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class TMSEEG_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TMS-EEG Preprocessing and PCIst")
        
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
            'ds_sfreq': {'min': 100, 'max': 5000, 'type': float},
            'random_seed': {'min': 0, 'max': 1000000, 'type': int},
            'bad_channels_threshold': {'min': 0, 'max': 10, 'type': float},
            'bad_epochs_threshold': {'min': 0, 'max': 10, 'type': float},
            'tms_muscle_thresh': {'min': 0, 'max': 10, 'type': float},
            'n_components': {'min': 1, 'max': 100, 'type': int},
            'ssp_n_eeg': {'min': 1, 'max': 10, 'type': int},
            'l_freq': {'min': 0, 'max': 100, 'type': float},
            'h_freq': {'min': 1, 'max': 1000, 'type': float},
            'notch_freq': {'min': 0, 'max': 1000, 'type': float},
            'notch_width': {'min': 0, 'max': 10, 'type': float},
            'cut_times_tms_start': {'min': -100, 'max': 0, 'type': float},
            'cut_times_tms_end': {'min': 0, 'max': 100, 'type': float},
            'interp_window_start': {'min': 0, 'max': 100, 'type': float},
            'interp_window_end': {'min': 0, 'max': 100, 'type': float},
            'epochs_tmin': {'min': -2, 'max': 0, 'type': float},
            'epochs_tmax': {'min': 0, 'max': 2, 'type': float},
            'baseline_start': {'min': -1000, 'max': 0, 'type': int},
            'baseline_end': {'min': -1000, 'max': 0, 'type': int},
            'k': {'min': 0, 'max': 10, 'type': float},
            'min_snr': {'min': 0, 'max': 10, 'type': float},
            'max_var': {'min': 0, 'max': 100, 'type': float},
            'n_steps': {'min': 10, 'max': 1000, 'type': int},
            'n_clusters': {'min': 2, 'max': 20, 'type': int},
            'n_resamples': {'min': 1, 'max': 100, 'type': int},
            'n_samples': {'min': 100, 'max': 10000, 'type': int},
            'min_peak_distance': {'min': 1, 'max': 100, 'type': int},
            'substitute_zero_events_with': {'min': 1, 'max': 100, 'type': int},
            'threshold_factor': {'min': 0.1, 'max': 10.0, 'type': float},
            'lambda2': {'min': 1e-6, 'max': 1e-4, 'type': float},
            'stiffness': {'min': 1, 'max': 10, 'type': int},
            'ica_method': {'type': 'str', 'choices': ['fastica', 'infomax']},
            'second_ica_method': {'type': 'str', 'choices': ['fastica', 'infomax']},
            'interpolation_method': {'type': 'str', 'choices': ['cubic', 'linear']},
            'response_start': {'min': -1000, 'max': 1000, 'type': int},
            'response_end': {'min': -1000, 'max': 1000, 'type': int},
            
        }

    def init_tooltips(self):
        self.tooltips = {
            'ds_sfreq': 'Downsampling frequency in Hz',
            'random_seed': 'Random seed for reproducibility',
            'bad_channels_threshold': 'Threshold for detecting bad channels',
            'bad_epochs_threshold': 'Threshold for detecting bad epochs',
            'tms_muscle_thresh': 'Threshold for detecting muscle artifacts',
            'n_components': 'Number of components for PARAFAC',
            'ssp_n_eeg': 'Number of EEG components for SSP',
            'l_freq': 'Lower frequency cutoff for filtering',
            'h_freq': 'Upper frequency cutoff for filtering',
            'notch_freq': 'Frequency for notch filter',
            'notch_width': 'Width of the notch filter',
            'cut_times_tms_start': 'Start time for cutting TMS artifact (ms)',
            'cut_times_tms_end': 'End time for cutting TMS artifact (ms)',
            'interp_window_start': 'Start time for interpolation window (ms)',
            'interp_window_end': 'End time for interpolation window (ms)',
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
            'fix_artifact_window_start': 'Start time for fixing TMS artifact window (s)',
            'fix_artifact_window_end': 'End time for fixing TMS artifact window (s)',
            'threshold_factor': 'Threshold factor for muscle artifact cleaning',
            'muscle_window_start': 'Start time for muscle artifact window (s)',
            'muscle_window_end': 'End time for muscle artifact window (s)',
            'ica_method': 'Method for first ICA (fastica or infomax)',
            'second_ica_method': 'Method for second ICA (fastica or infomax)',
            'lambda2': 'Regularization parameter for CSD transformation',
            'stiffness': 'Stiffness parameter for CSD transformation',
            'response_start': 'Start of response window in ms',
            'response_end': 'End of response window in ms',
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
        
        # Toggle button for advanced options
        ttk.Checkbutton(main_frame, text="Show Advanced Options", 
                        variable=self.show_advanced,
                        command=self.toggle_advanced_options).grid(
                            row=1, column=0, sticky=tk.W, pady=(10,0))
        
        # Create frames for options (hidden by default)
        self.options_frame = ttk.Frame(main_frame)
        self.options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        self.options_frame.grid_remove()  # Hide initially
        
        # Create basic and advanced options in the hidden frame
        self.create_basic_options(self.options_frame)
        self.create_advanced_options(self.options_frame)
        
        # Console always visible at bottom
        self.create_console(main_frame)
        self.create_control_buttons(main_frame)

    def toggle_advanced_options(self):
        if self.show_advanced.get():
            self.options_frame.grid()
        else:
            self.options_frame.grid_remove()

    def create_directory_selectors(self, parent):
        dir_frame = ttk.LabelFrame(parent, text="Data Selection", padding="10")
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

    def run_analysis(self):
        if self.running:
            return
            
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
            'data_dir': self.data_dir.get(),
            'output_dir': self.output_dir.get(),
            'plot_preproc': self.plot_preproc.get(),
            'clean_muscle_artifacts': self.clean_muscle.get(),
            'show_evoked': self.show_evoked.get(),
            'research': self.research_stats.get(),
            'preproc_qc': self.preproc_qc.get()
        }
        
        # Add numerical parameters with proper type conversion
        for param, var in self.params.items():
            try:
                value = float(var.get())
                if self.validation_rules[param]['type'] == int:
                    value = round(value)
                args_dict[param] = value
            except (ValueError, KeyError):
                messagebox.showerror("Error", f"Invalid value for {param}")
                self.progress.stop()
                self.running = False
                return
        
        # Create Args object
        class Args:
            pass
        args = Args()
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
        
        # Checkboxes for basic options
        self.plot_preproc = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Plot Preprocessing Steps", 
                       variable=self.plot_preproc).grid(row=0, column=0, sticky=tk.W)
        
        self.clean_muscle = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Clean Muscle Artifacts with PARAFAC decomposition", 
                       variable=self.clean_muscle).grid(row=0, column=1, sticky=tk.W)
        
        self.show_evoked = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Show Evoked Plot", 
                       variable=self.show_evoked).grid(row=1, column=0, sticky=tk.W)
        
        self.research_stats = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Generate Research Statistics", 
                       variable=self.research_stats).grid(row=1, column=1, sticky=tk.W)
        
        self.preproc_qc = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Generate Preprocessing QC", 
                       variable=self.preproc_qc).grid(row=2, column=1, sticky=tk.W)
        
        plot_button = ttk.Checkbutton(options_frame, text="Plot Preprocessing Steps", 
                variable=self.plot_preproc)
        plot_button.grid(row=0, column=0, sticky=tk.W)
        ToolTip(plot_button, "Show preprocessing visualization steps")
        
    def create_advanced_options(self, parent):
        # Notebook for advanced parameters
        notebook = ttk.Notebook(parent)
        notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Create different tabs for parameter categories
        self.params = {}
        
        # Preprocessing Parameters
        preproc_frame = ttk.Frame(notebook, padding="5")
        notebook.add(preproc_frame, text="Preprocessing")
        self.add_parameter_group(preproc_frame, {
            'Downsampling Frequency (Hz)': ('ds_sfreq', 725),
            'Random Seed': ('random_seed', 42),
            'Bad Channels Threshold': ('bad_channels_threshold', 3),
            'Bad Epochs Threshold': ('bad_epochs_threshold', 3),
            'TMS Muscle Threshold': ('tms_muscle_thresh', 3.0),
            'Number of PARAFAC Components': ('n_components', 5),
            'SSP EEG Components': ('ssp_n_eeg', 2),
            'Substitute Zero Events With': ('substitute_zero_events_with', 10),
        })
        
        # Filtering Parameters
        filter_frame = ttk.Frame(notebook, padding="5")
        notebook.add(filter_frame, text="Filtering")
        self.add_parameter_group(filter_frame, {
            'Low Frequency (Hz)': ('l_freq', 0.1),
            'High Frequency (Hz)': ('h_freq', 45),
            'Notch Frequency (Hz)': ('notch_freq', 50),
            'Notch Width': ('notch_width', 2),
        })
        
        # TMS Parameters
        tms_frame = ttk.Frame(notebook, padding="5")
        notebook.add(tms_frame, text="TMS Settings")
        self.add_parameter_group(tms_frame, {
            'Cut Times TMS Start (ms)': ('cut_times_tms_start', -2),
            'Cut Times TMS End (ms)': ('cut_times_tms_end', 10),
            'Interpolation Window Start (ms)': ('interp_window_start', 20),
            'Interpolation Window End (ms)': ('interp_window_end', 20),
            'Fix Artifact Window Start': ('fix_artifact_window_start', -0.005),
            'Fix Artifact Window End': ('fix_artifact_window_end', 0.015),
        })
        
        # Muscle Artifact Parameters
        muscle_frame = ttk.Frame(notebook, padding="5")
        notebook.add(muscle_frame, text="Muscle Artifacts")
        self.add_parameter_group(muscle_frame, {
            'Muscle Window Start (s)': ('muscle_window_start', 0.005),
            'Muscle Window End (s)': ('muscle_window_end', 0.030),
            'Threshold Factor': ('threshold_factor', 1.0),
        })
        
        # ICA Parameters
        ica_frame = ttk.Frame(notebook, padding="5")
        notebook.add(ica_frame, text="ICA Settings")
        self.add_parameter_group(ica_frame, {
            'ICA Method': ('ica_method', 'fastica'),
            'Second ICA Method': ('second_ica_method', 'infomax'),
        })
        
        # CSD Parameters
        csd_frame = ttk.Frame(notebook, padding="5")
        notebook.add(csd_frame, text="CSD Settings")
        self.add_parameter_group(csd_frame, {
            'Lambda2': ('lambda2', 1e-5),
            'Stiffness': ('stiffness', 4),
        })
        
        # Epoch Parameters
        epoch_frame = ttk.Frame(notebook, padding="5")
        notebook.add(epoch_frame, text="Epoching")
        self.add_parameter_group(epoch_frame, {
            'Epoch Start Time (s)': ('epochs_tmin', -0.41),
            'Epoch End Time (s)': ('epochs_tmax', 0.41),
            'Baseline Start (ms)': ('baseline_start', -400),
            'Baseline End (ms)': ('baseline_end', -50),
        })
        
        # PCIst Parameters
        pcist_frame = ttk.Frame(notebook, padding="5")
        notebook.add(pcist_frame, text="PCIst")
        self.add_parameter_group(pcist_frame, {
            'k Parameter': ('k', 1.2),
            'Min SNR': ('min_snr', 1.1),
            'Max Variance (%)': ('max_var', 99.0),
            'Number of Steps': ('n_steps', 100),
        })
        
        
    def add_parameter_group(self, parent, params):
        for i, (label, (param, default)) in enumerate(params.items()):
            ttk.Label(parent, text=label + ":").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            self.params[param] = tk.StringVar(value=str(default))
            ttk.Entry(parent, textvariable=self.params[param], width=10).grid(row=i, column=1, padx=5, pady=2)
            
    def create_console(self, parent):
        # Console frame
        console_frame = ttk.LabelFrame(parent, text="Console Output", padding="5")
        console_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.console = tk.Text(console_frame, height=15, width=120)
        self.console.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self.console.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.console.configure(yscrollcommand=scrollbar.set)
        
    def create_control_buttons(self, parent):
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Get PCIst", command=self.run_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_analysis).pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(parent, length=300, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, pady=5)
        
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
        
    def run_analysis(self):
        if self.running:
            return
                
        self.console.delete(1.0, tk.END)
        self.progress.start()
        self.running = True
        
        # Prepare arguments
        args_dict = {
            # Directory and basic options
            'data_dir': self.data_dir.get(),
            'output_dir': self.output_dir.get(),
            'plot_preproc': self.plot_preproc.get(),
            'clean_muscle_artifacts': self.clean_muscle.get(),
            'show_evoked': self.show_evoked.get(),
            'research': self.research_stats.get(),
            'preproc_qc': self.preproc_qc.get(),
            
            # Additional arguments with default values
            'substitute_zero_events_with': 10,
            'ica_method': 'fastica',
            'second_ica_method': 'infomax',
            'interpolation_method': 'cubic',
            'no_second_ICA': False,
            'apply_csd': False,
            'embed': False,
            'fix_artifact_window_start': -0.005,
            'fix_artifact_window_end': 0.015,
            'threshold_factor': 1.0,
            'muscle_window_start': 0.005,
            'muscle_window_end': 0.030,
            'lambda2': 1e-5,
            'stiffness': 4,
            'response_start': 0,        # Add this
            'response_end': 299,   
        }
        
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
        
        # Create Args object
        class Args:
            pass
        args = Args()
        for key, value in args_dict.items():
            setattr(args, key, value)
        
        # Run analysis in separate thread
        thread = threading.Thread(target=self.run_analysis_thread, args=(args,))
        thread.daemon = True
        thread.start()
        
    def run_analysis_thread(self, args):
        try:
            self.redirect_output()
            # Set matplotlib backend to avoid GUI issues in thread
            import matplotlib
            matplotlib.use('Agg')
            
            from run import process_subjects
            pcists = process_subjects(args)
            self.root.after(0, self.analysis_complete, pcists)
        except Exception as e:
            self.root.after(0, self.analysis_error, str(e))
            
    def analysis_complete(self, pcists):
        self.progress.stop()
        self.running = False
        messagebox.showinfo("Complete", f"Analysis completed successfully!\nPCIst values: {pcists}")
        
    def analysis_error(self, error_msg):
        self.progress.stop()
        self.running = False
        messagebox.showerror("Error", f"An error occurred:\n{error_msg}")
        
    def stop_analysis(self):
        if self.running:
            self.running = False
            self.progress.stop()
            messagebox.showinfo("Stopped", "Analysis stopped by user")

if __name__ == "__main__":
    root = tk.Tk()
    app = TMSEEG_GUI(root)
    root.mainloop()