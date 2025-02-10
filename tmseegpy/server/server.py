# tmseegpy/server/server.py

# Standard library imports
import sys
import os
import json
import time
from queue import Queue, Empty
import threading
import tempfile
import traceback
from pathlib import Path
from multiprocessing import Process, Pipe
import argparse
import shutil
import builtins
import logging
import appdirs

# Third-party imports
from flask import Flask, request, jsonify, Blueprint, current_app
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import mne
from PyQt6.QtWidgets import QApplication

# Project imports
from tmseegpy.run import process_subjects, setup_qt_plugin_path
from tmseegpy.preproc import TMSEEGPreprocessor
from tmseegpy.dataloader import TMSEEGLoader
from .server_logger import ServerLogger


# Configure environment
if sys.platform == "darwin":
    os.environ["QT_QPA_PLATFORM"] = "cocoa"


# Constants
ALLOWED_EXTENSIONS = {'ses', 'set', 'eeg', 'vhdr', 'edf', 'cnt', 'fif', 'vmrk', 'fdt'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024 * 1024  # 5GB

ica_selection_queue = Queue()
ica_result_queue = Queue()

# --- Global State and Logging ---
current_progress = {  # Initialize global state
    'status': 'idle',
    'progress': 0,
    'step': '',
    'logs': [],
    'results': None,
    'error': None
}

logger = logging.getLogger(__name__)  # Initialize logger
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('server.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

setup_qt_plugin_path()

api_bp = Blueprint('api', __name__, url_prefix='/api')

app = None  # Flask app
socketio = None  # SocketIO instance
server_logger = None

output_capturer = None

def create_directories(base_dir):
    """Create necessary directories"""
    upload_folder = base_dir / 'uploads'
    tmseeg_data_dir = upload_folder / 'TMSEEG'
    upload_folder.mkdir(exist_ok=True, parents=True) # Ensure parent dirs exists
    tmseeg_data_dir.mkdir(exist_ok=True)
    return upload_folder, tmseeg_data_dir


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER, TMSEEG_DATA_DIR = create_directories(BASE_DIR) # Moved the creation of the directories here. It is called only in this file, and it also fixes the name error


# --- Helper Functions ---


def setup_matplotlib():
    """Setup matplotlib backend"""
    try:
        import matplotlib
        matplotlib.use('agg') # Use non-interactive backend
        import matplotlib.pyplot as plt
        plt.ioff()
    except Exception as e:
        print(f"Error setting Matplotlib backend: {e}")
        traceback.print_exc()

# Global variable to store the ICA callback
ica_callback = None

def map_frontend_to_backend_params(frontend_params):
    """
    Maps parameters from the frontend naming convention to backend parameters.
    Ensures alignment with run.py's argparse parameters.
    """
    required_params = ['dataFormat']
    for param in required_params:
        if param not in frontend_params:
            raise ValueError(f"Missing required parameter: {param}")
    data_dir = Path(frontend_params.get('dataDir', ''))
    if 'TMSEEG/TMSEEG' in str(data_dir):
        data_dir = str(data_dir).replace('TMSEEG/TMSEEG', 'TMSEEG')

    backend_params = {
        # Core Processing Parameters
        'data_dir': str(data_dir),
        'output_dir': frontend_params.get('outputDir', str(Path.cwd() / 'output')),
        'data_format': frontend_params.get('dataFormat', 'neurone'),
        'no_preproc_output': frontend_params.get('noPreprocessOutput', False),
        'no_pcist': frontend_params.get('noPcist', False),
        'eeglab_montage_units': frontend_params.get('eeglabMontageUnits', 'auto'),
        'stim_channel': frontend_params.get('stimChannel', 'STI 014'),
        'save_preproc': frontend_params.get('savePreproc', False),
        'random_seed': int(frontend_params.get('randomSeed', 42)),
        'substitute_zero_events_with': int(frontend_params.get('substituteZeroEventsWith', 10)),

        # Sampling and Window Parameters
        'initial_sfreq': float(frontend_params.get('initialSfreq', 1000)),
        'final_sfreq': float(frontend_params.get('finalSfreq', 725)),
        'initial_window_start': float(frontend_params.get('initialWindowStart', -2)),
        'initial_window_end': float(frontend_params.get('initialWindowEnd', 10)),
        'extended_window_start': float(frontend_params.get('extendedWindowStart', -2)),
        'extended_window_end': float(frontend_params.get('extendedWindowEnd', 15)),
        'initial_interp_window': float(frontend_params.get('initialInterpWindow', 1.0)),
        'extended_interp_window': float(frontend_params.get('extendedInterpWindow', 5.0)),
        'interpolation_method': frontend_params.get('interpolationMethod', 'cubic'),

        # Processing Options
        'skip_second_artifact_removal': frontend_params.get('skipSecondArtifactRemoval', False),
        'mne_filter_epochs': frontend_params.get('mneFilterEpochs', False),
        'plot_raw': frontend_params.get('plotRaw', False),
        'filter_raw': frontend_params.get('filterRaw', False),

        # Filtering Parameters
        'l_freq': float(frontend_params.get('lFreq', 0.1)),
        'h_freq': float(frontend_params.get('hFreq', 45)),
        'raw_h_freq': float(frontend_params.get('rawHFreq', 250)),
        'notch_freq': float(frontend_params.get('notchFreq', 50)),
        'notch_width': float(frontend_params.get('notchWidth', 2)),

        # Epoch Parameters
        'epochs_tmin': float(frontend_params.get('epochsTmin', -0.41)),
        'epochs_tmax': float(frontend_params.get('epochsTmax', 0.41)),

        # Artifact Detection Parameters
        'bad_channels_threshold': float(frontend_params.get('badChannelsThreshold', 3)),
        'bad_epochs_threshold': float(frontend_params.get('badEpochsThreshold', 3)),
        'amplitude_threshold': float(frontend_params.get('amplitudeThreshold', 300.0)),

        # ICA Parameters
        'ica_method': frontend_params.get('icaMethod', 'fastica'),
        'first_ica_manual': frontend_params.get('firstIcaManual', True),
        'second_ica_manual': frontend_params.get('secondIcaManual', True),
        'no_first_ica': frontend_params.get('noFirstIca', False),
        'no_second_ica': frontend_params.get('noSecondIca', False),
        'second_ica_method': frontend_params.get('secondIcaMethod', 'fastica'),

        # Artifact Thresholds
        'blink_thresh': float(frontend_params.get('blinkThresh', 2.5)),
        'lat_eye_thresh': float(frontend_params.get('latEyeThresh', 2.0)),
        'noise_thresh': float(frontend_params.get('noiseThresh', 4.0)),
        'tms_muscle_thresh': float(frontend_params.get('tmsMuscleThresh', 2.0)),
        'muscle_thresh': float(frontend_params.get('muscleThresh', 0.6)),

        # Muscle Artifact Parameters
        'parafac_muscle_artifacts': frontend_params.get('parafacMuscleArtifacts', False),
        'muscle_window_start': float(frontend_params.get('muscleWindowStart', 0.005)),
        'muscle_window_end': float(frontend_params.get('muscleWindowEnd', 0.030)),
        'threshold_factor': float(frontend_params.get('thresholdFactor', 1.0)),
        'n_components': int(frontend_params.get('nComponents', 5)),

        # SSP and CSD Parameters
        'apply_ssp': frontend_params.get('applySsp', False),
        'ssp_n_eeg': int(frontend_params.get('sspNEeg', 2)),
        'apply_csd': frontend_params.get('applyCsd', False),
        'lambda2': float(frontend_params.get('lambda2', 1e-3)),
        'stiffness': int(frontend_params.get('stiffness', 4)),

        # TEP Analysis Parameters
        'save_evoked': frontend_params.get('saveEvoked', False),
        'analyze_teps': frontend_params.get('analyzeTeps', True),
        'save_validation': frontend_params.get('saveValidation', False),
        'tep_analysis_type': frontend_params.get('tepAnalysisType', 'gmfa'),
        'tep_roi_channels': frontend_params.get('tepRoiChannels', ['C3', 'C4']),
        'tep_method': frontend_params.get('tepMethod', 'largest'),
        'tep_samples': int(frontend_params.get('tepSamples', 5)),

        # Window Parameters
        'baseline_start': int(frontend_params.get('baselineStart', -400)),
        'baseline_end': int(frontend_params.get('baselineEnd', -50)),
        'response_start': int(frontend_params.get('responseStart', 0)),
        'response_end': int(frontend_params.get('responseEnd', 299)),

        # PCIst Parameters
        'k': float(frontend_params.get('k', 1.2)),
        'min_snr': float(frontend_params.get('minSnr', 1.1)),
        'max_var': float(frontend_params.get('maxVar', 99.0)),
        'embed': frontend_params.get('embed', False),
        'n_steps': int(frontend_params.get('nSteps', 100)),
        'pre_window_start': int(frontend_params.get('preWindowStart', -400)),
        'pre_window_end': int(frontend_params.get('preWindowEnd', -50)),
        'post_window_start': int(frontend_params.get('postWindowStart', 0)),
        'post_window_end': int(frontend_params.get('postWindowEnd', 300)),

        # Research Mode
        'research': frontend_params.get('research', False),
    }


    return backend_params


def get_ica_callback(gui_mode=False):
    """Get appropriate ICA callback based on mode"""
    if gui_mode:
        from ..ica_selector_gui.websocket_ica_selector import WebSocketICASelector
        def callback(ica_instance, inst, component_scores=None):
            try:
                print("Starting ICA component selection...")
                selector = WebSocketICASelector(ica_selection_queue, ica_result_queue)

                # Format component data first
                component_data = selector._get_component_data(ica_instance, inst)
                print(f"Generated data for {len(component_data)} components")

                # Put data in queue before emitting event
                ica_selection_queue.put({
                    "data": component_data,
                    "scores": component_scores
                })

                # Emit event to notify frontend
                print("Emitting ica_required event...")
                socketio.emit('ica_required', {
                    'componentCount': len(component_data)
                })

                # Wait for selection
                print("Waiting for component selection...")
                selected = selector.select_components(ica_instance, inst)
                print(f"Received selected components: {selected}")
                return selected

            except Exception as e:
                print(f"Error in ICA GUI callback: {str(e)}")
                import traceback
                traceback.print_exc()
                return []

        return callback
    else:
        # Use existing CLI callback for command line usage
        from tmseegpy.cli_ica_selector import get_cli_ica_callback
        return get_cli_ica_callback()

def create_args_from_params(params):
    """Convert mapped parameters to argparse.Namespace"""
    parser = argparse.ArgumentParser()

    # Add all arguments from run.py with their mapped values
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value)

    # Parse empty args first, then set values
    args = parser.parse_args([])
    for key, value in params.items():
        setattr(args, key, value)

    return args


def update_progress(step, percent):
    """Update the current progress state"""
    current_progress['step'] = step
    current_progress['progress'] = percent
    current_progress['logs'].append(f"{step}: {percent}%")


def cleanup_files():
    """Clean up temporary files after processing"""
    # Implement cleanup logic here if needed
    pass


def get_file_size(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)


def validate_file(file):
    """Validate uploaded file"""
    if not file:
        raise ValueError("No file provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext[1:] not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
    return True




def ensure_directories(output_dir):
    """Ensure output directory exists"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_parameters(params):
    """Validate incoming parameters"""
    required_params = ['processingMode', 'dataDir', 'outputDir']
    for param in required_params:
        if param not in params:
            raise ValueError(f"Missing required parameter: {param}")

    numeric_validations = {
        'epochsTmin': (-10, 0),
        'epochsTmax': (0, 10),
        'lFreq': (0, 100),
        'hFreq': (0, 1000),
        'notchFreq': (0, 1000),
        'amplitudeThreshold': (0, 1000),
        'initialSfreq': (0, 10000),
        'finalSfreq': (0, 10000),
    }

    for param, (min_val, max_val) in numeric_validations.items():
        if param in params:
            try:
                value = float(params[param])
                if not min_val <= value <= max_val:
                    raise ValueError(f"Parameter {param} must be between {min_val} and {max_val}")
            except ValueError:
                raise ValueError(f"Invalid numeric value for parameter: {param}")


def handle_file_upload(file, session_dir, request_files=None):
    """Handle file upload with proper directory structure"""
    try:
        validate_file(file)
        filename = secure_filename(file.filename)
        file_path = None

        if filename.endswith('.ses'):
            file_path = TMSEEG_DATA_DIR / filename
            file.save(str(file_path))

            # Handle NeurOne naming convention
            if filename.startswith('NeurOne-'):
                # Remove 'NeurOne-' prefix and '.ses' extension
                session_name = filename.replace('NeurOne-', '')[:-4]
            else:
                session_name = filename[:-4]

            data_folder = TMSEEG_DATA_DIR / session_name
            data_folder.mkdir(exist_ok=True)
        else:
            file_path = session_dir / filename
            file.save(str(file_path))

            # Handle related files
            if request_files and file_path.suffix.lower() in ['.vhdr', '.set']:
                base_name = file_path.stem
                related_extensions = ['.eeg', '.vmrk'] if file_path.suffix.lower() == '.vhdr' else ['.fdt']

                for related_file in request_files:
                    if (related_file.filename.startswith(base_name) and
                            any(related_file.filename.lower().endswith(ext) for ext in related_extensions)):
                        related_path = session_dir / secure_filename(related_file.filename)
                        related_file.save(str(related_path))

        return file_path
    except Exception as e:
        raise ValueError(f"File upload failed: {str(e)}")


# Global window manager instance
window_manager = None

# Create the Flask application

def init_app(user_data_dir):
    user_data_dir = Path(user_data_dir)
    global app, socketio, current_progress, server_logger, output_capturer, UPLOAD_FOLDER, TMSEEG_DATA_DIR

    # Initialize QApplication in the main thread
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt

    if QApplication.instance() is None:
        import sys
        qt_app = QApplication(sys.argv)
        # Configure Qt for macOS
        if sys.platform == "darwin":
            qt_app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar)
            import os
            os.environ["QT_MAC_WANTS_LAYER"] = "1"
            os.environ["QT_QPA_PLATFORM"] = "cocoa"
        print("Created QApplication instance in main thread")

    # Use Qt5Agg backend for matplotlib
    import os
    import matplotlib
    matplotlib.use('Qt5Agg')

    # Initialize Flask app
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Configure app directories
    UPLOAD_FOLDER = user_data_dir / 'uploads'
    TMSEEG_DATA_DIR = UPLOAD_FOLDER / 'TMSEEG'

    # Create directories
    UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
    TMSEEG_DATA_DIR.mkdir(exist_ok=True)

    # Update app configuration
    app.config.update(
        UPLOAD_FOLDER=str(UPLOAD_FOLDER),
        MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
        SECRET_KEY=os.environ.get('SECRET_KEY', 'your-secret-key-here'),
        ALLOWED_EXTENSIONS=ALLOWED_EXTENSIONS
    )

    # Initialize SocketIO
    socketio = SocketIO(app,
                        cors_allowed_origins="*",
                        async_mode='gevent',
                        logger=True,
                        engineio_logger=True,
                        ping_timeout=60,
                        ping_interval=25)

    # Initialize other components
    server_logger = ServerLogger(socketio)
    output_capturer = OutputCapturer(socketio)
    current_progress = initialize_state()

    # Register blueprint
    app.register_blueprint(api_bp)

    return app, socketio, server_logger, output_capturer, UPLOAD_FOLDER, TMSEEG_DATA_DIR


@api_bp.route('/results', methods=['GET']) # Added route for results
def get_results(): # Fixed function definition
    try:
        results = current_progress.get('results', {}) # Access from global
        return jsonify({
            'status': 'success',
            'results': results
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/upload', methods=['POST'])
def handle_upload():
    """Unified upload handler for both files and directory structures"""
    try:
        print("Received upload request")
        print("Form data:", request.form)

        # Get directory information
        parent_dir = request.form.get('parentDirectory')
        tmseeg_dir = request.form.get('tmseegDirectory')

        if not parent_dir or not tmseeg_dir:
            return jsonify({'error': 'Missing directory information'}), 400

        # Convert to Path objects and resolve any duplicate TMSEEG directories
        parent_path = Path(parent_dir)
        tmseeg_path = Path(tmseeg_dir)

        # Remove duplicate TMSEEG from path if it exists
        if 'TMSEEG/TMSEEG' in str(tmseeg_path):
            tmseeg_path = Path(str(tmseeg_path).replace('TMSEEG/TMSEEG', 'TMSEEG'))

        if not parent_path.exists():
            return jsonify({'error': 'Parent directory does not exist'}), 400

        # Create TMSEEG directory if it doesn't exist
        tmseeg_path.mkdir(parents=True, exist_ok=True)

        # Verify the structure and collect sessions
        session_files = []

        # First, find all .ses files in the parent directory
        ses_files = list(parent_path.glob('*.ses'))

        if not ses_files:
            return jsonify({'error': 'No .ses files found in the parent directory'}), 400

        for ses_file in ses_files:
            if ses_file.name.startswith('NeurOne-'):
                session_name = ses_file.stem[8:]  # Remove 'NeurOne-' prefix
                session_dir = tmseeg_path / session_name

                # Create session directory
                session_dir.mkdir(exist_ok=True)

                # Copy the .ses file to TMSEEG directory
                dest_ses = tmseeg_path / ses_file.name
                if not dest_ses.exists():
                    shutil.copy2(ses_file, dest_ses)

                # Copy session data directory if it exists
                source_session_dir = parent_path / session_name
                if source_session_dir.exists() and source_session_dir.is_dir():
                    for item in source_session_dir.rglob('*'):
                        if item.is_file():
                            rel_path = item.relative_to(source_session_dir)
                            dest_item = session_dir / rel_path
                            dest_item.parent.mkdir(parents=True, exist_ok=True)
                            if not dest_item.exists():
                                shutil.copy2(item, dest_item)

                session_files.append(session_name)

        if not session_files:
            return jsonify({'error': 'No valid NeurOne session files processed'}), 400

        return jsonify({
            'message': 'Upload successful',
            'tmseeg_dir': str(tmseeg_path),
            'parent_dir': str(parent_path),
            'sessions': session_files,
            'session_count': len(session_files)
        }), 200

    except Exception as e:
        error_msg = f"Upload error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


@api_bp.route('/test_data_loading', methods=['POST'])
def test_data_loading():
    """Test endpoint to verify data loading"""
    try:
        params = request.get_json()
        session_dir = Path(params.get('session_dir'))
        tmseeg_dir = session_dir / 'TMSEEG'

        if not tmseeg_dir.exists():
            return jsonify({'error': 'TMSEEG directory not found'}), 404

        # Try loading the data
        loader = TMSEEGLoader(
            data_path=tmseeg_dir,
            format=params.get('data_format', 'auto'),
            substitute_zero_events_with=params.get('substitute_zero_events_with', 10),
            eeglab_montage_units=params.get('eeglab_montage_units', 'auto'),
            verbose=True
        )

        raw_list = loader.load_data()
        session_info = loader.get_session_info()

        return jsonify({
            'success': True,
            'message': 'Data loaded successfully',
            'sessions': len(raw_list),
            'session_info': session_info
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Cleanup scheduler
def cleanup_old_sessions():
    """Clean up old session directories"""
    try:
        current_time = time.time()
        for dir_path in TMSEEG_DATA_DIR.iterdir():
            if dir_path.is_dir():
                try:
                    dir_time = int(dir_path.name)
                    if current_time - dir_time > 24 * 60 * 60:  # 24 hours
                        shutil.rmtree(dir_path)
                except ValueError:
                    continue
    except Exception as e:
        app.logger.error(f"Cleanup error: {str(e)}")

def cleanup_scheduler():
    while True:
        cleanup_old_sessions()
        time.sleep(3600)  # Run every hour

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_scheduler, daemon=True)
cleanup_thread.start()


@api_bp.route('/process', methods=['POST'])
def start_processing():
    """Start processing endpoint"""
    try:
        params = request.get_json()
        if not params:
            return jsonify({'error': 'No parameters provided'}), 400

        # Fix the data directory path - remove any duplicate TMSEEG
        data_dir = Path(params.get('dataDir', ''))
        if 'TMSEEG/TMSEEG' in str(data_dir):
            data_dir = Path(str(data_dir).replace('TMSEEG/TMSEEG', 'TMSEEG'))
            params['dataDir'] = str(data_dir)

        print(f"Processing data directory: {data_dir}")

        # Verify the directory exists
        if not data_dir.exists():
            return jsonify({'error': f'Data directory does not exist: {data_dir}'}), 400

        # Ensure QApplication exists in main thread
        from PyQt6.QtWidgets import QApplication
        if QApplication.instance() is None:
            print("Creating QApplication instance before starting processing")
            QApplication([])

        # Import here to ensure Qt is initialized first

        # Add ICA callback to params
        params['ica_callback'] = get_ica_callback(gui_mode=True)

        # Start processing in a separate thread
        processing_thread = threading.Thread(target=process_task, args=(params,), daemon=True)
        processing_thread.start()

        return jsonify({
            'message': 'Processing started',
            'status': current_progress['status'],
            'dataDir': str(data_dir)
        }), 202

    except Exception as e:
        error_msg = f"Error starting processing: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


@api_bp.route('/ica_components', methods=['POST'])
def get_ica_components():
    """Get ICA component data for visualization"""
    try:
        ica_instance = request.json.get('ica_instance')
        raw_data = request.json.get('raw_data')
        epochs_data = request.json.get('epochs_data')

        # Convert data to format suitable for frontend
        component_data = []

        if epochs_data is not None:
            data = ica_instance.get_sources(epochs_data).get_data()
            mean_data = data.mean(axis=0)
            var_data = data.std(axis=0)
            times = epochs_data.times

            for idx in range(len(mean_data)):
                component = []
                for t_idx, t in enumerate(times):
                    component.append({
                        'time': float(t),
                        'value': float(mean_data[idx][t_idx]),
                        'variance': float(var_data[idx][t_idx])
                    })
                component_data.append(component)

        elif raw_data is not None:
            data = ica_instance.get_sources(raw_data).get_data()
            times = np.arange(len(data[0])) / raw_data.info['sfreq']

            for idx in range(len(data)):
                component = []
                for t_idx, t in enumerate(times):
                    component.append({
                        'time': float(t),
                        'value': float(data[idx][t_idx])
                    })
                component_data.append(component)

        return jsonify({
            'status': 'success',
            'data': component_data
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@api_bp.route('/stop', methods=['POST'])
def stop_processing():
    """Stop processing endpoint"""
    try:
        setattr(builtins, 'STOP_PROCESSING', True)
        current_progress['status'] = 'cancelled'
        current_progress['logs'].append("Processing cancelled by user")
        socketio.emit('status_update', current_progress)
        return jsonify({'message': 'Processing stopped'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/status', methods=['GET'])
def get_status():
    """Get current processing status"""
    return jsonify(current_progress), 200


@api_bp.route('/test', methods=['GET'])
def test():
    try:
        return jsonify({
            'status': 'success',
            'message': 'Server is running',
            'version': '1.0.0',
            'upload_folder': str(UPLOAD_FOLDER),
            'tmseeg_dir': str(TMSEEG_DATA_DIR)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@api_bp.before_request
def log_request():
    print(f"[DEBUG] Request: {request.method} {request.path}")

# Add better error handling
@api_bp.errorhandler(404)
def not_found_error(error):
    print(f"[DEBUG] 404 Error: {request.path}")
    return jsonify({'error': f'Endpoint not found: {request.path}'}), 404

@api_bp.errorhandler(500)
def internal_server_error(error):
    print(f"[DEBUG] 500 Error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500



@api_bp.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


def initialize_state():
    """Initialize global state"""
    return {
        'status': 'idle',
        'progress': 0,
        'step': '',
        'logs': [],
        'results': None,
        'error': None
    }

class OutputCapturer:
    def __init__(self, socketio):
        self.socketio = socketio
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.buffer = []
        print("OutputCapturer initialized with socketio:", socketio)  # Debug log

    def write(self, text):
        if text.strip():  # Only process non-empty lines
            # Write to original stdout for terminal logging
            self.original_stdout.write(text)
            self.original_stdout.flush()

            # Add debug output
            self.original_stdout.write(f"DEBUG: Attempting to emit: {text.strip()}\n")

            # Emit the output to the client
            try:
                self.socketio.emit('processing_output', {
                    'output': text.strip(),
                    'timestamp': time.strftime('%H:%M:%S')
                }, namespace='/')  # Add explicit namespace
                self.original_stdout.write("DEBUG: Successfully emitted\n")
            except Exception as e:
                self.original_stderr.write(f"Error emitting output: {str(e)}\n")
                self.original_stderr.write(traceback.format_exc())

        def flush(self):
            self.original_stdout.flush()

        def isatty(self):
            return self.original_stdout.isatty()

def process_task(params):
    """Process the TMS-EEG data with logging and error handling"""
    global current_progress, output_capturer
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    output_capture = OutputCapturer(socketio)

    def status_callback(msg, progress=None):
        """Send status updates to the client"""
        try:
            # Update progress state
            if progress is not None:
                current_progress['progress'] = progress

            # Add message to logs and emit it directly
            msg_text = str(msg).strip()
            if msg_text:
                current_progress['logs'].append(msg_text)
                socketio.emit('processing_output', {
                    'output': msg_text,
                    'timestamp': time.strftime('%H:%M:%S')
                }, namespace='/')

            # Emit status update separately
            socketio.emit('status_update', {
                'status': current_progress['status'],
                'progress': current_progress['progress'],
                'logs': current_progress['logs'],
                'error': current_progress.get('error')
            }, namespace='/')

        except Exception as e:
            print(f"Error in status callback: {str(e)}")

    try:
        # Redirect stdout and stderr to our capture
        sys.stdout = output_capture
        sys.stderr = output_capture

        server_logger.info("Starting processing task...")
        current_progress.update({ # Access the global instance of current_progress
            'status': 'processing',
            'logs': [],
            'error': None,
            'progress': 0
        })
        socketio.emit('status_update', current_progress)

        # Map frontend parameters to backend parameters
        server_logger.info("Mapping frontend parameters...")
        backend_params = map_frontend_to_backend_params(params)
        backend_params['gui_mode'] = True

        # Get and fix the data directory path
        data_dir = Path(params.get('dataDir', ''))
        if 'TMSEEG/TMSEEG' in str(data_dir):
            data_dir = Path(str(data_dir).replace('TMSEEG/TMSEEG', 'TMSEEG'))
            backend_params['dataDir'] = str(data_dir)

        server_logger.info(f"Checking directory structure in: {data_dir}")

        # Verify the directory exists
        if not data_dir.exists():
            error_msg = f"Directory does not exist: {data_dir}"
            server_logger.error(error_msg)
            raise ValueError(error_msg)

        # Add detailed directory information
        server_logger.info(f"Directory exists: {data_dir}")
        server_logger.info(f"Directory is absolute path: {data_dir.is_absolute()}")

        # List directory contents
        server_logger.info("Scanning directory contents...")
        try:
            for item in data_dir.iterdir():
                server_logger.info(
                    f"Found item: {item} (is_file: {item.is_file()}, is_dir: {item.is_dir()})")
        except Exception as e:
            server_logger.error(f"Error listing directory contents: {str(e)}")

        # Search for .ses files
        server_logger.info("Searching for .ses files...")
        ses_files = []
        for root, dirs, files in os.walk(data_dir):
            server_logger.debug(f"Scanning directory: {root}")
            server_logger.debug(f"Found directories: {dirs}")
            server_logger.debug(f"Found files: {files}")

            for file in files:
                if file.endswith('.ses'):
                    full_path = Path(root) / file
                    ses_files.append(full_path)
                    server_logger.info(f"Found .ses file: {full_path}")

        if not ses_files:
            error_msg = (f"No .ses files found in directory tree: {data_dir}\n"
                         f"Please check if the files exist and have the correct permissions.")
            server_logger.error(error_msg)
            current_progress.update({
                'status': 'error',
                'error': error_msg,
                'logs': current_progress['logs'] + [error_msg]
            })
            socketio.emit('status_update', current_progress)
            raise ValueError(error_msg)

        # Verify TMSEEG directory structure
        tmseeg_dir = data_dir
        if not any(f.endswith('.ses') for f in os.listdir(tmseeg_dir)):
            error_msg = f"No .ses files found in TMSEEG directory: {tmseeg_dir}"
            server_logger.error(error_msg)
            raise ValueError(error_msg)

        # Update backend parameters with verified TMSEEG directory
        backend_params['dataDir'] = str(tmseeg_dir)

        # Create argparse.Namespace object
        args = create_args_from_params(backend_params)

        # Set up ICA callback if manual mode is enabled
        if args.first_ica_manual or args.second_ica_manual:
            ica_callback = get_ica_callback(gui_mode=args.gui_mode)
            args.ica_callback = ica_callback

        # Process based on mode

        server_logger.info("Starting epoched processing mode...")
        current_progress['logs'].append("Running epoched processing mode...")
        socketio.emit('status_update', current_progress)

        results = process_subjects(args, status_callback=status_callback)
        current_progress['results'] = {
            'pcist_values': results if isinstance(results, list) else [],
            'processing_mode': 'epoched'
        }

            #server_logger.info("Starting continuous processing mode...")
            #current_progress['logs'].append("Running continuous processing mode...")
           # socketio.emit('status_update', current_progress)

           # processed_raws, pcist_values, pcist_objects, pcist_details, session_names = process_continuous_data(args)
           # current_progress['results'] = {
            #    'pcist_values': pcist_values,
           #     'processing_mode': 'continuous',
           #     'session_names': session_names
           # }

        # Process completion
        server_logger.info("Processing completed successfully")
        current_progress.update({
            'status': 'complete',
            'progress': 100,
            'logs': current_progress['logs'] + ["Processing completed successfully"]
        })
        socketio.emit('status_update', current_progress)

    except Exception as e:  # Improved error handling

        error_details = traceback.format_exc()
        server_logger.error(f"Error in processing task: {str(e)}\n{error_details}")
        current_progress.update({  # Access the global instance of current_progress

            'status': 'error',
            'error': str(e),
            'logs': current_progress['logs'] + [
                f"Error occurred: {str(e)}",
                "Full traceback:",
                error_details
            ]
        })

        socketio.emit('status_update', current_progress)

        raise  # Reraise the exception for proper handling


    finally:

        sys.stdout = original_stdout
        sys.stderr = original_stderr
        server_logger.flush_queue()


def run_server(port=5001, debug=False):
    """Run the Flask server with SocketIO"""
    import multiprocessing
    import signal
    global app, socketio, server_logger, output_capturer, current_progress, UPLOAD_FOLDER, TMSEEG_DATA_DIR, api_bp
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt, QCoreApplication
    import sys

    # Initialize Qt in main thread
    if QApplication.instance() is None:
        qt_app = QApplication(sys.argv)
        # Configure Qt for macOS

        if sys.platform == "darwin":
            qt_app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar)
            import os
            os.environ["QT_MAC_WANTS_LAYER"] = "1"
            os.environ["QT_QPA_PLATFORM"] = "cocoa"

    try:
        # Ensure multiprocessing start method is set before initializing Flask and SocketIO
        multiprocessing.set_start_method('spawn', force=True)

        # Get user data directory
        user_data_dir = Path(appdirs.user_data_dir("tmseegpy", "LazyCyborg"))

        # Initialize Flask app and related components
        app, socketio, server_logger, output_capturer, UPLOAD_FOLDER, TMSEEG_DATA_DIR = init_app(user_data_dir)

        @socketio.on('connect')
        def handle_connect():
            print("Client connected")
            socketio.emit('processing_output', {
                'output': 'Socket connection established',
                'timestamp': time.strftime('%H:%M:%S')
            })

        @socketio.on('disconnect')
        def handle_disconnect():
            # Clear any pending ICA selections
            while not ica_selection_queue.empty():
                ica_selection_queue.get()
            while not ica_result_queue.empty():
                ica_result_queue.get()

        @socketio.on('request_ica_data')
        def handle_ica_data_request():
            from queue import Empty
            """Handle ICA data requests with better synchronization"""
            try:
                print("Received ICA data request, queue size:", ica_selection_queue.qsize())

                # Increased timeout to handle potential delays
                ica_payload = ica_selection_queue.get(timeout=10)

                print(f"Sending ICA data with {len(ica_payload['data'])} components")
                emit('ica_data', ica_payload)

                # Immediately put back for multiple requests
                ica_selection_queue.put(ica_payload)

            except Empty:
                msg = "No ICA data available (queue empty)"
                print(msg)
                emit('ica_error', {'message': msg})
            except Exception as e:
                print(f"Error handling ICA request: {str(e)}")
                emit('ica_error', {'message': str(e)})

        @socketio.on('ica_selection_complete')
        def handle_ica_selection_complete(data):
            """Handle completed ICA component selection"""
            try:
                print("Received ICA selection complete")
                selected_components = data.get('selectedComponents', [])
                print(f"Selected components: {selected_components}")

                # Put the components in the result queue and clear any existing items
                while not ica_result_queue.empty():
                    try:
                        ica_result_queue.get_nowait()
                    except Empty:
                        break

                ica_result_queue.put(selected_components)

                # Explicitly emit the success event
                socketio.emit('ica_selection_success', {'status': 'success'})
                print("Emitted ica_selection_success")

                return {'status': 'success', 'selected': selected_components}

            except Exception as e:
                error_msg = f"Error in ICA selection complete: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                socketio.emit('ica_error', {'message': error_msg})
                return {'status': 'error', 'message': error_msg}

        @socketio.on('ica_selection_cancel')
        def handle_ica_selection_cancel():
            """Handle cancelled ICA selection"""
            try:
                ica_result_queue.put([])
                emit('ica_selection_cancelled')
            except Exception as e:
                emit('ica_error', {'message': str(e)})

        def signal_handler(sig, frame):
            print('Keyboard interrupt received. Shutting down server...')
            socketio.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        def cleanup_old_uploads():
            """Clean up old upload directories older than 24 hours."""
            try:
                upload_path = Path(UPLOAD_FOLDER)
                current_time = time.time()

                for item in upload_path.iterdir():
                    if item.is_dir():
                        try:
                            dir_creation_time = item.stat().st_mtime

                            if current_time - dir_creation_time > 24 * 60 * 60:  # 24 hours
                                shutil.rmtree(item)
                                server_logger.info(f"Removed old upload directory: {item}")
                        except (OSError, ValueError) as e:
                            server_logger.error(f"Error cleaning up {item}: {str(e)}")
                            continue
            except Exception as e:
                server_logger.error(f"Error during cleanup: {str(e)}")

        # Add cleanup function to app context
        app.cleanup_uploads = cleanup_old_uploads

        cleanup_thread = threading.Thread(target=cleanup_scheduler, daemon=True)
        cleanup_thread.start()

        print(f"Starting TMSeegpy server on port {port}...")
        print(f"Upload directory: {UPLOAD_FOLDER}")
        app.cleanup_uploads()

        if os.environ.get('FLASK_ENV') == 'production':
            # Use geventwebsocket handler for production
            from gevent import pywsgi
            from geventwebsocket.handler import WebSocketHandler

            server = pywsgi.WSGIServer(
                ('0.0.0.0', port),
                app,
                handler_class=WebSocketHandler
            )
            server.serve_forever()
        else:
            # Development mode
            socketio.run(
                app,
                host='localhost',
                port=port,
                debug=debug,
                use_reloader=False,
                allow_unsafe_werkzeug=True  # Changed to True for development
            )

    except Exception as e:
        print(f"Failed to start server: {e}")
        traceback.print_exc()
        raise


# Register shutdown handler
import atexit

@atexit.register
def shutdown_server():
    if socketio is not None:
        server_logger.info("Shutting down server...")
        socketio.stop()
    sys.exit(0)





if __name__ == '__main__':
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_scheduler, daemon=True) # Fixed call for cleanup_scheduler
    cleanup_thread.start()
    run_server(debug=True)

