# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import QApplication
import sys
from filelock import FileLock
from pathlib import Path
import argparse
import threading
import queue
import json
import os
import tempfile
import shutil
from werkzeug.utils import secure_filename
import time
import numpy as np
from tmseegpy.run import process_subjects, process_continuous_data, setup_qt_plugin_path
from tmseegpy.preproc import TMSEEGPreprocessor
from tmseegpy.dataloader import TMSEEGLoader
import builtins
import mne
import sys
from tmseegpy.cli_ica_selector import CLIICASelector, get_cli_ica_callback
import traceback
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # For headless servers

# Define constants at the top level
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = BASE_DIR / 'uploads'  # Main upload directory
TMSEEG_DATA_DIR = UPLOAD_FOLDER / 'TMSEEG' # Create TMSEEG subdirectory for loader
ALLOWED_EXTENSIONS = {'ses', 'set', 'eeg', 'vhdr', 'edf', 'cnt', 'fif', 'vmrk', 'fdt'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024 * 1024  # 5GB

#setup_qt_plugin_path()


# Global window manager instance
window_manager = None
# Create the Flask application
# Create the Flask application
# Create the Flask application
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Basic configuration
app.config.update(
    UPLOAD_FOLDER=str(UPLOAD_FOLDER),
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    SECRET_KEY=os.environ.get('SECRET_KEY', 'your-secret-key-here'),
    ALLOWED_EXTENSIONS=ALLOWED_EXTENSIONS
)

# Initialize Socket.IO
socketio = SocketIO(app,
                   cors_allowed_origins="*",
                   async_mode='threading',  # Change to threading mode
                   logger=True,
                   engineio_logger=True)
# Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
TMSEEG_DATA_DIR.mkdir(exist_ok=True)

# Global variables to track processing state
current_progress = {
    'status': 'idle',
    'progress': 0,
    'step': '',
    'logs': [],
    'results': None,
    'error': None
}

# Global variable to store the ICA callback
ica_callback = None

def map_frontend_to_backend_params(frontend_params):
    """
    Maps parameters from the frontend naming convention to backend parameters.
    Ensures alignment with run.py's argparse parameters.
    """
    required_params = ['processingMode', 'dataFormat']
    for param in required_params:
        if param not in frontend_params:
            raise ValueError(f"Missing required parameter: {param}")
    data_dir = Path(frontend_params.get('dataDir', ''))
    if 'TMSEEG/TMSEEG' in str(data_dir):
        data_dir = str(data_dir).replace('TMSEEG/TMSEEG', 'TMSEEG')

    backend_params = {
        # Core Processing Parameters
        'processing_mode': frontend_params.get('processingMode', 'epoched'),
        'data_dir': str(data_dir),
        'output_dir': frontend_params.get('outputDir', str(Path.cwd() / 'output')),
        'data_format': frontend_params.get('dataFormat', 'neurone'),
        'no_preproc_output': frontend_params.get('noPreprocessOutput', False),
        'no_pcist': frontend_params.get('noPcist', False),
        'eeglab_montage_units': frontend_params.get('eeglabMontageUnits', 'auto'),
        'stim_channel': frontend_params.get('stimChannel', 'STI 014'),
        'plot_preproc': frontend_params.get('plotPreproc', False),
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
        'notch_freq': float(frontend_params.get('notchFreq', 50)),
        'notch_width': float(frontend_params.get('notchWidth', 2)),

        # Epoch Parameters
        'epochs_tmin': float(frontend_params.get('epochsTmin', -0.41)),
        'epochs_tmax': float(frontend_params.get('epochsTmax', 0.41)),

        # Artifact Detection Parameters
        'bad_channels_threshold': float(frontend_params.get('badChannelsThreshold', 2)),
        'bad_epochs_threshold': float(frontend_params.get('badEpochsThreshold', 2)),
        'amplitude_threshold': float(frontend_params.get('amplitudeThreshold', 300.0)),

        # ICA Parameters
        'ica_method': frontend_params.get('icaMethod', 'fastica'),
        'first_ica_manual': frontend_params.get('firstIcaManual', False),
        'second_ica_manual': frontend_params.get('secondIcaManual', False),
        'no_second_ICA': frontend_params.get('noSecondIca', False),
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
        'validate_teps': frontend_params.get('validateTeps', True),
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


from multiprocessing import Process, Pipe


def handle_ica_selection(ica_obj, inst, component_scores=None):
    """Handle ICA component selection using a pipe for communication"""
    from ica_process import run_ica_selection, process_target  # change to .ica_process when packaged,
    parent_conn, child_conn = Pipe()

    # Start process
    process = Process(target=process_target,
                      args=(child_conn, ica_obj, inst, component_scores))
    process.start()

    # Wait for result
    try:
        result = parent_conn.recv()
    except Exception as e:
        print(f"Error receiving ICA selection result: {str(e)}")
        result = []
    finally:
        parent_conn.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join()

    return result



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

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
# Route handlers

@app.route('/api/', methods=['GET'])
def api_root():
    return jsonify({
        'status': 'success',
        'message': 'TMSEEG API Server',
        'version': '1.0.0',
        'endpoints': [
            '/api/test',
            '/api/upload',
            # List other available endpoints here
        ]
    })

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'success',
        'message': 'Server is running',
        'upload_folder': str(UPLOAD_FOLDER),
        'tmseeg_dir': str(TMSEEG_DATA_DIR)
    })


@app.route('/api/upload', methods=['POST'])
def handle_upload():
    """Unified upload handler for both files and directory structures"""
    try:
        print("Received upload request")
        print("Form data:", request.form)
        print("Files:", request.files)

        # Get directory information
        parent_dir = request.form.get('parentDirectory')
        tmseeg_dir_path = request.form.get('tmseegDirectory')

        if not parent_dir or not tmseeg_dir_path:
            return jsonify({'error': 'Missing directory information'}), 400

        # Convert to Path objects
        parent_path = Path(parent_dir)
        tmseeg_path = Path(tmseeg_dir_path)

        # Create TMSEEG directory if it doesn't exist
        tmseeg_path.mkdir(parents=True, exist_ok=True)

        # Handle file if present
        if 'file' in request.files:
            file = request.files['file']
            if file.filename:
                if file.filename.startswith('NeurOne-') and file.filename.endswith('.ses'):
                    # Copy .ses file to TMSEEG directory
                    file_path = tmseeg_path / file.filename
                    file.save(str(file_path))

                    # Create session directory (remove 'NeurOne-' prefix and '.ses' extension)
                    session_name = file.filename[8:-4]  # Remove 'NeurOne-' and '.ses'
                    session_dir = tmseeg_path / session_name
                    session_dir.mkdir(exist_ok=True)

                    # Look for and copy related files
                    for source_file in parent_path.iterdir():
                        if (source_file.is_file() and
                                source_file.name.endswith(session_name) and
                                not source_file.name.endswith('.ses')):
                            dest_file = session_dir / source_file.name
                            shutil.copy2(source_file, dest_file)
                            print(f"Copied related file: {source_file.name}")

        # Verify the structure
        ses_files = list(tmseeg_path.glob('NeurOne-*.ses'))
        if not ses_files:
            return jsonify({'error': 'No NeurOne .ses files found'}), 400

        # Check if each .ses file has a corresponding directory
        sessions = []
        for ses_file in ses_files:
            session_name = ses_file.stem[8:]  # Remove 'NeurOne-' prefix
            session_dir = tmseeg_path / session_name
            if not session_dir.is_dir():
                return jsonify({'error': f'Missing directory for session {session_name}'}), 400
            sessions.append(session_name)

        # Return success response with directory information
        response_data = {
            'message': 'Upload successful',
            'tmseeg_dir': str(tmseeg_path),
            'parent_dir': str(parent_path),
            'sessions': sessions
        }

        print("Sending response:", response_data)
        return jsonify(response_data), 200

    except Exception as e:
        error_msg = f"Upload error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

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


@app.route('/api/process', methods=['POST'])
def start_processing():
    """Start processing endpoint"""
    try:
        params = request.get_json()
        if not params:
            return jsonify({'error': 'No parameters provided'}), 400

        # Fix the data directory path - remove any duplicate TMSEEG
        data_dir = Path(params.get('dataDir', ''))
        if 'TMSEEG/TMSEEG' in str(data_dir):
            # Remove the duplicate TMSEEG
            data_dir = Path(str(data_dir).replace('TMSEEG/TMSEEG', 'TMSEEG'))
            params['dataDir'] = str(data_dir)

        print(f"Processing data directory: {data_dir}")

        # Verify the directory exists
        if not data_dir.exists():
            return jsonify({'error': f'Data directory does not exist: {data_dir}'}), 400

        # Add ICA callback to params
        params['ica_callback'] = get_cli_ica_callback()

        # Start processing in a separate thread
        threading.Thread(target=process_task, args=(params,), daemon=True).start()

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

@app.route('/api/stop', methods=['POST'])
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


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current processing status"""
    return jsonify(current_progress), 200


# WebSocket event handlers
@socketio.on('request_status')
def handle_status_request():
    """Handle status request from client"""
    emit('status_update', current_progress)


@socketio.on('cancel_processing')
def handle_cancel_request():
    """Handle processing cancellation request"""
    setattr(builtins, 'STOP_PROCESSING', True)
    current_progress['status'] = 'cancelled'
    current_progress['logs'].append("Processing cancelled by user")
    emit('status_update', current_progress)


@app.route('/api/test_data_loading', methods=['POST'])
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


def process_task(params):
    """Modified process task to work with correct TMSEEG structure"""
    # Set up output capturing
    from io import StringIO
    import sys, time, threading
    output_capture = StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Update initial status
        current_progress.update({
            'status': 'processing',
            'logs': [],
            'error': None,
            'progress': 0
        })
        socketio.emit('status_update', current_progress)

        # Set up output capturing
        sys.stdout = output_capture
        sys.stderr = output_capture

        # Start periodic output emission
        def emit_output():
            while True:
                output = output_capture.getvalue()
                if output:
                    socketio.emit('processing_output', {'output': output})
                    output_capture.truncate(0)
                    output_capture.seek(0)
                time.sleep(0.5)  # Check every half second

        output_thread = threading.Thread(target=emit_output)
        output_thread.daemon = True
        output_thread.start()

        # Map frontend parameters to backend parameters
        backend_params = map_frontend_to_backend_params(params)

        # Get and fix the data directory path
        data_dir = Path(params.get('dataDir', ''))
        if 'TMSEEG/TMSEEG' in str(data_dir):
            data_dir = Path(str(data_dir).replace('TMSEEG/TMSEEG', 'TMSEEG'))
            backend_params['dataDir'] = str(data_dir)

        print(f"Checking directory structure in: {data_dir}")

        # Verify the directory exists
        if not data_dir.exists():
            error_msg = f"Directory does not exist: {data_dir}"
            raise ValueError(error_msg)

        # Add more detailed logging
        current_progress['logs'].append(f"Directory exists: {data_dir}")
        current_progress['logs'].append(f"Directory is absolute path: {data_dir.is_absolute()}")

        # List all files in directory for debugging
        current_progress['logs'].append("Directory contents:")
        try:
            for item in data_dir.iterdir():
                current_progress['logs'].append(
                    f"Found item: {item} (is_file: {item.is_file()}, is_dir: {item.is_dir()})")
        except Exception as e:
            current_progress['logs'].append(f"Error listing directory contents: {str(e)}")

        # Recursively search for .ses files with more detailed logging
        ses_files = []
        for root, dirs, files in os.walk(data_dir):
            current_progress['logs'].append(f"Scanning directory: {root}")
            current_progress['logs'].append(f"Found directories: {dirs}")
            current_progress['logs'].append(f"Found files: {files}")

            for file in files:
                current_progress['logs'].append(f"Checking file: {file}")
                if file.endswith('.ses'):
                    full_path = Path(root) / file
                    ses_files.append(full_path)
                    current_progress['logs'].append(f"Found .ses file: {full_path}")

        if not ses_files:
            error_msg = (f"No .ses files found in directory tree: {data_dir}\n"
                         f"Please check if the files exist and have the correct permissions.")
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
            raise ValueError(error_msg)

        # Update backend parameters with verified TMSEEG directory
        backend_params['dataDir'] = str(tmseeg_dir)

        # Create argparse.Namespace object
        args = create_args_from_params(backend_params)

        # Add ICA callback if manual mode is enabled
        if args.first_ica_manual or args.second_ica_manual:
            args.ica_callback = handle_ica_selection

        # Process based on mode
        if args.processing_mode == 'epoched':
            current_progress['logs'].append("Running epoched processing mode...")
            socketio.emit('status_update', current_progress)
            results = process_subjects(args)
            current_progress['results'] = {
                'pcist_values': results if isinstance(results, list) else [],
                'processing_mode': 'epoched'
            }
        else:
            current_progress['logs'].append("Running continuous processing mode...")
            socketio.emit('status_update', current_progress)
            processed_raws, pcist_values, pcist_objects, pcist_details, session_names = process_continuous_data(args)
            current_progress['results'] = {
                'pcist_values': pcist_values,
                'processing_mode': 'continuous',
                'session_names': session_names
            }

        # Get final captured output
        final_output = output_capture.getvalue()
        if final_output:
            socketio.emit('processing_output', {'output': final_output})

        # Update final status
        current_progress.update({
            'status': 'complete',
            'progress': 100,
            'logs': current_progress['logs'] + ["Processing completed successfully"]
        })
        socketio.emit('status_update', current_progress)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        # Get error output
        error_output = output_capture.getvalue()
        if error_output:
            socketio.emit('processing_output', {'output': error_output})

        current_progress.update({
            'status': 'error',
            'error': str(e),
            'logs': current_progress['logs'] + [
                f"Error occurred: {str(e)}",
                "Full traceback:",
                error_details
            ]
        })
        socketio.emit('status_update', current_progress)
        raise

    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        output_capture.close()


if __name__ == '__main__':
    import multiprocessing

    # Create required directories
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    TMSEEG_DATA_DIR.mkdir(exist_ok=True)

    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)

    # Start the Flask server
    socketio.run(app,
                 host='0.0.0.0',
                 port=5001,
                 debug=True,
                 use_reloader=False,
                 allow_unsafe_werkzeug=True)