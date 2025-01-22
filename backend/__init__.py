from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import os
from pathlib import Path


def create_app():
    # Create the Flask application
    app = Flask(__name__)
    CORS(app)

    # Define base directory and constants
    BASE_DIR = Path(__file__).resolve().parent
    UPLOAD_FOLDER = BASE_DIR / 'temp_uploads'
    ALLOWED_EXTENSIONS = {'ses', 'set', 'eeg', 'vhdr', 'edf', 'cnt', 'fif', 'vmrk', 'fdt'}
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024 * 1024  # 5GB

    # Basic configuration
    app.config.update(
        UPLOAD_FOLDER=str(UPLOAD_FOLDER),
        MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
        SECRET_KEY=os.environ.get('SECRET_KEY', 'your-secret-key-here'),
        ALLOWED_EXTENSIONS=ALLOWED_EXTENSIONS,
        DEBUG=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )

    # Ensure upload folder exists
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    return app


# Create app instance
app = create_app()
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Import routes after app is created
from . import server

def run_server():
    try:
        debug = app.config['DEBUG']
        port = int(os.environ.get('PORT', 5001))
        print(f"Starting server on port {port}...")
        # Add async_mode='threading' to prevent eventlet/gevent issues
        socketio.run(app, host='0.0.0.0', port=port, debug=debug, async_mode='threading')
    except Exception as e:
        print(f"Failed to start server: {e}")
        import traceback
        print(traceback.format_exc())
        raise

if __name__ == '__main__':
    # Add this to keep the main thread running
    import threading
    main_thread = threading.current_thread()
    print(f"Main thread: {main_thread.name}")
    run_server()