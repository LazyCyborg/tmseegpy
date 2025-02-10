import logging
import sys
from flask_socketio import SocketIO
from typing import Optional
import queue
from threading import Lock

class ServerLogger:
    """
    Custom logger that handles both terminal output and websocket transmission
    """

    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.logger = logging.getLogger('TMSeegpy')
        self.logger.setLevel(logging.INFO)

        # Terminal handler
        terminal_handler = logging.StreamHandler(sys.stdout)
        terminal_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(terminal_handler)

        # Message queue for batching websocket transmissions
        self.message_queue = queue.Queue()
        self.queue_lock = Lock()

        # File handler
        file_handler = logging.FileHandler('tmseegpy.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def emit_log(self, message: str, level: str = 'info'):
        """Emit a log message both to terminal and through websocket"""
        # Log to terminal/file
        if level == 'error':
            self.logger.error(message)
        else:
            self.logger.info(message)

        # Queue message for websocket transmission
        with self.queue_lock:
            self.message_queue.put({
                'type': level,
                'message': message
            })

        # Emit through websocket
        self.socketio.emit('processing_output', {'output': message})

    def info(self, message: str):
        """Log an info message"""
        self.emit_log(message, 'info')

    def error(self, message: str):
        """Log an error message"""
        self.emit_log(message, 'error')

    def debug(self, message: str):
        """Log a debug message"""
        self.emit_log(message, 'debug')

    def warning(self, message: str):
        """Log a warning message"""
        self.emit_log(message, 'warning')

    def flush_queue(self):
        """Flush all queued messages through the websocket"""
        with self.queue_lock:
            while not self.message_queue.empty():
                msg = self.message_queue.get()
                self.socketio.emit('processing_output', {'output': msg['message']})