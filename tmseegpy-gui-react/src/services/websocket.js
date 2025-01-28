import { io } from 'socket.io-client';

// Configuration object
const CONFIG = {
    SOCKET_URL: process.env.REACT_APP_SOCKET_URL || 'http://localhost:5001',
    RECONNECTION_ATTEMPTS: 5,
    RECONNECTION_DELAY: 1000,
    RECONNECTION_DELAY_MAX: 5000,
    TIMEOUT: 20000
};

// Callback handlers
let callbacks = {
    status: () => {},
    output: () => {},
    error: () => {},
    ica: () => {},
    connection: () => {}
};

// Create socket instance
export const socket = io(CONFIG.SOCKET_URL, {
    autoConnect: true,
    reconnection: true,
    reconnectionAttempts: CONFIG.RECONNECTION_ATTEMPTS,
    reconnectionDelay: CONFIG.RECONNECTION_DELAY,
    reconnectionDelayMax: CONFIG.RECONNECTION_DELAY_MAX,
    timeout: CONFIG.TIMEOUT,
    transports: ['websocket', 'polling']
});

// Event logging wrapper
const logEvent = (eventName, data, isError = false) => {
    const timestamp = new Date().toISOString();
    const logMethod = isError ? console.error : console.log;
    logMethod(`[${timestamp}] ${eventName}:`, data);
};

// Enhanced console output handler
const updateConsoleOutput = (() => {
    const MAX_CONSOLE_LINES = 1000;
    let lineCount = 0;

    return (text, type = 'info') => {
        try {
            const timestamp = new Date().toLocaleTimeString();
            const formattedText = `[${timestamp}] ${text}`;

            // Call the appropriate callback based on type
            switch (type) {
                case 'error':
                    callbacks.error({ message: formattedText, type: 'error' });
                    break;
                case 'ica':
                    callbacks.ica({ message: formattedText, type: 'ica' });
                    break;
                case 'status':
                    callbacks.status({ message: formattedText, type: 'status' });
                    break;
                default:
                    callbacks.output({ message: formattedText, type: 'info' });
            }

            // Also log to console for debugging
            logEvent('Console Output', { text, type }, type === 'error');
        } catch (error) {
            logEvent('Console Output Error', error, true);
        }
    };
})();

// Connection event handlers
socket.on('connect', () => {
    logEvent('WebSocket', 'Connected');
    callbacks.connection(true);
    updateConsoleOutput('WebSocket Connected', 'status');
});

socket.on('connect_error', (error) => {
    logEvent('WebSocket Connection Error', error, true);
    callbacks.connection(false);
    updateConsoleOutput(`Connection Error: ${error.message}`, 'error');
});

socket.on('disconnect', (reason) => {
    logEvent('WebSocket Disconnected', reason);
    callbacks.connection(false);
    updateConsoleOutput(`Disconnected: ${reason}`, 'status');
});

socket.on('reconnect', (attemptNumber) => {
    logEvent('WebSocket Reconnected', `After ${attemptNumber} attempts`);
    callbacks.connection(true);
    updateConsoleOutput(`Reconnected after ${attemptNumber} attempts`, 'status');
});

socket.on('reconnect_error', (error) => {
    logEvent('WebSocket Reconnection Error', error, true);
    updateConsoleOutput(`Reconnection Error: ${error.message}`, 'error');
});

socket.on('reconnect_failed', () => {
    logEvent('WebSocket Reconnection Failed', 'Max attempts reached', true);
    updateConsoleOutput('Reconnection Failed: Max attempts reached', 'error');
});

// Custom event handlers
socket.on('status_update', (data) => {
    logEvent('Status Update', data);

    if (data.logs && Array.isArray(data.logs)) {
        data.logs.forEach(log => {
            updateConsoleOutput(log, 'status');
        });
    }

    if (data.status) {
        updateConsoleOutput(`Status: ${data.status}`, 'status');
    }

    if (data.error) {
        updateConsoleOutput(`Error: ${data.error}`, 'error');
    }
});

socket.on('processing_output', (data) => {
    console.log('Received processing output:', data); // Debug log
    if (typeof data === 'string') {
        updateConsoleOutput(data);
    } else if (data?.output) {
        updateConsoleOutput(data.output);
    }
});

socket.on('processing_error', (data) => {
    logEvent('Processing Error', data, true);
    updateConsoleOutput(data?.error || 'An unknown error occurred', 'error');
});

socket.on('processing_complete', (data) => {
    logEvent('Processing Complete', data);
    updateConsoleOutput('Processing complete!', 'status');
});

socket.on('ica_status', (data) => {
    logEvent('ICA Status', data);
    updateConsoleOutput(data?.message || JSON.stringify(data), 'ica');
});

// Register callbacks for different event types
export const registerCallbacks = ({
    onStatus,
    onOutput,
    onError,
    onICA,
    onConnection
}) => {
    if (onStatus) callbacks.status = onStatus;
    if (onOutput) callbacks.output = onOutput;
    if (onError) callbacks.error = onError;
    if (onICA) callbacks.ica = onICA;
    if (onConnection) callbacks.connection = onConnection;
};

// Connection management
let reconnectTimer = null;

export const connectWebSocket = () => {
    if (!socket.connected) {
        clearTimeout(reconnectTimer);
        socket.connect();

        reconnectTimer = setTimeout(() => {
            if (!socket.connected) {
                logEvent('WebSocket', 'Connection timeout - forcing reconnect');
                socket.disconnect().connect();
            }
        }, CONFIG.TIMEOUT);
    }
};

export const disconnectWebSocket = () => {
    clearTimeout(reconnectTimer);
    if (socket.connected) {
        socket.disconnect();
    }
};

export const isConnected = () => socket.connected;

// Health check functionality
export const checkConnection = () => {
    return new Promise((resolve) => {
        if (socket.connected) {
            resolve(true);
            return;
        }

        const timeoutId = setTimeout(() => {
            socket.off('connect', handleConnect);
            resolve(false);
        }, 5000);

        const handleConnect = () => {
            clearTimeout(timeoutId);
            socket.off('connect', handleConnect);
            resolve(true);
        };

        socket.once('connect', handleConnect);
        socket.connect();
    });
};

export default socket;