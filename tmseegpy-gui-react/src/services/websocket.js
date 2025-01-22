// websocket.js
import { io } from 'socket.io-client';

const SOCKET_URL = 'http://localhost:5001';

export const socket = io(SOCKET_URL, {
    autoConnect: true,
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    timeout: 20000,
    transports: ['websocket', 'polling']
});

// Connection event handlers
socket.on('connect', () => {
    console.log('WebSocket connected');
});

socket.on('connect_error', (error) => {
    console.error('WebSocket connection error:', error);
});

socket.on('disconnect', (reason) => {
    console.log('WebSocket disconnected:', reason);
});

socket.on('reconnect', (attemptNumber) => {
    console.log('WebSocket reconnected after', attemptNumber, 'attempts');
});

socket.on('reconnect_error', (error) => {
    console.error('WebSocket reconnection error:', error);
});

socket.on('reconnect_failed', () => {
    console.error('WebSocket reconnection failed after max attempts');
});

socket.on('error', (error) => {
    console.error('WebSocket error:', error);
});

// Helper function to safely update console element
const updateConsoleOutput = (text, isError = false) => {
    try {
        const consoleElement = document.getElementById('console-output');
        if (consoleElement) {
            const formattedText = isError ? `ERROR: ${text}\n` : `${text}\n`;
            consoleElement.textContent += formattedText;
            consoleElement.scrollTop = consoleElement.scrollHeight;
        } else {
            console.warn('Console output element not found');
        }
    } catch (error) {
        console.error('Error updating console output:', error);
    }
};

// Custom event handlers
socket.on('status_update', (data) => {
    console.log('Received status update:', data);
    updateConsoleOutput(`Status: ${data.status}`);
});

socket.on('processing_output', (data) => {
    if (data && data.output) {
        updateConsoleOutput(data.output);
    }
});

socket.on('processing_error', (data) => {
    console.error('Processing error:', data);
    updateConsoleOutput(data.error || 'An unknown error occurred', true);
});

socket.on('processing_complete', (data) => {
    console.log('Processing complete:', data);
    updateConsoleOutput('Processing complete!');
});

// Export connection management functions
export const connectWebSocket = () => {
    if (!socket.connected) {
        socket.connect();
    }
};

export const disconnectWebSocket = () => {
    if (socket.connected) {
        socket.disconnect();
    }
};

export const isConnected = () => socket.connected;

// Export the socket instance
export default socket;