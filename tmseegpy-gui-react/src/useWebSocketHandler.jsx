import { useEffect } from 'react';

import { addLogFactory } from './services/logger';

const useWebSocketHandler = ({
  socket,
  serverReady,
  consoleRef,
  setIsConnected,
  setProgress,
  setProcessingStatus,
  setProcessingLogs,
  setShowICAModal,
  setIcaDataReceived,
  setLoading,
  setIcaComponentData,
  setIcaComponentScores,
  setSelectedICAComponents,
  connectWebSocket,
  disconnectWebSocket,
  setTimeoutId,
  timeoutId,
  addLog
}) => {
  useEffect(() => {
let retryCount = 0;
    const maxRetries = 3;
    let reconnectAttempts = 0;
    const maxReconnects = 3;
    let observer;

    if (!serverReady || !consoleRef.current) {
      return;
    }

    // Initialize WebSocket connection
    const initializeSocket = () => {
      if (reconnectAttempts >= maxReconnects) {
        addLog('Max reconnection attempts reached', 'error');
        return;
      }

      connectWebSocket();
      setIsConnected(socket.connected);
      reconnectAttempts++;
    };

    initializeSocket();

    // Connection event handlers
    const handleConnect = () => {
      addLog('Connected to server', 'status');
      setIsConnected(true);
      reconnectAttempts = 0; // Reset on successful connection
    };

    const handleDisconnect = (reason) => {
      addLog(`Disconnected: ${reason}`, 'status');
      setIsConnected(false);

      // Only attempt reconnect if not a client-initiated disconnect
      if (reason !== 'io client disconnect' && reconnectAttempts < maxReconnects) {
        setTimeout(initializeSocket, 2000); // Wait 2s before reconnecting
      }
    };

    const handleICARequired = () => {
        setShowICAModal(true);
        setIcaDataReceived(false);
        setLoading(true);

        if (timeoutId) clearTimeout(timeoutId);

        const requestICAData = () => {
            if (!socket.connected) {
                addLog('Socket disconnected - attempting reconnect', 'error');
                initializeSocket();
                return;
            }

            console.log('Requesting ICA data, attempt:', retryCount + 1);
            socket.emit('request_ica_data');

            const newTimeoutId = setTimeout(() => {
                if (retryCount < maxRetries) {
                    retryCount++;
                    requestICAData();
                } else {
                    addLog('Server response timeout after retries', 'error');
                    setShowICAModal(false);
                    setLoading(false);
                    retryCount = 0;
                }
            }, 10000);

            setTimeoutId(newTimeoutId);
        };

        requestICAData();
    };

    // Status update handler
    const handleStatusUpdate = (data) => {
      // Use requestAnimationFrame for smoother progress updates
      requestAnimationFrame(() => {
        setProgress(data.progress || 0);
        setProcessingStatus(data.status || '');
      });

      // Batch log updates
      if (data.status) {
        addLog(`Status: ${data.status}`, 'status');
      }

      if (data.error) {
        addLog(`Error: ${data.error}`, 'error');
      }

      // Process logs in batches
      if (data.logs && Array.isArray(data.logs)) {
        const batchedLogs = data.logs.map(serverLog => ({
          message: typeof serverLog === 'string' ? serverLog : JSON.stringify(serverLog),
          type: 'status',
          timestamp: new Date()
        }));

        setProcessingLogs(prev => [...prev, ...batchedLogs]);
      }
    };


    // Processing event handlers
    const handleProcessingOutput = (data) => {
      const message = typeof data === 'string' ? data : data.output || JSON.stringify(data);
      addLog(message, 'info');
    };

    const handleProcessingError = (data) => {
      addLog(data?.error || 'An unknown error occurred', 'error');
    };

    const handleProcessingComplete = () => {
      addLog('Processing complete!', 'status');
    };

    // ICA event handlers
    const handleICAStatus = (data) => {
      addLog(data?.message || JSON.stringify(data), 'ica');
    };




        const handleICAData = (payload) => {
            if (timeoutId) {
                clearTimeout(timeoutId);
                setTimeoutId(null);
            }
            retryCount = 0;

            console.log('Received ICA data:', payload);

            if (!payload?.data?.length) {
                addLog('No ICA component data received', 'error');
                setShowICAModal(false);
                return;
            }

            setIcaComponentData(payload.data);
            setIcaComponentScores(payload.scores || {});
            setIcaDataReceived(true);
            setLoading(false);
            //addLog('ICA component data received', 'ica');
        };

    const handleICASelectionSuccess = () => {
      setShowICAModal(false);
      setIcaComponentData([]);
      setSelectedICAComponents([]);
      addLog('ICA component selection completed', 'ica');
    };

    const handleICASelectionCancelled = () => {
      setShowICAModal(false);
      setIcaComponentData([]);
      setSelectedICAComponents([]);
      addLog('ICA component selection cancelled', 'ica');
    };

    const handleICAError = (error) => {
      clearTimeout(timeoutId);  // Clear the timeout
      addLog(error.message || 'Failed to load ICA components', 'error');
      setShowICAModal(false);
      setLoading(false);
    };

    // Register all event listeners
    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);
    socket.on('status_update', handleStatusUpdate);
    socket.on('processing_output', handleProcessingOutput);
    socket.on('processing_error', handleProcessingError);
    socket.on('processing_complete', handleProcessingComplete);
    socket.on('ica_status', handleICAStatus);
    socket.on('ica_required', handleICARequired);
    socket.on('ica_data', handleICAData);
    socket.on('ica_selection_success', handleICASelectionSuccess);
    socket.on('ica_selection_cancelled', handleICASelectionCancelled);
    socket.on('ica_error', handleICAError);

    // Set up auto-scroll observer
    observer = new MutationObserver(() => {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    });
    observer.observe(consoleRef.current, { childList: true });

    // Cleanup function
    return () => {
      // Remove all event listeners
      socket.off('connect', handleConnect);
      socket.off('disconnect', handleDisconnect);
      socket.off('status_update', handleStatusUpdate);
      socket.off('processing_output', handleProcessingOutput);
      socket.off('processing_error', handleProcessingError);
      socket.off('processing_complete', handleProcessingComplete);
      socket.off('ica_status', handleICAStatus);
      socket.off('ica_required', handleICARequired);
      socket.off('ica_data', handleICAData);
      socket.off('ica_selection_success', handleICASelectionSuccess);
      socket.off('ica_selection_cancelled', handleICASelectionCancelled);
      socket.off('ica_error', handleICAError);

      // Clean up observer and disconnect
      if (observer) {
        observer.disconnect();
      }
        if (timeoutId) {
    clearTimeout(timeoutId);
  }
      disconnectWebSocket();
    };
  }, [serverReady, consoleRef.current, timeoutId]);
};

export default useWebSocketHandler;