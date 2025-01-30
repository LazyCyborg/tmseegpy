export function addLogFactory(setProcessingLogs) {
  let batchedLogs = [];
  let timeoutId = null;

  return function addLog(message, type = 'info') {
    batchedLogs.push({
      message,
      type,
      timestamp: new Date()
    });

    // Clear existing timeout
    if (timeoutId) {
      clearTimeout(timeoutId);
    }

    // Update logs after a short delay to batch multiple updates
    timeoutId = setTimeout(() => {
      setProcessingLogs(prev => [...prev, ...batchedLogs]);
      batchedLogs = [];
    }, 16); // Approximately one frame at 60fps
  };
}
