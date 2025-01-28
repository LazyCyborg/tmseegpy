const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electron', {
    // Python paths
    getPythonPaths: () => ipcRenderer.invoke('get-python-paths'),

    // Directory operations
    selectDirectory: (type) => ipcRenderer.invoke('select-directory', type),
    validateOutputDirectory: (path) => ipcRenderer.invoke('validate-output-directory', path),
    createOutputDirectory: (path) => ipcRenderer.invoke('create-output-directory', path),
    getDirectoryContents: (path) => ipcRenderer.invoke('get-directory-contents', path),

    // File operations
    checkFileExists: (path) => ipcRenderer.invoke('check-file-exists', path),
    getFileStats: (path) => ipcRenderer.invoke('get-file-stats', path),

    // Path operations
    joinPaths: (paths) => ipcRenderer.invoke('join-paths', paths),
    pathBasename: (path, ext) => ipcRenderer.invoke('path-basename', { path, ext }),
    pathDirname: (path) => ipcRenderer.invoke('path-dirname', path),
    joinPath: (...args) => ipcRenderer.invoke('join-path', ...args),
    pathJoin: (pathParts) => ipcRenderer.invoke('path-join', pathParts),

    // Update operations
    checkForUpdates: () => ipcRenderer.invoke('check-for-updates'),
    getAppVersion: () => ipcRenderer.invoke('get-app-version'),
    downloadUpdate: () => ipcRenderer.invoke('download-update'),
    installUpdate: () => ipcRenderer.invoke('install-update'),

    // Update event listeners
    onUpdateStatus: (callback) => {
        const wrappedCallback = (_, ...args) => callback(...args);
        ipcRenderer.on('update-status', wrappedCallback);
        return () => ipcRenderer.removeListener('update-status', wrappedCallback);
    },

    onUpdateError: (callback) => {
        const wrappedCallback = (_, ...args) => callback(...args);
        ipcRenderer.on('update-error', wrappedCallback);
        return () => ipcRenderer.removeListener('update-error', wrappedCallback);
    },

    onUpdateProgress: (callback) => {
        const wrappedCallback = (_, ...args) => callback(...args);
        ipcRenderer.on('download-progress', wrappedCallback);
        return () => ipcRenderer.removeListener('download-progress', wrappedCallback);
    },

    // Keep the original onDownloadProgress for backward compatibility if needed
    onDownloadProgress: (callback) => {
        const wrappedCallback = (_, ...args) => callback(...args);
        ipcRenderer.on('download-progress', wrappedCallback);
        return () => ipcRenderer.removeListener('download-progress', wrappedCallback);
    },

    // Cleanup helper
    removeAllListeners: () => {
        ipcRenderer.removeAllListeners('update-status');
        ipcRenderer.removeAllListeners('update-error');
        ipcRenderer.removeAllListeners('download-progress');
    }
});