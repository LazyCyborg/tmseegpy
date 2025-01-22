const { contextBridge, ipcRenderer } = require('electron');

// Handle update events
ipcRenderer.on('update-status', (_, status) => {
    window.dispatchEvent(new CustomEvent('update-status', { detail: status }));
});

ipcRenderer.on('download-progress', (_, progressObj) => {
    window.dispatchEvent(new CustomEvent('download-progress', { detail: progressObj }));
});

ipcRenderer.on('update-error', (_, error) => {
    window.dispatchEvent(new CustomEvent('update-error', { detail: error }));
});

contextBridge.exposeInMainWorld('electron', {
    // Directory operations
    selectDirectory: (type = 'data') => ipcRenderer.invoke('select-directory', type),
    selectOutputDirectory: () => ipcRenderer.invoke('select-output-directory'),
    getDirectoryContents: (dirPath) => ipcRenderer.invoke('get-directory-contents', dirPath),
    checkFileExists: (filePath) => ipcRenderer.invoke('check-file-exists', filePath),
    getFileStats: (filePath) => ipcRenderer.invoke('get-file-stats', filePath),
    createDirectory: (dirPath) => ipcRenderer.invoke('create-directory', dirPath),

    // Path operations
    path: {
        join: (...args) => ipcRenderer.invoke('join-paths', args),
        basename: (path, ext) => ipcRenderer.invoke('path-basename', { path, ext }),
        dirname: (path) => ipcRenderer.invoke('path-dirname', path)
    },

    // Directory validation
    validateOutputDirectory: (dirPath) => ipcRenderer.invoke('validate-output-directory', dirPath),
    ensureOutputDirectory: (dirPath) => ipcRenderer.invoke('ensure-output-directory', dirPath),

    // Auto-update functionality
    getAppVersion: () => ipcRenderer.invoke('get-app-version'),
    checkForUpdates: () => ipcRenderer.invoke('check-for-updates'),
    downloadUpdate: () => ipcRenderer.invoke('download-update'),
    installUpdate: () => ipcRenderer.invoke('install-update'),
    onUpdateStatus: (callback) => {
        const handler = (event) => callback(event.detail);
        window.addEventListener('update-status', handler);
        return () => window.removeEventListener('update-status', handler);
    },
    onUpdateProgress: (callback) => {
        const handler = (event) => callback(event.detail);
        window.addEventListener('download-progress', handler);
        return () => window.removeEventListener('download-progress', handler);
    },
    onUpdateError: (callback) => {
        const handler = (event) => callback(event.detail);
        window.addEventListener('update-error', handler);
        return () => window.removeEventListener('update-error', handler);
    },

    // Python integration - Updated for new structure
    getPythonPaths: () => ipcRenderer.invoke('get-python-paths'),
    runPythonScript: (script, args) => ipcRenderer.invoke('run-python-script', script, args),
    getTmseegpyPath: () => ipcRenderer.invoke('get-tmseegpy-path'),
    getBackendPath: () => ipcRenderer.invoke('get-backend-path'),

    // Version information
    getVersion: () => ipcRenderer.invoke('get-version'),
    getTmseegpyVersion: () => ipcRenderer.invoke('get-tmseegpy-version'),
    getPythonVersion: () => ipcRenderer.invoke('get-python-version'),

    // Python process events
    onPythonOutput: (callback) => {
        const handler = (event) => callback(event.detail);
        window.addEventListener('python-output', handler);
        return () => window.removeEventListener('python-output', handler);
    },
    onPythonError: (callback) => {
        const handler = (event) => callback(event.detail);
        window.addEventListener('python-error', handler);
        return () => window.removeEventListener('python-error', handler);
    },
    onPythonExit: (callback) => {
        const handler = (event) => callback(event.detail);
        window.addEventListener('python-exit', handler);
        return () => window.removeEventListener('python-exit', handler);
    }
});

// Error handling for IPC
ipcRenderer.on('error', (_, error) => {
    console.error('IPC Error:', error);
    window.dispatchEvent(new CustomEvent('ipc-error', { detail: error }));
});

// Handle Python process output
ipcRenderer.on('python-output', (_, data) => {
    window.dispatchEvent(new CustomEvent('python-output', { detail: data }));
});

ipcRenderer.on('python-error', (_, error) => {
    window.dispatchEvent(new CustomEvent('python-error', { detail: error }));
});

ipcRenderer.on('python-exit', (_, code) => {
    window.dispatchEvent(new CustomEvent('python-exit', { detail: code }));
});