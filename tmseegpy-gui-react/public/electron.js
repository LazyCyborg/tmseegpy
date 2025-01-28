const log = require('electron-log');
const path = require('path');
const fs = require('fs-extra');
const isDev = require('electron-is-dev');
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { autoUpdater } = require('electron-updater');
const { spawn } = require('child_process');
const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args));
const kill = require('tree-kill');

// Set app name for proper logging directory
const appName = isDev ? 'TMSeegpy GUI-dev' : 'TMSeegpy GUI';
app.name = appName;

let mainWindow;
const SERVER_PORT = 5001;

// Initialize logging and app setup (async to handle potential file system operations)
const initializeApp = async () => {
    try {
        console.log('Starting application initialization...');

        // Create logs directory
        const userDataPath = app.getPath('userData');
        const logsDir = path.join(userDataPath, 'logs');

        console.log('Creating logs directory at:', logsDir);
        await fs.ensureDir(logsDir);

        // Configure log file path
        log.transports.file.resolvePath = () => path.join(logsDir, 'main.log');

        // Configure auto-updater logging
        autoUpdater.logger = log;

        // Log startup information
        const logFile = log.transports.file.getFile();
        console.log('Log file path:', logFile.path);

        log.info('------- Application Starting -------');
        log.info('Log file location:', logFile.path);
        log.info('App version:', app.getVersion());
        log.info('Electron version:', process.versions.electron);
        log.info('Chrome version:', process.versions.chrome);
        log.info('Node version:', process.versions.node);
        log.info('Platform:', process.platform);
        log.info('Architecture:', process.arch);
        log.info('User data path:', userDataPath);
        log.info('App path:', app.getAppPath());
        log.info('Logs directory:', logsDir);
        log.info('Development mode:', isDev);

        // Verify logs directory
        const stats = await fs.stat(logsDir);
        log.info('Logs directory permissions:', stats.mode.toString(8));

        return true;
    } catch (error) {
        console.error('Error during initialization:', error);
        log.error('Error during initialization:', error);
        throw error;
    }
};

// Modify app.whenReady() to use the initialization
app.whenReady().then(async () => {
    try {
        await initializeApp();
        await createWindow();
        setupIpcHandlers();
        setupAutoUpdater();

        log.info('Application successfully initialized');
    } catch (error) {
        log.error('Failed to initialize application:', error);
        dialog.showErrorBox(
            'Startup Error',
            `Failed to initialize application: ${error.message}`
        );
        app.quit();
    }
});

// Check if TMSeegpy server is running
async function checkServerRunning() {
    try {
        const response = await fetch(`http://localhost:${SERVER_PORT}/api/test`);
        return response.ok;
    } catch (error) {
        return false;
    }
}



// Helper function to get the correct resource path
function getResourcePath(relativePath) {
    return isDev
        ? path.join(__dirname, '..', relativePath)
        : path.join(process.resourcesPath, relativePath);
}

// Get paths for tmseegpy and backend
const tmseegpyPath = getResourcePath('tmseegpy');
const backendPath = getResourcePath('server');


// Add this function to verify Python installation

// Modified createWindow function
async function createWindow() {
    try {
        const serverRunning = await checkServerRunning();

        if (!serverRunning) {
            // Log the error and show a dialog instead of throwing
            log.error('TMSeegpy server not running. Please start TMSeegpy first.');
            dialog.showErrorBox(
                'Server Error',
                'TMSeegpy server is not running. Please start the server and try again.'
            );
            app.quit();
            return;
        }

        mainWindow = new BrowserWindow({
            width: 1200, height: 800,
            webPreferences: {
                nodeIntegration: false, contextIsolation: true,
                enableRemoteModule: false,
                preload: path.join(__dirname, 'preload.js')
            }
        });

        if (isDev) {
            mainWindow.loadURL('http://localhost:3000');
            mainWindow.webContents.openDevTools();
        } else {
            mainWindow.loadFile(path.join(__dirname, 'index.html')); // Just load from the current directory
            autoUpdater.checkForUpdatesAndNotify();
        }

        // Add environment variables for Python paths
        process.env.TMSEEGPY_PATH = tmseegpyPath;
        process.env.BACKEND_PATH = backendPath;

        mainWindow.on('closed', async () => {
            mainWindow = null;
        });

    } catch (error) {
        log.error('Failed to create window:', error);
        dialog.showErrorBox(
            'Startup Error',
            `Failed to start the application: ${error.message}`
        );
        app.quit();
    }
}


// Helper functions


async function checkServerRunning() {
    try {
        const response = await fetch(`http://localhost:${SERVER_PORT}/api/test`);
        return response.ok;
    } catch (error) {
        return false;
    }
}

// Also add the waitForPort function that's referenced
async function waitForPort(port, timeout = 30000) {
    const start = Date.now();
    while (Date.now() - start < timeout) {
        try {
            const response = await fetch(`http://localhost:${port}/api/test`);
            if (response.ok) return true;
        } catch (error) {
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
    return false;
}
async function getDirectoryContents(dirPath) {
    try {
        const files = await fs.readdir(dirPath);
        const fileDetails = await Promise.all(
            files.map(async (file) => {
                const fullPath = path.join(dirPath, file);
                const stats = await fs.stat(fullPath);
                return {
                    name: file,
                    path: fullPath,
                    isDirectory: stats.isDirectory(),
                    size: stats.size,
                    modified: stats.mtime
                };
            })
        );
        return fileDetails;
    } catch (error) {
        console.error('Error reading directory:', error);
        return null;
    }
}

async function ensureDirectoryExists(dirPath) {
    try {
        await fs.access(dirPath);
    } catch {
        await fs.mkdir(dirPath, { recursive: true });
    }
}

// Setup IPC Handlers
function setupIpcHandlers() {
    // Directory Selection Handler
    ipcMain.handle('select-directory', async (event, type = 'data') => {
        try {
            const options = {
                properties: ['openDirectory'],
                ...(type === 'output' && {
                    title: 'Select Output Directory',
                    buttonLabel: 'Select Output Directory',
                    properties: ['openDirectory', 'createDirectory']
                })
            };

            const result = await dialog.showOpenDialog(mainWindow, options);

            if (!result.canceled && result.filePaths.length > 0) {
                const dirPath = result.filePaths[0];

                if (type === 'output') {
                    await ensureDirectoryExists(dirPath);
                    return { path: dirPath, exists: true, isOutput: true };
                }

                const tmseegPath = path.join(dirPath, 'TMSEEG');
                let tmseegContents = null;
                let tmseegExists = false;

                try {
                    await fs.access(tmseegPath);
                    tmseegExists = true;
                    tmseegContents = await getDirectoryContents(tmseegPath);
                } catch (error) {
                    console.log('TMSEEG directory not found:', error);
                }

                const files = await getDirectoryContents(dirPath);
                const hasSesFile = tmseegContents?.some(file => !file.isDirectory && file.name.endsWith('.ses')) ?? false;

                let validStructure = false;
                if (tmseegContents) {
                    const sesFiles = tmseegContents.filter(file => !file.isDirectory && file.name.endsWith('.ses'));
                    validStructure = sesFiles.every(sesFile => {
                        const sessionName = sesFile.name.replace('.ses', '');
                        return tmseegContents.some(item => item.isDirectory && item.name === sessionName);
                    });
                }

                const defaultOutputDir = path.join(dirPath, 'output');
                await ensureDirectoryExists(defaultOutputDir);

                return {
                    path: dirPath,
                    files,
                    tmseegPath,
                    tmseegExists,
                    tmseegContents,
                    hasSesFile,
                    validStructure,
                    outputDir: defaultOutputDir,
                    tmseegpyPath,
                    backendPath
                };
            }
            return null;
        } catch (error) {
            console.error('Error in select-directory:', error);
            throw error;
        }
    });

    ipcMain.handle('get-directory-contents', async (event, dirPath) => {
        try {
            return await getDirectoryContents(dirPath);
        } catch (error) {
            console.error('Error getting directory contents:', error);
            throw error;
        }
    });

    ipcMain.handle('validate-output-directory', async (event, dirPath) => {
        try {
            await ensureDirectoryExists(dirPath);
            const stats = await fs.stat(dirPath);
            return {
                exists: true,
                isDirectory: stats.isDirectory(),
                isWriteable: await fs.access(dirPath, fs.constants.W_OK).then(() => true).catch(() => false)
            };
        } catch (error) {
            console.error('Error validating output directory:', error);
            return { exists: false, isDirectory: false, isWriteable: false, error: error.message };
        }
    });

    // Path Operation Handlers
    ipcMain.handle('join-paths', (event, paths) => path.join(...paths));
    ipcMain.handle('path-basename', (event, { path: pathStr, ext }) => path.basename(pathStr, ext));
    ipcMain.handle('path-dirname', (event, pathStr) => path.dirname(pathStr));
    ipcMain.handle('join-path', (event, ...args) => path.join(...args));
    ipcMain.handle('path-join', (event, pathParts) => path.join(...pathParts));

    // Update Handlers
    ipcMain.handle('check-for-updates', () => !isDev ? autoUpdater.checkForUpdatesAndNotify() : null);
    ipcMain.handle('get-app-version', () => app.getVersion());
    ipcMain.handle('download-update', () => !isDev ? autoUpdater.downloadUpdate() : null);
    ipcMain.handle('install-update', () => !isDev ? autoUpdater.quitAndInstall() : null);
}

// Auto-updater event handlers
function setupAutoUpdater() {
    autoUpdater.on('checking-for-update', () => {
        log.info('Checking for update...');
        mainWindow?.webContents.send('update-status', 'Checking for updates...');
    });

    autoUpdater.on('update-available', (info) => {
        log.info('Update available:', info);
        dialog.showMessageBox(mainWindow, {
            type: 'info',
            title: 'Update Available',
            message: `Version ${info.version} is available. Would you like to download it now?`,
            buttons: ['Yes', 'No']
        }).then(result => result.response === 0 && autoUpdater.downloadUpdate());
    });

    autoUpdater.on('update-not-available', (info) => log.info('Update not available:', info));

    autoUpdater.on('error', (err) => {
        log.error('Error in auto-updater:', err);
        mainWindow?.webContents.send('update-error', err.message);
    });

    autoUpdater.on('download-progress', (progressObj) => {
        const message = `Download speed: ${progressObj.bytesPerSecond} - Downloaded ${progressObj.percent}% (${progressObj.transferred}/${progressObj.total})`;
        log.info(message);
        mainWindow?.webContents.send('download-progress', progressObj);
    });

    autoUpdater.on('update-downloaded', (info) => {
        log.info('Update downloaded:', info);
        mainWindow?.webContents.send('update-status', 'Update downloaded');
        dialog.showMessageBox(mainWindow, {
            type: 'info',
            title: 'Update Ready',
            message: 'Update downloaded. Would you like to install it now?',
            buttons: ['Yes', 'No']
        }).then(result => result.response === 0 && autoUpdater.quitAndInstall());
    });
}


app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});


// Export the necessary functions
module.exports = { createWindow, getResourcePath };