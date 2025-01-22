const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { autoUpdater } = require('electron-updater');
const log = require('electron-log');
const path = require('path');
const isDev = require('electron-is-dev');
const fs = require('fs').promises;

// Configure logging
log.transports.file.level = 'info';
autoUpdater.logger = log;

let mainWindow;

// Helper function to get the correct resource path
function getResourcePath(relativePath) {
    return isDev
        ? path.join(__dirname, '../../..', relativePath) // Dev path
        : path.join(process.resourcesPath, 'app', relativePath); // Prod path
}

// Get paths for tmseegpy and backend
const tmseegpyPath = getResourcePath('tmseegpy');
const backendPath = getResourcePath('backend');

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            enableRemoteModule: false,
            preload: path.join(__dirname, 'preload.js')
        }
    });

    if (isDev) {
        mainWindow.loadURL('http://localhost:3000');
        mainWindow.webContents.openDevTools();
    } else {
        mainWindow.loadFile(path.join(__dirname, '../build/index.html'));
        autoUpdater.checkForUpdatesAndNotify();
    }

    // Add environment variables for Python paths
    process.env.TMSEEGPY_PATH = tmseegpyPath;
    process.env.BACKEND_PATH = backendPath;
}

// Auto-updater events
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
    }).then((result) => {
        if (result.response === 0) {
            autoUpdater.downloadUpdate();
        }
    });
});

autoUpdater.on('update-not-available', (info) => {
    log.info('Update not available:', info);
});

autoUpdater.on('error', (err) => {
    log.error('Error in auto-updater:', err);
    mainWindow?.webContents.send('update-error', err.message);
});

autoUpdater.on('download-progress', (progressObj) => {
    let message = `Download speed: ${progressObj.bytesPerSecond}`;
    message += ` - Downloaded ${progressObj.percent}%`;
    message += ` (${progressObj.transferred}/${progressObj.total})`;
    log.info(message);
    mainWindow?.webContents.send('download-progress', progressObj);
});

autoUpdater.on('update-available', (info) => {
    log.info('Update available:', info);
    mainWindow?.webContents.send('update-status', 'Update available');
    dialog.showMessageBox(mainWindow, {
        type: 'info',
        title: 'Update Ready',
        message: 'Update downloaded. Would you like to install it now?',
        buttons: ['Yes', 'No']
    }).then((result) => {
        if (result.response === 0) {
            autoUpdater.quitAndInstall();
        }
    });
});

// App lifecycle events
app.whenReady().then(() => {
    createWindow();

    // Add update check menu item for development
    if (isDev) {
        mainWindow.webContents.on('context-menu', (_, props) => {
            dialog.showMessageBox(mainWindow, {
                type: 'info',
                title: 'Check for Updates',
                message: 'Would you like to check for updates?',
                buttons: ['Yes', 'No']
            }).then((result) => {
                if (result.response === 0) {
                    autoUpdater.checkForUpdatesAndNotify();
                }
            });
        });
    }
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

// Helper functions
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

// IPC Handlers
ipcMain.handle('select-directory', async (event, type = 'data') => {
    try {
        const options = {
            properties: ['openDirectory']
        };

        if (type === 'output') {
            options.title = 'Select Output Directory';
            options.buttonLabel = 'Select Output Directory';
            options.properties.push('createDirectory');
        }

        const result = await dialog.showOpenDialog(mainWindow, options);

        if (!result.canceled && result.filePaths.length > 0) {
            const dirPath = result.filePaths[0];

            if (type === 'output') {
                await ensureDirectoryExists(dirPath);
                return {
                    path: dirPath,
                    exists: true,
                    isOutput: true
                };
            }

            // Add tmseegpy path information
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

            // Add validation for .ses files and directory structure
            const hasSesFile = tmseegContents ?
                tmseegContents.some(file => !file.isDirectory && file.name.endsWith('.ses')) :
                false;

            let validStructure = false;
            if (tmseegContents) {
                const sesFiles = tmseegContents.filter(file => !file.isDirectory && file.name.endsWith('.ses'));
                validStructure = sesFiles.every(sesFile => {
                    const sessionName = sesFile.name.replace('.ses', '');
                    return tmseegContents.some(item =>
                        item.isDirectory && item.name === sessionName
                    );
                });
            }

            const defaultOutputDir = path.join(dirPath, 'output');
            await ensureDirectoryExists(defaultOutputDir);

            return {
                path: dirPath,
                files: files,
                tmseegPath: tmseegPath,
                tmseegExists: tmseegExists,
                tmseegContents: tmseegContents,
                hasSesFile: hasSesFile,
                validStructure: validStructure,
                outputDir: defaultOutputDir,
                tmseegpyPath: tmseegpyPath,  // Add bundled tmseegpy path
                backendPath: backendPath      // Add bundled backend path
            };
        }
        return null;
    } catch (error) {
        console.error('Error in select-directory:', error);
        throw error;
    }
});

// Add new IPC handler for getting Python paths
ipcMain.handle('get-python-paths', async () => {
    return {
        tmseegpyPath,
        backendPath
    };
});


// [Add new IPC handlers for updates]
ipcMain.handle('check-for-updates', async () => {
    if (!isDev) {
        return autoUpdater.checkForUpdatesAndNotify();
    }
    return null;
});

ipcMain.handle('get-app-version', () => {
    return app.getVersion();
});

ipcMain.handle('download-update', async () => {
    if (!isDev) {
        return autoUpdater.downloadUpdate();
    }
    return null;
});

ipcMain.handle('install-update', async () => {
    if (!isDev) {
        return autoUpdater.quitAndInstall();
    }
    return null;
});



ipcMain.handle('validate-output-directory', async (event, dirPath) => {
    try {
        await ensureDirectoryExists(dirPath);
        const stats = await fs.stat(dirPath);
        return {
            exists: true,
            isDirectory: stats.isDirectory(),
            isWriteable: await fs.access(dirPath, fs.constants.W_OK)
                .then(() => true)
                .catch(() => false)
        };
    } catch (error) {
        console.error('Error validating output directory:', error);
        return {
            exists: false,
            isDirectory: false,
            isWriteable: false,
            error: error.message
        };
    }
});

ipcMain.handle('create-output-directory', async (event, dirPath) => {
    try {
        await ensureDirectoryExists(dirPath);
        return {
            success: true,
            path: dirPath
        };
    } catch (error) {
        console.error('Error creating output directory:', error);
        return {
            success: false,
            error: error.message
        };
    }
});

ipcMain.handle('get-directory-contents', async (event, dirPath) => {
    try {
        return await getDirectoryContents(dirPath);
    } catch (error) {
        console.error('Error getting directory contents:', error);
        return null;
    }
});

ipcMain.handle('check-file-exists', async (event, filePath) => {
    try {
        await fs.access(filePath);
        return true;
    } catch {
        return false;
    }
});

ipcMain.handle('get-file-stats', async (event, filePath) => {
    try {
        const stats = await fs.stat(filePath);
        return {
            size: stats.size,
            modified: stats.mtime,
            created: stats.birthtime,
            isDirectory: stats.isDirectory()
        };
    } catch (error) {
        console.error('Error getting file stats:', error);
        return null;
    }
});

// Path operation handlers
ipcMain.handle('join-paths', async (event, paths) => {
    return path.join(...paths);
});

ipcMain.handle('path-basename', async (event, { path: pathStr, ext }) => {
    return path.basename(pathStr, ext);
});

ipcMain.handle('path-dirname', async (event, pathStr) => {
    return path.dirname(pathStr);
});

// For backward compatibility
ipcMain.handle('join-path', async (event, ...args) => {
    return path.join(...args);
});

ipcMain.handle('path-join', async (event, pathParts) => {
    return path.join(...pathParts);
});