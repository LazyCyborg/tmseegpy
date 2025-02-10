import React, { useState, useRef, useEffect } from 'react';
import { Tab } from '@headlessui/react';
import { api } from './services/api';
import  ICAComponentVisualizer  from './ICAComponentVisualizer'
import  {mapOptionsToBackendParams}  from './services/parameterMapping'
import './styles/Console.css';
import { socket, registerCallbacks, connectWebSocket, disconnectWebSocket } from './services/websocket';
import  ServerCheck  from './ServerCheck';
import useWebSocketHandler from './useWebSocketHandler';
import {
    Settings,
    Check,
    FileInput,
    Activity,
    Sliders,
    Brain,
    Loader2,
    AlertTriangle,
    FolderOpen,
    X,
    Download
} from 'lucide-react';


class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    console.error('Component Error:', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 bg-red-50 text-red-700">
          <AlertTriangle className="inline-block mr-2" />
          Component failed to load
        </div>
      );
    }
    return this.props.children;
  }
}

function classNames(...classes) {
    return classes.filter(Boolean).join(' ');
}



function TmseegpyGUI() {

    const addLog = (message, type = 'info') => {
        setProcessingLogs((prev) => [
            ...prev,
            { message, type, timestamp: new Date() }
        ]);
    };

    const [serverReady, setServerReady] = useState(false);
    const logOutputRef = useRef(null);
    const [timeoutId, setTimeoutId] = useState(null);

    // File and Processing State
    const [selectedFile, setSelectedFile] = useState(null);
    const [processingStatus, setProcessingStatus] = useState('idle');
    const [processingStep, setProcessingStep] = useState('');
    const [progress, setProgress] = useState(0);
    const [processingLogs, setProcessingLogs] = useState([]);
    const [error, setError] = useState(null);
    const [processingComplete, setProcessingComplete] = useState(false);
    const [processingTime, setProcessingTime] = useState('');
    const [filesProcessed, setFilesProcessed] = useState(0);
    const [resultsSummary, setResultsSummary] = useState([]);
    const [statusMessage, setStatusMessage] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [logOutput, setLogOutput] = useState('');
    const [selectedDirectory, setSelectedDirectory] = useState('');
    const [selectedFiles, setSelectedFiles] = useState([]);
    const [updateAvailable, setUpdateAvailable] = useState(false);
    const [updateProgress, setUpdateProgress] = useState(0);
    const [updateError, setUpdateError] = useState(null);
    const [currentVersion, setCurrentVersion] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const consoleRef = useRef(null);
    const [logs, setLogs] = useState([]);
    const [isReady, setIsReady] = useState(false);
    const [showICAModal, setShowICAModal] = useState(false);
    const [icaComponentData, setIcaComponentData] = useState([]);
    const [icaComponentScores, setIcaComponentScores] = useState({});
    const [selectedICAComponents, setSelectedICAComponents] = useState([]);
    const [icaDataReceived, setIcaDataReceived] = useState(false);
    const [loading, setLoading] = useState(false);





         useWebSocketHandler({
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
    timeoutId,
    setTimeoutId,
    addLog,
  });


    const [icaSelectionStatus, setIcaSelectionStatus] = useState({
        isSelecting: false,
        selectedComponents: [],
        totalComponents: 0
    });

    // Basic Options State
    // Basic Options State
const [basicOptions, setBasicOptions] = useState({
    // Core Processing
    processingMode: 'epoched',
    dataFormat: 'neurone',
    dataDir: './data',
    outputDir: './output',
    noPreprocessOutput: false,
    noPcist: false,  // Inverted from performPCIst
    eeglabMontageUnits: 'auto',
    stimChannel: 'STI 014',
    savePreproc: false,
    randomSeed: 42,
    substituteZeroEventsWith: 10,

    // Epoch Settings
    epochsTmin: -0.41,
    epochsTmax: 0.41,

    // Basic Filtering
    lFreq: 0.1,
    hFreq: 45,
    rawHFreq: 250,
    notchFreq: 50,

    // Basic Thresholds
    amplitudeThreshold: 300.0,
    badChannelsThreshold: 3,
    badEpochsThreshold: 3,

    // Basic Processing Flags
    analyzeTeps: true,
    parafacMuscleArtifacts: false,
    plotPreproc: false
});

// Advanced Options State
const [advancedOptions, setAdvancedOptions] = useState({
    // Preprocessing Parameters
    initialSfreq: 1000,
    finalSfreq: 725,

    // Window Parameters
    initialWindowStart: -2,
    initialWindowEnd: 10,
    extendedWindowStart: -2,
    extendedWindowEnd: 15,
    initialInterpWindow: 1.0,
    extendedInterpWindow: 5.0,
    interpolationMethod: 'cubic',

    // Processing Options
    skipSecondArtifactRemoval: false,
    mneFilterEpochs: false,
    plotRaw: false,
    filterRaw: false,

    // Filtering Parameters
    notchWidth: 2,

    // ICA Parameters
    icaMethod: 'fastica',
    firstIcaManual: true,
    secondIcaManual: true,
    noFirstIca: false,
    noSecondIca: false,
    secondIcaMethod: 'fastica',

    // Artifact Thresholds
    blinkThresh: 2.5,
    latEyeThresh: 2.0,
    noiseThresh: 4.0,
    tmsMuscleThresh: 2.0,
    muscleThresh: 0.6,

    // Muscle Artifact Parameters
    muscleWindowStart: 0.005,
    muscleWindowEnd: 0.030,
    thresholdFactor: 1.0,
    nComponents: 5,

    // SSP and CSD Parameters
    applySsp: false,
    sspNEeg: 2,
    applyCsd: false,
    lambda2: 1e-3,
    stiffness: 4,

    // TEP Analysis Parameters
    saveEvoked: false,
    tepAnalysisType: 'gmfa',  // Changed from tep_analysis_type
    tepRoiChannels: ['C3', 'C4'],  // Changed from tep_roi_channels
    tepMethod: 'largest',  // Changed from tep_method
    tepSamples: 5,  // Changed from tep_samples
    saveValidation: false,

    // Window Parameters for Analysis
    baselineStart: -400,
    baselineEnd: -50,
    responseStart: 0,
    responseEnd: 299,

    // PCIst Parameters
    k: 1.2,
    minSnr: 1.1,
    maxVar: 99.0,
    embed: false,
    nSteps: 100,
    preWindowStart: -400,
    preWindowEnd: -50,
    postWindowStart: 0,
    postWindowEnd: 300,

    // Research Mode
    research: false
});




// Keep the update checking effect separate
useEffect(() => {
    if (window.electron?.getAppVersion) {
        window.electron.getAppVersion().then(version => {
            setCurrentVersion(version);
        });
    }

    if (window.electron?.onUpdateStatus) {
        const removeUpdateStatus = window.electron.onUpdateStatus((status) => {
            console.log('Update status:', status);
            if (status.includes('Update available')) {
                setUpdateAvailable(true);
            }
        });

        const removeUpdateProgress = window.electron.onUpdateProgress((progress) => {
            setUpdateProgress(progress.percent);
        });

        const removeUpdateError = window.electron.onUpdateError((error) => {
            setUpdateError(error);
            console.error('Update error:', error);
        });

        if (window.electron.checkForUpdates) {
            window.electron.checkForUpdates();
        }

        return () => {
            removeUpdateStatus();
            removeUpdateProgress();
            removeUpdateError();
        };
    }
}, []);


    const handleServerReady = () => {
        setServerReady(true);
    };




const handleDirectorySelect = async () => {
    try {
        setIsProcessing(true);
        setError(null);

        if (!window?.electron) {
            throw new Error('Electron API not available');
        }

        const result = await window.electron.selectDirectory();
        console.log('Directory selection result:', result);

        if (result) {
            const parentDir = result.path;
            console.log('Selected directory:', parentDir); // Debug log

            // Set up default output directory
            const defaultOutputDir = `${parentDir}/output`;

            // Update states
            setSelectedDirectory(parentDir);
            setBasicOptions(prev => ({
                ...prev,
                dataDir: parentDir,
                outputDir: defaultOutputDir
            }));

            // Notify server
            const formData = new FormData();
            formData.append('parentDirectory', parentDir);
            formData.append('tmseegDirectory', result.tmseegPath);

            const response = await api.upload(formData);
            if (response.data?.message) {
                setStatusMessage(response.data.message);
                setSelectedFiles(response.data.sessions || []);
            }
        }
    } catch (error) {
        console.error('Directory selection failed:', error);
        setError(error.message || 'Failed to select directory');
    } finally {
        setIsProcessing(false);
    }
};

const handleClearOutputDirectory = () => {
    setBasicOptions(prev => ({
        ...prev,
        outputDir: './output'
    }));
};

const handleClearSelection = () => {
    setSelectedDirectory(null);
    setSelectedFiles([]);
    setStatusMessage('');
    setError(null);
};




const handleStartProcessing = async () => {
    if (!selectedDirectory) {
        setError('Please select a directory first');
        return;
    }

    try {
        // Reset states
        setProcessingStatus('processing');
        setProgress(0);
        setError(null);
        setIsProcessing(true);
        setProcessingComplete(false);
        setProcessingTime('');
        setFilesProcessed(0);
        setResultsSummary([]);

        // Map the parameters
        const mappedParams = mapParametersToBackend(basicOptions, advancedOptions, selectedDirectory);

        // Validate the mapped parameters
        validateParameters(mappedParams);

        // Log what we're sending (for debugging)
        console.log('Starting processing with options:', mappedParams);

        setProcessingLogs(prev => [
            ...prev,
            'Starting processing...',
            `Input directory: ${selectedDirectory}`,
            `Output directory: ${mappedParams.output_dir}`
        ]);

        // Make the API call with the mapped parameters
        const response = await api.process(mappedParams);

        if (response.data?.message) {
            setStatusMessage(response.data.message);
            setProcessingLogs(prev => [...prev, response.data.message]);
        }

    } catch (error) {
        console.error('Processing failed:', error);
        const errorMessage = error.response?.data?.error || error.message || 'Processing failed';

        setProcessingStatus('error');
        setError(errorMessage);
        setIsProcessing(false);
        setProcessingLogs(prev => [
            ...prev,
            `Error: ${errorMessage}`,
            'Processing stopped due to error'
        ]);

        try {
            await api.stop();
        } catch (cleanupError) {
            console.error('Failed to stop processing:', cleanupError);
        }
    }
};

const handleStopProcessing = async () => {
    try {
        await api.stop();
        setProcessingStatus('stopped');
        setProcessingLogs(prev => [...prev, 'Processing stopped by user']);
    } catch (error) {
        console.error('Failed to stop processing:', error);
        setError(error.message);
    }
};

const handleClearLog = () => {
    setProcessingLogs([]);
};

const handleDownloadResults = async () => {
    try {
        const response = await api.getResults();
        const results = response.data;

        const element = document.createElement('a');
        const file = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
        element.href = URL.createObjectURL(file);
        element.download = 'processing_results.json';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    } catch (error) {
        console.error('Failed to download results:', error);
        setError(error.message);
    }
};

const renderConsole = () => (
    <div className="console-container">
        <div className="console-header">
            <span className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
                {isConnected ? 'Connected' : 'Disconnected'}
            </span>
        </div>
        <div
            ref={consoleRef}
            className="console-output"
            id="console-output"
        >
            {logs.map((log, index) => (
                <div
                    key={index}
                    className={`console-line ${log.type}`}
                >
                    {log.message}
                </div>
            ))}
        </div>
    </div>
);

    return (
    <>
        <ServerCheck onServerReady={handleServerReady} />
        {serverReady && (
            <div className="min-h-screen bg-gray-50">
                {/* Header */}
                <header className="bg-white shadow">
                    <div className="max-w-7xl mx-auto py-4 px-4">
                        <h1 className="text-2xl font-semibold text-gray-900">TMSeegpy GUI</h1>
                    </div>
                </header>

                            {/* Main Content */}
                            <main className="max-w-7xl mx-auto py-6 px-4">
                                {updateAvailable && (
                    <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-md">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center">
                                <Download className="h-5 w-5 text-blue-400 mr-2" />
                                <div className="text-sm text-blue-700">
                                    A new version is available.
                                    {currentVersion && ` Current version: ${currentVersion}`}
                                </div>
                            </div>
                            <button
                                onClick={() => window.electron?.downloadUpdate()}
                                className="px-3 py-1 text-sm text-white bg-blue-600 hover:bg-blue-700 rounded-md"
                            >
                                Update Now
                            </button>
                        </div>
                        {updateProgress > 0 && (
                            <div className="mt-2">
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                    <div
                                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                        style={{ width: `${updateProgress}%` }}
                                    ></div>
                                </div>
                                <div className="text-xs text-gray-500 mt-1">
                                    Downloading: {updateProgress.toFixed(1)}%
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {updateError && (
                    <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-md">
                        <div className="flex items-center">
                            <AlertTriangle className="h-5 w-5 text-red-400 mr-2" />
                            <div className="text-sm text-red-700">
                                Update error: {updateError}
                            </div>
                        </div>
                    </div>
                )}
                <Tab.Group>
                    <Tab.List className="flex space-x-1 rounded-xl bg-blue-900/20 p-1">
                        {/* Input Data Tab */}
                        <Tab className={({selected}) =>
                            classNames(
                                'w-full rounded-lg py-2.5 text-sm font-medium leading-5',
                                'ring-white ring-opacity-60 ring-offset-2 ring-offset-blue-400 focus:outline-none focus:ring-2',
                                selected
                                    ? 'bg-white shadow text-blue-700'
                                    : 'text-blue-500 hover:bg-white/[0.12] hover:text-blue-600'
                            )
                        }>
                            <div className="flex items-center justify-center space-x-2">
                                <FileInput className="w-5 h-5"/>
                                <span>Input Data</span>
                            </div>
                        </Tab>

                        {/* Basic Options Tab */}
                        <Tab className={({selected}) =>
                            classNames(
                                'w-full rounded-lg py-2.5 text-sm font-medium leading-5',
                                'ring-white ring-opacity-60 ring-offset-2 ring-offset-blue-400 focus:outline-none focus:ring-2',
                                selected
                                    ? 'bg-white shadow text-blue-700'
                                    : 'text-blue-500 hover:bg-white/[0.12] hover:text-blue-600'
                            )
                        }>
                            <div className="flex items-center justify-center space-x-2">
                                <Settings className="w-5 h-5"/>
                                <span>Basic Options</span>
                            </div>
                        </Tab>

                        {/* Advanced Options Tab */}
                        <Tab className={({selected}) =>
                            classNames(
                                'w-full rounded-lg py-2.5 text-sm font-medium leading-5',
                                'ring-white ring-opacity-60 ring-offset-2 ring-offset-blue-400 focus:outline-none focus:ring-2',
                                selected
                                    ? 'bg-white shadow text-blue-700'
                                    : 'text-blue-500 hover:bg-white/[0.12] hover:text-blue-600'
                            )
                        }>
                            <div className="flex items-center justify-center space-x-2">
                                <Sliders className="w-5 h-5"/>
                                <span>Advanced Options</span>
                            </div>
                        </Tab>

                        {/* Processing Tab */}
                        <Tab className={({selected}) =>
                            classNames(
                                'w-full rounded-lg py-2.5 text-sm font-medium leading-5',
                                'ring-white ring-opacity-60 ring-offset-2 ring-offset-blue-400 focus:outline-none focus:ring-2',
                                selected
                                    ? 'bg-white shadow text-blue-700'
                                    : 'text-blue-500 hover:bg-white/[0.12] hover:text-blue-600'
                            )
                        }>
                            <div className="flex items-center justify-center space-x-2">
                                <Activity className="w-5 h-5"/>
                                <span>Processing</span>
                            </div>
                        </Tab>
                    </Tab.List>

                                    <Tab.Panels className="mt-4">
                                {/* Input Data Panel */}
                                <Tab.Panel className="bg-white rounded-xl p-6 shadow">
                                    <div className="space-y-6">
                                        <h2 className="text-lg font-medium">Data Input</h2>

                                        {error && (
                                            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded relative">
                                                <AlertTriangle className="inline-block mr-2 h-5 w-5" />
                                                <span>{error}</span>
                                            </div>
                                        )}

                                        {statusMessage && !error && (
                                            <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded relative">
                                                <span>{statusMessage}</span>
                                            </div>
                                        )}

                                        <div className="space-y-6">
                                            {/* Data Directory Selection */}
                                            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
                                                <div className="text-center mb-4">
                                                    <h3 className="text-lg font-medium text-gray-900">Select Data Directory</h3>
                                                    <p className="text-sm text-gray-500">
                                                        Choose a directory named TMSEEG containing .ses files and related data folders (folder must be named TMSEEG lol)
                                                    </p>
                                                </div>

                                                <div className="flex flex-col items-center space-y-4">
                                                    <input
                                                        type="text"
                                                        className="w-full px-4 py-2 border border-gray-300 rounded-md bg-gray-50"
                                                        value={selectedDirectory || ''}
                                                        placeholder="No directory selected"
                                                        readOnly
                                                    />

                                                    <div className="flex space-x-4">
                                                        <button
                                                            onClick={handleDirectorySelect}
                                                            className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 transition-colors duration-200 ${
                                                                isProcessing ? 'opacity-50 cursor-not-allowed' : ''
                                                            }`}
                                                            disabled={isProcessing}
                                                        >
                                                            {isProcessing ? (
                                                                <Loader2 className="animate-spin -ml-1 mr-2 h-4 w-4" />
                                                            ) : (
                                                                <FolderOpen className="mr-2 h-4 w-4" />
                                                            )}
                                                            Select Directory
                                                        </button>

                                                        {selectedDirectory && (
                                                            <button
                                                                onClick={handleClearSelection}
                                                                className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 transition-colors duration-200"
                                                            >
                                                                <X className="mr-2 h-4 w-4" />
                                                                Clear Selection
                                                            </button>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Output Directory Selection */}
                                            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
                                                <div className="text-center mb-4">
                                                    <h3 className="text-lg font-medium text-gray-900">Select Output Directory</h3>
                                                    <p className="text-sm text-gray-500">
                                                        Choose a directory where processed files will be saved
                                                    </p>
                                                </div>

                                                <div className="flex flex-col items-center space-y-4">
                                                    <input
                                                        type="text"
                                                        className="w-full px-4 py-2 border border-gray-300 rounded-md bg-gray-50"
                                                        value={basicOptions.outputDir || ''}
                                                        placeholder="No output directory selected"
                                                        readOnly
                                                    />

                                                    <div className="flex space-x-4">
                                                        <button
                                                            onClick={async () => {
                                                                try {
                                                                    if (!window?.electron) {
                                                                        throw new Error('Electron API not available');
                                                                    }
                                                                    const result = await window.electron.selectDirectory();
                                                                    if (result && result.path) {
                                                                        setBasicOptions(prev => ({
                                                                            ...prev,
                                                                            outputDir: result.path
                                                                        }));
                                                                    }
                                                                } catch (error) {
                                                                    setError('Failed to select output directory: ' + error.message);
                                                                }
                                                            }}
                                                            className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 transition-colors duration-200 ${
                                                                isProcessing ? 'opacity-50 cursor-not-allowed' : ''
                                                            }`}
                                                            disabled={isProcessing}
                                                        >
                                                            <FolderOpen className="mr-2 h-4 w-4" />
                                                            Select Output Directory
                                                        </button>

                                                        {basicOptions.outputDir && basicOptions.outputDir !== './output' && (
                                                            <button
                                                                onClick={() => {
                                                                    setBasicOptions(prev => ({
                                                                        ...prev,
                                                                        outputDir: './output'  // Reset to default
                                                                    }));
                                                                }}
                                                                className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 transition-colors duration-200"
                                                            >
                                                                <X className="mr-2 h-4 w-4" />
                                                                Reset to Default
                                                            </button>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Selected Files Display */}
                                            {selectedDirectory && (
                                                <div className="mt-6">
                                                    <h3 className="text-md font-medium text-gray-700 mb-2">Selected Files</h3>
                                                    <div className="bg-gray-50 rounded-md p-4 max-h-60 overflow-y-auto">
                                                        <div className="space-y-2">
                                                            {selectedFiles.map((file, index) => {
                                                                const fileName = typeof file === 'string' ? file : file.name;
                                                                const isSesFile = fileName.toLowerCase().endsWith('.ses');

                                                                return (
                                                                    <div key={index} className="flex items-center justify-between py-1 px-2 hover:bg-gray-100 rounded-md">
                                                                        <span className="text-sm text-gray-600">
                                                                            {fileName}
                                                                        </span>
                                                                        <span className={`text-xs px-2 py-1 rounded-full ${
                                                                            isSesFile 
                                                                                ? 'bg-blue-100 text-blue-800' 
                                                                                : 'bg-gray-100 text-gray-600'
                                                                        }`}>
                                                                            {isSesFile ? 'Session File' : 'Data File'}
                                                                        </span>
                                                                    </div>
                                                                );
                                                            })}
                                                        </div>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                                    </Tab.Panel>
                        {/* Basic Options Panel */}
                        <Tab.Panel className="bg-white rounded-xl p-6 shadow">
                            <div className="space-y-6">
                                <h2 className="text-lg font-medium">Basic Configuration</h2>

                                {/* Core Processing Section */}
                                <div className="space-y-4">
                                    <h3 className="text-md font-medium text-gray-700">Core Processing</h3>

                                    <div className="grid grid-cols-2 gap-4">

                                        {/* Data Format */}
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Data Format</label>
                                            <select
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={basicOptions.dataFormat}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    dataFormat: e.target.value
                                                }))}
                                            >
                                                <option value="neurone">Neurone</option>
                                                <option value="brainvision">BrainVision</option>
                                                <option value="edf">EDF</option>
                                                <option value="cnt">CNT</option>
                                                <option value="eeglab">EEGLAB</option>
                                                <option value="auto">Auto</option>
                                            </select>
                                        </div>
                                    </div>

                                    {/* EEGLAB Montage Units */}
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700">EEGLAB Montage Units</label>
                                        <select
                                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                            value={basicOptions.eeglabMontageUnits}
                                            onChange={(e) => setBasicOptions(prev => ({
                                                ...prev,
                                                eeglabMontageUnits: e.target.value
                                            }))}
                                        >
                                            <option value="auto">Auto</option>
                                            <option value="mm">Millimeters</option>
                                            <option value="cm">Centimeters</option>
                                            <option value="m">Meters</option>
                                        </select>
                                    </div>

                                    {/* Stim Channel */}
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700">Stim Channel</label>
                                        <input
                                            type="text"
                                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                            value={basicOptions.stimChannel}
                                            onChange={(e) => setBasicOptions(prev => ({
                                                ...prev,
                                                stimChannel: e.target.value
                                            }))}
                                            placeholder="STI 014"
                                        />
                                    </div>

                                    {/* Substitute Zero Events With */}
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700">Substitute Zero Events With</label>
                                        <input
                                            type="number"
                                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                            value={basicOptions.substituteZeroEventsWith}
                                            onChange={(e) => setBasicOptions(prev => ({
                                                ...prev,
                                                substituteZeroEventsWith: parseInt(e.target.value)
                                            }))}
                                        />
                                    </div>

                                    {/* Checkboxes for boolean options */}
                                    <div className="space-y-3">
                                        <div className="flex items-center">
                                            <input
                                                type="checkbox"
                                                id="noPreprocessOutput"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={basicOptions.noPreprocessOutput}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    noPreprocessOutput: e.target.checked
                                                }))}
                                            />
                                            <label htmlFor="noPreprocessOutput" className="ml-2 text-sm font-medium text-gray-700">
                                                Skip Preprocess Output
                                            </label>
                                        </div>

                                        <div className="flex items-center">
                                            <input
                                                type="checkbox"
                                                id="noPcist"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={basicOptions.noPcist}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    noPcist: e.target.checked
                                                }))}
                                            />
                                            <label htmlFor="noPcist" className="ml-2 text-sm font-medium text-gray-700">
                                                Skip PCIst Calculation
                                            </label>
                                        </div>

                                        <div className="flex items-center">
                                            <input
                                                type="checkbox"
                                                id="savePreproc"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={basicOptions.savePreproc}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    savePreproc: e.target.checked
                                                }))}
                                            />
                                            <label htmlFor="savePreproc" className="ml-2 text-sm font-medium text-gray-700">
                                                Save Preprocessing Results
                                            </label>
                                        </div>
                                    </div>
                                </div>

                                {/* Epoch Settings Section */}
                                <div className="space-y-4">
                                    <h3 className="text-md font-medium text-gray-700">Epoch Settings</h3>

                                    <div className="grid grid-cols-2 gap-4">
                                        {/* Epoch Start Time */}
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">
                                                Epoch Start (s)
                                                <span className="text-gray-500 text-xs ml-1">Default: -0.41</span>
                                            </label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={basicOptions.epochsTmin}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    epochsTmin: parseFloat(e.target.value)
                                                }))}
                                                step="0.01"
                                            />
                                            <p className="mt-1 text-xs text-gray-500">Recommended range: -2 to 0</p>
                                        </div>

                                        {/* Epoch End Time */}
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">
                                                Epoch End (s)
                                                <span className="text-gray-500 text-xs ml-1">Default: 0.41</span>
                                            </label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={basicOptions.epochsTmax}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    epochsTmax: parseFloat(e.target.value)
                                                }))}
                                                step="0.01"
                                            />
                                            <p className="mt-1 text-xs text-gray-500">Recommended range: 0 to 2</p>
                                        </div>
                                    </div>

                                    {/* Window Parameters */}
                                    <div className="grid grid-cols-2 gap-4 mt-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">
                                                Baseline Start (ms)
                                                <span className="text-gray-500 text-xs ml-1">Default: -400</span>
                                            </label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={basicOptions.baselineStart}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    baselineStart: parseInt(e.target.value)
                                                }))}
                                                step="10"
                                            />
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">
                                                Baseline End (ms)
                                                <span className="text-gray-500 text-xs ml-1">Default: -50</span>
                                            </label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={basicOptions.baselineEnd}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    baselineEnd: parseInt(e.target.value)
                                                }))}
                                                step="10"
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Basic Filtering Section */}
                                <div className="space-y-4 mt-6">
                                    <h3 className="text-md font-medium text-gray-700">Basic Filtering</h3>

                                    <div className="grid grid-cols-2 gap-4">
                                        {/* Low Frequency Filter */}
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">
                                                Low-pass Filter (Hz)
                                                <span className="text-gray-500 text-xs ml-1">Default: 0.1</span>
                                            </label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={basicOptions.lFreq}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    lFreq: parseFloat(e.target.value)
                                                }))}
                                                step="0.1"
                                                min="0"
                                            />
                                            <p className="mt-1 text-xs text-gray-500">Recommended range: 0.1 to 1 Hz</p>
                                        </div>

                                        {/* High Frequency Filter */}
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">
                                                High-pass Filter (Hz)
                                                <span className="text-gray-500 text-xs ml-1">Default: 45</span>
                                            </label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={basicOptions.hFreq}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    hFreq: parseFloat(e.target.value)
                                                }))}
                                                step="1"
                                                min="0"
                                            />
                                            <p className="mt-1 text-xs text-gray-500">Recommended range: 30 to 100 Hz</p>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        {/* Raw High Frequency Filter */}
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">
                                                Raw High-pass Filter (Hz)
                                                <span className="text-gray-500 text-xs ml-1">Default: 250</span>
                                            </label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={basicOptions.rawHFreq}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    rawHFreq: parseFloat(e.target.value)
                                                }))}
                                                step="1"
                                                min="0"
                                            />
                                        </div>

                                        {/* Notch Filter */}
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">
                                                Notch Filter (Hz)
                                                <span className="text-gray-500 text-xs ml-1">Default: 50</span>
                                            </label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={basicOptions.notchFreq}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    notchFreq: parseFloat(e.target.value)
                                                }))}
                                                step="1"
                                                min="0"
                                            />
                                            <p className="mt-1 text-xs text-gray-500">Usually 50 Hz (Europe) or 60 Hz (USA)</p>
                                        </div>
                                    </div>

                                    {/* Notch Width */}
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700">
                                            Notch Width (Hz)
                                            <span className="text-gray-500 text-xs ml-1">Default: 2</span>
                                        </label>
                                        <input
                                            type="number"
                                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                            value={basicOptions.notchWidth}
                                            onChange={(e) => setBasicOptions(prev => ({
                                                ...prev,
                                                notchWidth: parseFloat(e.target.value)
                                            }))}
                                            step="0.1"
                                            min="0"
                                        />
                                    </div>

                                    {/* Filter Options */}
                                    <div className="space-y-3 mt-4">
                                        <div className="flex items-center">
                                            <input
                                                type="checkbox"
                                                id="filterRaw"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={basicOptions.filterRaw}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    filterRaw: e.target.checked
                                                }))}
                                            />
                                            <label htmlFor="filterRaw" className="ml-2 text-sm font-medium text-gray-700">
                                                Apply Raw Data Filtering
                                            </label>
                                        </div>

                                        <div className="flex items-center">
                                            <input
                                                type="checkbox"
                                                id="mneFilterEpochs"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={basicOptions.mneFilterEpochs}
                                                onChange={(e) => setBasicOptions(prev => ({
                                                    ...prev,
                                                    mneFilterEpochs: e.target.checked
                                                }))}
                                            />
                                            <label htmlFor="mneFilterEpochs" className="ml-2 text-sm font-medium text-gray-700">
                                                Use MNE Filter for Epochs
                                            </label>
                                        </div>
                                    </div>
                                </div>

                                {/* Basic Thresholds Section */}
                                <div className="space-y-4">
                                    <h3 className="text-md font-medium text-gray-700">Basic Thresholds</h3>

                                    <div className="grid grid-cols-3 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Amplitude Threshold (µV)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={basicOptions.amplitudeThreshold}
                                                onChange={(e) => setBasicOptions({
                                                    ...basicOptions,
                                                    amplitudeThreshold: parseFloat(e.target.value)
                                                })}
                                                step="1"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Bad Channels Threshold</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={basicOptions.badChannelsThreshold}
                                                onChange={(e) => setBasicOptions({
                                                    ...basicOptions,
                                                    badChannelsThreshold: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Bad Epochs Threshold</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={basicOptions.badEpochsThreshold}
                                                onChange={(e) => setBasicOptions({
                                                    ...basicOptions,
                                                    badEpochsThreshold: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Basic Processing Flags Section */}
                                <div className="space-y-4">
                                    <h3 className="text-md font-medium text-gray-700">Processing Flags</h3>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="space-y-3">
                                            <div className="flex items-center space-x-3">
                                                <input
                                                    type="checkbox"
                                                    id="validate-teps"
                                                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                    checked={basicOptions.validateTEPs}
                                                    onChange={(e) => setBasicOptions({
                                                        ...basicOptions,
                                                        validateTEPs: e.target.checked
                                                    })}
                                                />
                                                <label htmlFor="validate-teps" className="text-sm font-medium text-gray-700">
                                                    Validate TEPs
                                                </label>
                                            </div>

                                            <div className="flex items-center space-x-3">
                                                <input
                                                    type="checkbox"
                                                    id="perform-pcist"
                                                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                    checked={basicOptions.performPCIst}
                                                    onChange={(e) => setBasicOptions({
                                                        ...basicOptions,
                                                        performPCIst: e.target.checked
                                                    })}
                                                />
                                                <label htmlFor="perform-pcist" className="text-sm font-medium text-gray-700">
                                                    Perform PCIst
                                                </label>
                                            </div>
                                        </div>

                                        <div className="space-y-3">
                                            <div className="flex items-center space-x-3">
                                                <input
                                                    type="checkbox"
                                                    id="plot-preproc"
                                                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                    checked={basicOptions.plot_preproc}
                                                    onChange={(e) => setBasicOptions({
                                                        ...basicOptions,
                                                        plot_preproc: e.target.checked
                                                    })}
                                                />
                                                <label htmlFor="plot-preproc" className="text-sm font-medium text-gray-700">
                                                    Plot Preprocessing
                                                </label>
                                            </div>

                                            <div className="flex items-center space-x-3">
                                                <input
                                                    type="checkbox"
                                                    id="parafac-muscle"
                                                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                    checked={basicOptions.parafac_muscle_artifacts}
                                                    onChange={(e) => setBasicOptions({
                                                        ...basicOptions,
                                                        parafac_muscle_artifacts: e.target.checked
                                                    })}
                                                />
                                                <label htmlFor="parafac-muscle" className="text-sm font-medium text-gray-700">
                                                    Use PARAFAC Muscle Artifact Removal
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Tab.Panel>
                        {/* Advanced Options Panel */}
                        <Tab.Panel className="bg-white rounded-xl p-6 shadow">
                            <div className="space-y-6">
                                <h2 className="text-lg font-medium">Advanced Configuration</h2>

                                {/* Preprocessing Parameters Section */}
                                <div className="space-y-4">
                                    <h3 className="text-md font-medium text-gray-700">Preprocessing Parameters</h3>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Final Sampling
                                                Rate (Hz)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.finalSfreq}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    finalSfreq: parseFloat(e.target.value)
                                                })}
                                            />

                                        </div>
                                        <div>

                                        </div>
                                    </div>

                                    <div>
                                        <label className="block text-sm font-medium text-gray-700">Stim Channel</label>
                                        <input
                                            type="text"
                                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                            value={advancedOptions.stimChannel}
                                            onChange={(e) => setAdvancedOptions({
                                                ...advancedOptions,
                                                stimChannel: e.target.value
                                            })}
                                        />
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>

                                        </div>

                                    </div>

                                    <div>
                                        <label className="block text-sm font-medium text-gray-700">EEGLab Montage Units</label>
                                        <select
                                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                            value={advancedOptions.eegLabMontageUnits}
                                            onChange={(e) => setAdvancedOptions({
                                                ...advancedOptions,
                                                eegLabMontageUnits: e.target.value
                                            })}
                                        >
                                            <option value="auto">Auto</option>
                                            <option value="mm">Millimeters</option>
                                            <option value="cm">Centimeters</option>
                                            <option value="m">Meters</option>
                                        </select>
                                    </div>
                                </div>
                        {/* Artifact Removal Section */}
                                <div className="space-y-4">
                                    <h3 className="text-md font-medium text-gray-700">Artifact Removal</h3>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Initial Window Start (s)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.initialWindowStart}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    initialWindowStart: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Initial Window End (s)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.initialWindowEnd}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    initialWindowEnd: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Extended Window Start (s)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.extendedWindowStart}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    extendedWindowStart: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Extended Window End (s)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.extendedWindowEnd}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    extendedWindowEnd: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Initial Interpolation Window (s)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.initialInterpWindow}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    initialInterpWindow: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Extended Interpolation Window (s)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.extendedInterpWindow}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    extendedInterpWindow: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                    </div>

                                    <div className="flex items-center space-x-3">
                                        <input
                                            type="checkbox"
                                            id="skip-second-artifact"
                                            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                            checked={advancedOptions.skipSecondArtifactRemoval}
                                            onChange={(e) => setAdvancedOptions({
                                                ...advancedOptions,
                                                skipSecondArtifactRemoval: e.target.checked
                                            })}
                                        />
                                        <label htmlFor="skip-second-artifact" className="text-sm font-medium text-gray-700">
                                            Skip Second Artifact Removal
                                        </label>
                                    </div>
                                </div>

                                {/* ICA Settings Section */}
                                <div className="space-y-4">
                                    <h3 className="text-md font-medium text-gray-700">ICA Settings</h3>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="space-y-3">
                                            <div className="flex items-center space-x-3">
                                                <input
                                                    type="checkbox"
                                                    id="first-ica-manual"
                                                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                    checked={advancedOptions.firstIcaManual}
                                                    onChange={(e) => setAdvancedOptions({
                                                        ...advancedOptions,
                                                        firstIcaManual: e.target.checked
                                                    })}
                                                />
                                                <label htmlFor="first-ica-manual" className="text-sm font-medium text-gray-700">
                                                    Manual First ICA
                                                </label>
                                            </div>
                                            <div className="flex items-center space-x-3">
                                                <input
                                                    type="checkbox"
                                                    id="second-ica-manual"
                                                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                    checked={advancedOptions.secondIcaManual}
                                                    onChange={(e) => setAdvancedOptions({
                                                        ...advancedOptions,
                                                        secondIcaManual: e.target.checked
                                                    })}
                                                />
                                                <label htmlFor="second-ica-manual" className="text-sm font-medium text-gray-700">
                                                    Manual Second ICA
                                                </label>
                                            </div>
                                        </div>

                                        <div className="space-y-3">
                                            <div className="flex items-center space-x-3">
                                                <input
                                                    type="checkbox"
                                                    id="no-second-ica"
                                                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                    checked={advancedOptions.no_second_ICA}
                                                    onChange={(e) => setAdvancedOptions({
                                                        ...advancedOptions,
                                                        no_second_ICA: e.target.checked
                                                    })}
                                                />
                                                <label htmlFor="no-second-ica" className="text-sm font-medium text-gray-700">
                                                    Skip Second ICA
                                                </label>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">First ICA Method</label>
                                            <select
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.icaMethod}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    icaMethod: e.target.value
                                                })}
                                            >
                                                <option value="fastica">FastICA</option>
                                                <option value="infomax">Infomax</option>
                                                <option value="picard">Picard</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Second ICA Method</label>
                                            <select
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.secondIcaMethod}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    secondIcaMethod: e.target.value
                                                })}
                                            >
                                                <option value="fastica">FastICA</option>
                                                <option value="infomax">Infomax</option>
                                                <option value="picard">Picard</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Blink Threshold</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.blinkThresh}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    blinkThresh: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Lateral Eye Movement Threshold</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.latEyeThresh}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    latEyeThresh: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-3 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Noise Threshold</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.noiseThresh}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    noiseThresh: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">TMS Muscle Threshold</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.tmsMuscleThresh}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    tmsMuscleThresh: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Muscle Threshold</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.muscleThresh}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    muscleThresh: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                    </div>
                                </div>
                                {/* PARAFAC Muscle Artifact Settings */}
                                <div className="space-y-4">
                                    <h3 className="text-md font-medium text-gray-700">PARAFAC Muscle Artifact Settings</h3>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Muscle Window Start (s)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.muscleWindowStart}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    muscleWindowStart: parseFloat(e.target.value)
                                                })}
                                                step="0.001"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Muscle Window End (s)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.muscleWindowEnd}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    muscleWindowEnd: parseFloat(e.target.value)
                                                })}
                                                step="0.001"
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Threshold Factor</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.thresholdFactor}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    thresholdFactor: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Number of Components</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.nComponents}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    nComponents: parseInt(e.target.value)
                                                })}
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Filter Settings */}
                                <div className="space-y-4">
                                    <h3 className="text-md font-medium text-gray-700">Filter Settings</h3>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Notch Filter Width (Hz)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.notchWidth}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    notchWidth: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                    </div>

                                    <div className="space-y-3">
                                        <div className="flex items-center space-x-3">
                                            <input
                                                type="checkbox"
                                                id="mne-filter-epochs"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={advancedOptions.mneFilterEpochs}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    mneFilterEpochs: e.target.checked
                                                })}
                                            />
                                            <label htmlFor="mne-filter-epochs" className="text-sm font-medium text-gray-700">
                                                Use MNE Filter for Epochs
                                            </label>
                                        </div>

                                        <div className="flex items-center space-x-3">
                                            <input
                                                type="checkbox"
                                                id="filter-raw"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={advancedOptions.filterRaw}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    filterRaw: e.target.checked
                                                })}
                                            />
                                            <label htmlFor="filter-raw" className="text-sm font-medium text-gray-700">
                                                Filter Raw Data (Lowpass filter at 250 Hz)
                                            </label>
                                        </div>
                                    </div>
                                </div>

                                {/* Additional Processing */}
                                <div className="space-y-4">
                                    <h3 className="text-md font-medium text-gray-700">Additional Processing</h3>

                                    <div className="space-y-3">
                                        <div className="flex items-center space-x-3">
                                            <input
                                                type="checkbox"
                                                id="apply-ssp"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={advancedOptions.applySsp}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    applySsp: e.target.checked
                                                })}
                                            />
                                            <label htmlFor="apply-ssp" className="text-sm font-medium text-gray-700">
                                                Apply Signal Space Projection (SSP)
                                            </label>
                                        </div>

                                        {advancedOptions.applySsp && (
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700">Number of EEG Components for SSP</label>
                                                <input
                                                    type="number"
                                                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                    value={advancedOptions.sspNEeg}
                                                    onChange={(e) => setAdvancedOptions({
                                                        ...advancedOptions,
                                                        sspNEeg: parseInt(e.target.value)
                                                    })}
                                                />
                                            </div>
                                        )}

                                        <div className="flex items-center space-x-3">
                                            <input
                                                type="checkbox"
                                                id="apply-csd"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={advancedOptions.applyCsd}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    applyCsd: e.target.checked
                                                })}
                                            />
                                            <label htmlFor="apply-csd" className="text-sm font-medium text-gray-700">
                                                Apply Current Source Density (CSD)
                                            </label>
                                        </div>

                                        {advancedOptions.applyCsd && (
                                            <div className="grid grid-cols-2 gap-4">
                                                <div>
                                                    <label className="block text-sm font-medium text-gray-700">Lambda²</label>
                                                    <input
                                                        type="number"
                                                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                        value={advancedOptions.lambda2}
                                                        onChange={(e) => setAdvancedOptions({
                                                            ...advancedOptions,
                                                            lambda2: parseFloat(e.target.value)
                                                        })}
                                                        step="0.001"
                                                    />
                                                </div>
                                                <div>
                                                    <label className="block text-sm font-medium text-gray-700">Stiffness</label>
                                                    <input
                                                        type="number"
                                                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                        value={advancedOptions.stiffness}
                                                        onChange={(e) => setAdvancedOptions({
                                                            ...advancedOptions,
                                                            stiffness: parseInt(e.target.value)
                                                        })}
                                                    />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                <div className="space-y-4">
                                    <h3 className="text-md font-medium text-gray-700">TEP Analysis</h3>

                                    <div>
                                        <label className="block text-sm font-medium text-gray-700">Analysis Type</label>
                                        <select
                                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                            value={advancedOptions.tepAnalysisType}  // Changed from tep_analysis_type
                                            onChange={(e) => setAdvancedOptions({
                                                ...advancedOptions,
                                                tepAnalysisType: e.target.value  // Changed from tep_analysis_type
                                            })}
                                        >
                                            <option value="gmfa">Global Mean Field Amplitude (GMFA)</option>
                                            <option value="roi">Region of Interest (ROI)</option>
                                        </select>
                                    </div>

                                    {advancedOptions.tepAnalysisType === 'roi' && (
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">ROI
                                                Channels</label>
                                            <div className="mt-2 space-y-4">
                                                {/* Channel Selection */}
                                                <div className="grid grid-cols-4 gap-2">
                                                    {['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                                                        'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3',
                                                        'Pz', 'P4', 'T6', 'O1', 'O2'].map((channel) => {
                                                        // Check if the channel is in the selected channels array
                                                        const isSelected = advancedOptions.tepRoiChannels?.includes(channel) || false;

                                                        return (
                                                            <button
                                                                key={channel}
                                                                onClick={() => {
                                                                    const currentChannels = [...(advancedOptions.tepRoiChannels || [])];
                                                                    const newChannels = isSelected
                                                                        ? currentChannels.filter(ch => ch !== channel)
                                                                        : [...currentChannels, channel];

                                                                    setAdvancedOptions(prev => ({
                                                                        ...prev,
                                                                        tepRoiChannels: newChannels
                                                                    }));
                                                                }}
                                                                className={`px-3 py-2 text-sm rounded-md transition-colors ${
                                                                    isSelected
                                                                        ? 'bg-blue-600 text-white'
                                                                        : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                                                                }`}
                                                            >
                                                                {channel}
                                                            </button>
                                                        );
                                                    })}
                                                </div>

                                                {/* Selected Channels Display */}
                                                <div className="bg-gray-50 p-3 rounded-md">
                                                    <div className="text-sm font-medium text-gray-700 mb-2">Selected
                                                        Channels:
                                                    </div>
                                                    <div className="flex flex-wrap gap-2">
                                                        {(advancedOptions.tepRoiChannels || []).map((channel) => (
                                                            <div
                                                                key={channel}
                                                                className="flex items-center bg-blue-100 text-blue-800 px-2 py-1 rounded-md"
                                                            >
                                                                <span className="text-sm">{channel}</span>
                                                                <button
                                                                    onClick={() => {
                                                                        setAdvancedOptions(prev => ({
                                                                            ...prev,
                                                                            tepRoiChannels: prev.tepRoiChannels.filter(ch => ch !== channel)
                                                                        }));
                                                                    }}
                                                                    className="ml-2 text-blue-600 hover:text-blue-800"
                                                                >
                                                                    ×
                                                                </button>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">TEP
                                                Method</label>
                                            <select
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.tep_method}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    tep_method: e.target.value
                                                })}
                                            >
                                                <option value="largest">Largest</option>
                                                <option value="mean">Mean</option>
                                                <option value="median">Median</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">TEP
                                                Samples</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.tep_samples}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    tep_samples: parseInt(e.target.value)
                                                })}
                                            />
                                        </div>
                                    </div>

                                    <div className="space-y-3">
                                        <div className="flex items-center space-x-3">
                                            <input
                                                type="checkbox"
                                                id="save-validation"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={advancedOptions.save_validation}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    save_validation: e.target.checked
                                                })}
                                            />
                                            <label htmlFor="save-validation"
                                                   className="text-sm font-medium text-gray-700">
                                                Save Validation Results
                                            </label>
                                        </div>

                                        <div className="flex items-center space-x-3">
                                            <input
                                                type="checkbox"
                                                id="save-evoked"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={advancedOptions.save_evoked}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    save_evoked: e.target.checked
                                                })}
                                            />
                                            <label htmlFor="save-evoked" className="text-sm font-medium text-gray-700">
                                                Save Evoked Response
                                            </label>
                                        </div>
                                    </div>
                                </div>

                                {/* PCIst Parameters */}
                                <div className="space-y-4">
                                    <h3 className="text-md font-medium text-gray-700">PCIst Parameters</h3>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">k Value</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.k}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    k: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Maximum Variance (%)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.max_var}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    max_var: parseFloat(e.target.value)
                                                })}
                                                step="0.1"
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Number of Steps</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.n_steps}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    n_steps: parseInt(e.target.value)
                                                })}
                                            />
                                        </div>
                                        <div className="flex items-center space-x-3">
                                            <input
                                                type="checkbox"
                                                id="embed"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={advancedOptions.embed}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    embed: e.target.checked
                                                })}
                                            />
                                            <label htmlFor="embed" className="text-sm font-medium text-gray-700">
                                                Enable Embedding
                                            </label>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Pre-window Start (ms)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.pre_window_start}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    pre_window_start: parseInt(e.target.value)
                                                })}
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Pre-window End (ms)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.pre_window_end}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    pre_window_end: parseInt(e.target.value)
                                                })}
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Post-window Start (ms)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.post_window_start}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    post_window_start: parseInt(e.target.value)
                                                })}
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Post-window End (ms)</label>
                                            <input
                                                type="number"
                                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                                                value={advancedOptions.post_window_end}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    post_window_end: parseInt(e.target.value)
                                                })}
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Output Options */}
                                <div className="space-y-4">
                                    <h3 className="text-md font-medium text-gray-700">Output Options</h3>

                                    <div className="space-y-3">
                                        <div className="flex items-center space-x-3">
                                            <input
                                                type="checkbox"
                                                id="plot-raw"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={advancedOptions.plotRaw}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    plotRaw: e.target.checked
                                                })}
                                            />
                                            <label htmlFor="plot-raw" className="text-sm font-medium text-gray-700">
                                                Plot Raw Data
                                            </label>
                                        </div>

                                        <div className="flex items-center space-x-3">
                                            <input
                                                type="checkbox"
                                                id="research"
                                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                checked={advancedOptions.research}
                                                onChange={(e) => setAdvancedOptions({
                                                    ...advancedOptions,
                                                    research: e.target.checked
                                                })}
                                            />
                                            <label htmlFor="research" className="text-sm font-medium text-gray-700">
                                                Enable Research Mode
                                            </label>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Tab.Panel>

                      {/* Processing Panel */}
                      <Tab.Panel className="bg-white rounded-xl p-6 shadow">
                        <div className="space-y-6">
                          <h2 className="text-lg font-medium">Processing Status</h2>

                            <div className="space-y-4">
                                {/* Progress Bar */}
                                <div className="w-full bg-gray-200 rounded-full h-2.5">
                                    <div
                                        className="bg-blue-600 h-2.5 rounded-full transition-all duration-500"
                                        style={{width: `${progress}%`}}
                                    ></div>
                                </div>

                                {/* Status Message */}
                                <div className="text-sm text-gray-600">
                                    {statusMessage}
                                </div>

                                {/* Processing Controls */}
                                <div className="flex space-x-4">
                                    <button
                                        onClick={handleStartProcessing}
                                        disabled={isProcessing}
                                        className={`px-4 py-2 rounded-md text-white font-medium ${
                                            isProcessing
                                                ? 'bg-gray-400 cursor-not-allowed'
                                                : 'bg-blue-600 hover:bg-blue-700'
                                        }`}
                                    >
                                        Start Processing
                                    </button>

                                    <button
                                        onClick={handleStopProcessing}
                                        disabled={!isProcessing}
                                        className={`px-4 py-2 rounded-md text-white font-medium ${
                                            !isProcessing
                                                ? 'bg-gray-400 cursor-not-allowed'
                                                : 'bg-red-600 hover:bg-red-700'
                                        }`}
                                    >
                                        Stop Processing
                                    </button>
                                </div>
                                {/* Log Output */}
                                <div className="mt-6">
                                    <h3 className="text-md font-medium text-gray-700 mb-2">Processing Log</h3>
                                    <div className="relative">
                                        <div className="console-container">
                                            <div className="console-header">
                                                <span className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
                                                    {isConnected ? 'Connected' : 'Disconnected'}
                                                </span>
                                            </div>
                                            <div
                                                ref={consoleRef}
                                                className="console-output"
                                                id="console-output"
                                            >
                                                          {processingLogs.map((log, i) => {
                                            let timeString;
                                            if (log.timestamp instanceof Date) {
                                                timeString = log.timestamp.toLocaleTimeString();
                                            }
                                            else if (typeof log.timestamp === 'string' || typeof log.timestamp === 'number') {
                                                timeString = new Date(log.timestamp).toLocaleTimeString();
                                            }
                                            else {
                                                timeString = new Date().toLocaleTimeString();
                                            }

                                            return (
                                                <div key={i} className={`console-line ${log.type}`}>
                                                    [{timeString}] {log.message}
                                                </div>
                                            );
                                        })}

                                            </div>
                                        </div>
                                        <div className="absolute top-2 right-2 space-x-2">
                                            <button
                                                onClick={handleClearLog}
                                                className="px-2 py-1 text-xs text-gray-200 hover:text-white bg-gray-700 hover:bg-gray-600 rounded"
                                            >
                                                Clear Log (refresh)
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                {/* Error Display */}
                                {error && (
                                    <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
                                        <div className="flex">
                                            <AlertTriangle className="h-5 w-5 text-red-400 mr-2"/>
                                            <div className="text-sm text-red-700">{error}</div>
                                        </div>
                                    </div>
                                )}
                                {/* ICA selection */}
                                {icaSelectionStatus.isSelecting && (
                                    <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-md">
                                        <div className="flex items-center">
                                            <Brain className="h-5 w-5 text-blue-400 mr-2" />
                                            <div className="text-sm text-blue-700">
                                                ICA component selection in progress...
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {icaSelectionStatus.selectedComponents.length > 0 && (
                                    <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-md">
                                        <div className="flex items-center">
                                            <Check className="h-5 w-5 text-green-400 mr-2" />
                                            <div className="text-sm text-green-700">
                                                Selected ICA components: {icaSelectionStatus.selectedComponents.join(', ')}
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Results Section */}
                                {processingComplete && (
                                    <div className="mt-6 space-y-4">
                                        <h3 className="text-md font-medium text-gray-700">Processing Results</h3>

                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="p-4 bg-gray-50 rounded-md">
                                                <h4 className="text-sm font-medium text-gray-600 mb-2">Processing
                                                    Time</h4>
                                                <p className="text-lg font-medium text-gray-900">{processingTime}</p>
                                            </div>

                                            <div className="p-4 bg-gray-50 rounded-md">
                                                <h4 className="text-sm font-medium text-gray-600 mb-2">Files
                                                    Processed</h4>
                                                <p className="text-lg font-medium text-gray-900">{filesProcessed}</p>
                                            </div>
                                        </div>

                                        {/* Results Summary */}
                                        <div className="p-4 bg-gray-50 rounded-md">
                                            <h4 className="text-sm font-medium text-gray-600 mb-2">Summary</h4>
                                            <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
                                                {resultsSummary.map((item, index) => (
                                                    <li key={index}>{item}</li>
                                                ))}
                                            </ul>
                                        </div>

                                        {/* Download Results Button */}
                                        <button
                                            onClick={handleDownloadResults}
                                            className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-medium rounded-md"
                                        >
                                            Download Results
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                      </Tab.Panel>
                    </Tab.Panels>
                </Tab.Group>
            </main>
            {/* Add ICA Modal here */}
            {showICAModal && (
              <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
                <ErrorBoundary>
                  <ICAComponentVisualizer
                    socket={socket}
                    onClose={() => {
                      setShowICAModal(false);
                      setIcaComponentData([]);
                      setSelectedICAComponents([]);
                    }}
                    onSelectionComplete={(selected) => {
                      setSelectedICAComponents(selected);
                      setShowICAModal(false);
                    }}
                  />
                </ErrorBoundary>
              </div>
            )}
        </div>
        )}
        </>
    );
}

export default TmseegpyGUI;
