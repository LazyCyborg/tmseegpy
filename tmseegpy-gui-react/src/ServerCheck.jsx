import React, { useState, useEffect } from 'react';
import { api } from './services/api';

const ServerCheck = ({ onServerReady }) => {
  const [serverStatus, setServerStatus] = useState('checking');
  const [error, setError] = useState(null);

  useEffect(() => {
    checkServer();
  }, []);

const checkServer = async () => {
  try {
    const response = await api.test();
    if (response.status === 200) {
      // Make sure we have a valid server response
      if (response.data && response.data.status === 'success') {
        setServerStatus('connected');
        // Ensure we wait for the socket connection to be ready
        setTimeout(() => {
          onServerReady();
        }, 100); // Small delay to ensure socket setup
      } else {
        throw new Error('Invalid server response');
      }
    } else {
      throw new Error('Server returned an unexpected status');
    }
  } catch (err) {
    console.error('Server check failed:', err);
    setServerStatus('error');
    setError('TMSeegpy server not found. Please ensure TMSeegpy is running.');
  }
};

  if (serverStatus === 'checking') {
    return (
      <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center">
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <p className="text-lg">Checking for TMSeegpy server...</p>
          <div className="mt-4 animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
        </div>
      </div>
    );
  }

  if (serverStatus === 'error') {
    return (
      <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center">
        <div className="bg-white p-6 rounded-lg shadow-lg max-w-md">
          <h3 className="text-xl font-bold text-red-600 mb-4">Connection Error</h3>
          <p className="text-gray-700 mb-4">{error}</p>
          <p className="text-sm text-gray-600 mb-4">
            Please make sure you have installed and started the TMSeegpy package.
            You can start it by running:
          </p>
          <div className="bg-gray-100 p-2 rounded">
            <code>tmseegpy server</code>
          </div>
          <button
            onClick={checkServer}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return null;
};

export default ServerCheck;