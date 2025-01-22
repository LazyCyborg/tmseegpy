// api.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5001';

const axiosInstance = axios.create({
    baseURL: API_BASE_URL,
    timeout: 300000  // 5 minutes timeout
});

// Response interceptor for logging
axiosInstance.interceptors.response.use(
    (response) => {
        console.log(`API Response [${response.config.url}]:`, response.data);
        return response;
    },
    (error) => {
        console.error('API Error:', {
            url: error.config?.url,
            method: error.config?.method,
            status: error.response?.status,
            data: error.response?.data,
            message: error.message
        });
        return Promise.reject(error);
    }
);

// Request interceptor for logging
axiosInstance.interceptors.request.use(
    (config) => {
        console.log(`API Request [${config.method.toUpperCase()}] ${config.url}`);
        return config;
    },
    (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
    }
);

export const api = {
    upload: async (formData) => {
        try {
            const response = await axiosInstance.post('/api/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            return response;
        } catch (error) {
            console.error('Upload error:', error);
            throw error;
        }
    },

    testDataLoading: async (params) => {
        try {
            const response = await axiosInstance.post('/api/test_data_loading', params);
            return response;
        } catch (error) {
            console.error('Test data loading error:', error);
            throw error;
        }
    },

    process: async (options) => {
        try {
            console.log('Sending process request with options:', options);
            const response = await axiosInstance.post('/api/process', options);
            console.log('Process response:', response);
            return response;
        } catch (error) {
            console.error('Processing error:', {
                message: error.message,
                response: error.response?.data,
                status: error.response?.status
            });
            throw error;
        }
    },

    stop: async () => {
        try {
            const response = await axiosInstance.post('/api/stop');
            return response;
        } catch (error) {
            console.error('Stop processing error:', error);
            throw error;
        }
    },

    getStatus: async () => {
        try {
            const response = await axiosInstance.get('/api/status');
            return response;
        } catch (error) {
            console.error('Get status error:', error);
            throw error;
        }
    },

    getResults: async () => {
        try {
            const response = await axiosInstance.get('/api/results');
            return response;
        } catch (error) {
            console.error('Get results error:', error);
            throw error;
        }
    }
};