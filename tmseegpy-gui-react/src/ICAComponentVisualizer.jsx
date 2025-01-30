import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card';
import { Button } from './components/ui/button';
import { AlertCircle, Loader2 } from 'lucide-react';

const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-white p-3 rounded-lg shadow-lg border">
                <p className="font-medium">Time: {Number(label).toFixed(3)}s</p>
                <p>Amplitude: {payload[0].value.toFixed(3)}µV</p>
                {payload[0].payload.scores && Object.entries(payload[0].payload.scores).map(([key, value]) => (
                    value > 0 && (
                        <p key={key} className="text-xs">
                            {key}: {value.toFixed(2)}
                        </p>
                    )
                ))}
            </div>
        );
    }
    return null;
};

const ICAComponentVisualizer = ({ socket, onClose, onSelectionComplete }) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [componentData, setComponentData] = useState([]);
    const [selectedComponents, setSelectedComponents] = useState([]);
    const [layout, setLayout] = useState({ rows: 1, cols: 1 });

    const handleComponentClick = useCallback((index) => {
        setSelectedComponents(prev =>
            prev.includes(index)
                ? prev.filter(i => i !== index)
                : [...prev, index]
        );
    }, []);

    const handleFinishSelection = useCallback(() => {
        if (!socket?.connected) {
            setError('Lost connection to server');
            return;
        }

        console.log('Sending selected components:', selectedComponents);
        socket.emit('ica_selection_complete', {
            selectedComponents
        });
    }, [socket, selectedComponents]);

    useEffect(() => {
        if (!socket) {
            setError('No socket connection available');
            return;
        }

        const handleIcaData = (payload) => {
            try {
                if (!payload?.data || !Array.isArray(payload.data)) {
                    throw new Error('Invalid component data format');
                }

                const processedData = payload.data.map((comp, idx) => ({
                    data: comp.data.map(point => ({
                        time: Number(point.time),
                        value: Number(point.value),
                        scores: comp.scores
                    })),
                    scores: comp.scores
                }));

                setComponentData(processedData);
                setLoading(false);

                const sqrt = Math.sqrt(processedData.length);
                const rows = Math.ceil(sqrt);
                const cols = Math.ceil(processedData.length / rows);
                setLayout({ rows, cols });
            } catch (err) {
                setError(`Error processing component data: ${err.message}`);
                setLoading(false);
            }
        };

        const handleIcaError = (error) => {
            setError(error.message || 'Failed to load ICA components');
            setLoading(false);
        };

        const handleSelectionSuccess = (response) => {
            console.log('Selection successful:', response);
            onClose?.();
            onSelectionComplete?.(selectedComponents);
        };

        const handleSelectionError = (error) => {
            console.error('Selection error:', error);
            setError(error.message || 'Failed to confirm selection');
        };

        // Set up event listeners
        socket.on('ica_data', handleIcaData);
        socket.on('ica_error', handleIcaError);
        socket.on('ica_selection_success', handleSelectionSuccess);
        socket.on('ica_selection_error', handleSelectionError);

        // Request initial data
        socket.emit('request_ica_data');

        // Cleanup
        return () => {
            socket.off('ica_data', handleIcaData);
            socket.off('ica_error', handleIcaError);
            socket.off('ica_selection_success', handleSelectionSuccess);
            socket.off('ica_selection_error', handleSelectionError);
        };
    }, [socket, selectedComponents, onClose, onSelectionComplete]);

    return (
        <Card className="relative max-h-[90vh] overflow-y-auto bg-white shadow-lg border-0">
            <CardHeader className="bg-gray-50/80 sticky top-0 z-10 border-b">
                <CardTitle className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <span>ICA Component Selection</span>
                        {selectedComponents.length > 0 && (
                            <span className="text-sm text-muted-foreground">
                                ({selectedComponents.length} selected)
                            </span>
                        )}
                    </div>
                    <div className="flex gap-2">
                        <Button
                            onClick={handleFinishSelection}
                            disabled={loading || selectedComponents.length === 0}
                            className="bg-blue-600 hover:bg-blue-700 text-white"
                        >
                            Confirm Selection ({selectedComponents.length})
                        </Button>
                        <Button
                            variant="outline"
                            onClick={onClose}
                            className="hover:bg-gray-100"
                        >
                            Cancel
                        </Button>
                    </div>
                </CardTitle>
            </CardHeader>

            <CardContent className="p-6">
                {loading && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-white bg-opacity-90">
                        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
                        <p className="mt-2 text-gray-600">Loading components...</p>
                    </div>
                )}

                {error && (
                    <div className="flex items-center gap-2 p-4 mb-4 text-red-600 bg-red-50 rounded-lg border border-red-200">
                        <AlertCircle className="h-5 w-5 flex-shrink-0" />
                        <p className="text-sm font-medium">{error}</p>
                    </div>
                )}

                {componentData.length > 0 && (
                    <div
                        className="grid gap-6"
                        style={{
                            gridTemplateColumns: `repeat(${layout.cols}, minmax(0, 1fr))`
                        }}
                    >
                        {componentData.map((component, index) => (
                            <div
                                key={index}
                                className={`relative cursor-pointer rounded-xl border p-4 transition-all
                                    ${selectedComponents.includes(index) 
                                        ? 'border-blue-600 bg-blue-50 shadow-md' 
                                        : 'hover:border-blue-300 bg-white hover:bg-gray-50'}`}
                                onClick={() => handleComponentClick(index)}
                            >
                                <div className="absolute top-2 right-2 flex flex-wrap gap-1 max-w-[70%] justify-end">
                                    {Object.entries(component.scores || {})
                                        .filter(([_, value]) => value > 0)
                                        .map(([key, value]) => (
                                            <span
                                                key={key}
                                                className="rounded px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800"
                                                title={`${key}: ${value.toFixed(2)}`}
                                            >
                                                {key}
                                            </span>
                                        ))}
                                </div>

                                <ResponsiveContainer width="100%" height={150}>
                                    <LineChart
                                        data={component.data}
                                        margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                                        <XAxis
                                            dataKey="time"
                                            type="number"
                                            domain={['dataMin', 'dataMax']}
                                            tickFormatter={(t) => t.toFixed(2)}
                                            fontSize={10}
                                            stroke="#666"
                                        />
                                        <YAxis
                                            width={30}
                                            domain={['dataMin', 'dataMax']}
                                            tickFormatter={(v) => v.toFixed(1)}
                                            fontSize={10}
                                            stroke="#666"
                                        />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Line
                                            type="monotone"
                                            dataKey="value"
                                            stroke="#2563eb"
                                            strokeWidth={1.5}
                                            dot={false}
                                            isAnimationActive={false}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>

                                <div className="mt-2 text-center">
                                    <span className="text-sm font-medium text-gray-700">
                                        Component {index + 1}
                                    </span>
                                    {selectedComponents.includes(index) && (
                                        <span className="ml-2 text-xs text-blue-600">
                                            (Selected)
                                        </span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </CardContent>
        </Card>
    );
};

export default ICAComponentVisualizer;