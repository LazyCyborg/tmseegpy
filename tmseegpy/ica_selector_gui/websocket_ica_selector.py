# tmseegpy/ica_selector_gui/websocket_ica_selector.py
from queue import Queue, Empty
import traceback
import numpy as np
import mne


# In websocket_ica_selector.py

class WebSocketICASelector:
    def __init__(self, selection_queue, result_queue):
        self.selection_queue = selection_queue
        self.result_queue = result_queue
        self.timeout = 300
        self.cached_component_data = None
        self.max_retries = 3

    def _validate_json_serializable(self, data):
        """
        Validate that data is JSON serializable with detailed error reporting.

        Parameters
        ----------
        data : any
            The data structure to validate

        Raises
        ------
        ValueError
            If the data cannot be serialized to JSON
        """
        try:
            import json
            # Try to serialize the entire structure
            json.dumps(data)
        except (TypeError, OverflowError) as e:
            print("JSON serialization validation failed!")

            if isinstance(data, dict):
                # If it's a dictionary, check each key-value pair
                problematic_keys = []
                for key, value in data.items():
                    try:
                        json.dumps(value)
                    except (TypeError, OverflowError):
                        print(f"Non-serializable value for key '{key}': {type(value)}")
                        problematic_keys.append(key)

                if problematic_keys:
                    error_msg = f"Non-serializable values found in keys: {', '.join(problematic_keys)}"
                else:
                    error_msg = str(e)
            else:
                error_msg = f"Data of type {type(data)} is not JSON serializable: {str(e)}"

            raise ValueError(error_msg)

    def select_components(self, ica_instance, inst, component_scores=None):
        try:
            print("Starting ICA component selection...")

            # Generate component data
            self.cached_component_data = self._get_component_data(ica_instance, inst)

            # Create the payload
            payload = {
                "data": self.cached_component_data,
                "scores": self._format_scores(component_scores) if component_scores else {}
            }

            # Clear and update selection queue
            self._clear_queues()
            self.selection_queue.put(payload)

            # Register event handler for selection
            from ..server.server import socketio

            @socketio.on('ica_selection_complete')
            def handle_ica_complete(data):
                try:
                    print("Received ICA selection complete", data)
                    selected = data.get('selectedComponents', [])
                    print(f"Selected components: {selected}")
                    self.result_queue.put(selected)
                    socketio.emit('ica_selection_success', {'status': 'success'})
                except Exception as e:
                    print(f"Error handling selection: {str(e)}")
                    socketio.emit('ica_error', {'message': str(e)})

            @socketio.on('request_ica_data')
            def handle_data_request():
                try:
                    #print("Received ICA data request")
                    if not self.selection_queue.empty():
                        data = self.selection_queue.queue[0]  # Peek without removing
                        socketio.emit('ica_data', data)
                    else:
                        socketio.emit('ica_error', {'message': 'No ICA data available'})
                except Exception as e:
                    print(f"Error sending ICA data: {str(e)}")
                    socketio.emit('ica_error', {'message': str(e)})

            # Trigger the ICA component selection UI
            print("Emitting ica_required event...")
            socketio.emit('ica_required', {
                'componentCount': len(self.cached_component_data)
            })

            # Wait for selection result
            try:
                print("Waiting for component selection...")
                selected_components = self.result_queue.get(timeout=self.timeout)
                print(f"Received selected components: {selected_components}")
                return selected_components
            except Empty:
                print("Selection timed out")
                return []

        except Exception as e:
            print(f"Error in select_components: {str(e)}")
            traceback.print_exc()
            return []

    def _format_scores(self, scores):
        if not scores:
            return {}

        formatted_scores = {}
        try:
            for score_type, scores_array in scores.items():
                if not isinstance(scores_array, (list, np.ndarray)):
                    print(f"Warning: Invalid score format for {score_type}")
                    continue

                # Convert to Python native types for JSON serialization
                formatted_scores[str(score_type)] = [
                    float(score) if not np.isnan(score) else 0.0
                    for score in scores_array
                ]

            # Validate the formatted scores
            self._validate_json_serializable(formatted_scores)
            return formatted_scores

        except Exception as e:
            print(f"Error formatting scores: {str(e)}")
            return {}

    def _get_component_data(self, ica_instance, inst):
        try:
            if isinstance(inst, mne.Epochs):
                sources = ica_instance.get_sources(inst)
                data = sources.get_data()
                mean_data = np.mean(data, axis=0)
                std_data = np.std(data, axis=0)
                times = inst.times
                print(f"Processing epochs data: {data.shape}")

            else:  # Raw data
                sources = ica_instance.get_sources(inst)
                data = sources.get_data()
                times = np.arange(data.shape[1]) / inst.info['sfreq']
                print(f"Processing raw data: {data.shape}")

                # Downsample if needed
                max_points = 1000
                if len(times) > max_points:
                    step = len(times) // max_points
                    data = data[:, ::step]
                    times = times[::step]
                    print(f"Downsampled to {len(times)} points")

                mean_data = data
                std_data = None

            # Format component data as a list of components
            component_data = []
            for comp_idx in range(mean_data.shape[0]):
                # Create the data structure for each component
                component = {
                    "data": [
                        {
                            'time': float(t),
                            'value': float(mean_data[comp_idx, t_idx])
                        }
                        for t_idx, t in enumerate(times)
                    ]
                }

                # Add variance if available
                if std_data is not None:
                    for t_idx, point in enumerate(component["data"]):
                        point['variance'] = float(std_data[comp_idx, t_idx])

                component_data.append(component)

            # Validate the entire component data structure
            self._validate_json_serializable(component_data)
            print(f"Successfully formatted and validated {len(component_data)} components")

            # Print sample of the first component's data for debugging
            if component_data:
                print("Sample of first component data:")
                print(component_data[0]["data"][:5])

            return component_data

        except Exception as e:
            print(f"Error in _get_component_data: {str(e)}")
            traceback.print_exc()
            raise

    def _clear_queues(self):
        """Clear both selection and result queues"""
        try:
            while not self.selection_queue.empty():
                self.selection_queue.get_nowait()
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
        except Exception as e:
            print(f"Error clearing queues: {str(e)}")