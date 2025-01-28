from tmseegpy.cli_ica_selector import CLIICASelector
import os
import sys
from multiprocessing import Queue
import json
import threading


def run_ica_selection(ica_obj, inst, component_scores, status_queue=None):
    """Run ICA selection using the existing CLIICASelector with status updates"""
    try:
        def send_status(message):
            if status_queue:
                status_queue.put({
                    'type': 'info',
                    'message': message
                })
            print(message)  # Still print to console for debugging

        send_status("Starting ICA selection...")
        send_status(f"Current working directory: {os.getcwd()}")
        send_status(f"PYTHONPATH: {sys.path}")
        send_status(f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM')}")
        send_status(f"DISPLAY: {os.environ.get('DISPLAY')}")

        # Try setting platform to cocoa for macOS
        if sys.platform == 'darwin':
            os.environ['QT_QPA_PLATFORM'] = 'cocoa'
            send_status("Set QT_QPA_PLATFORM to cocoa for macOS")

        selector = CLIICASelector()
        send_status("Created ICA selector")

        # Add information about components
        if component_scores is not None:
            send_status(f"Number of components to analyze: {len(component_scores)}")

        send_status("Starting component selection...")
        result = selector.select_components(ica_obj, inst, component_scores)

        send_status(f"Component selection completed. Selected components: {result}")
        return result

    except Exception as e:
        error_msg = f"Error in ICA selection: {str(e)}"
        send_status(error_msg)
        import traceback
        tb = traceback.format_exc()
        send_status(f"Traceback:\n{tb}")
        return []


def process_target(conn, ica_obj, inst, scores):
    """Function that runs in separate process with enhanced status reporting"""
    status_queue = Queue()

    # Start a thread to forward status messages
    def forward_status():
        while True:
            try:
                status = status_queue.get()
                if status is None:  # Sentinel value
                    break
                conn.send(json.dumps({
                    'type': 'status',
                    'data': status
                }))
            except EOFError:
                break
            except Exception as e:
                print(f"Error forwarding status: {e}")
                break

    status_thread = threading.Thread(target=forward_status)
    status_thread.start()

    try:
        print("Starting process_target...")
        result = run_ica_selection(ica_obj, inst, scores, status_queue)
        print(f"Selection result: {result}")

        # Send the actual result
        conn.send(json.dumps({
            'type': 'result',
            'data': result
        }))

    except Exception as e:
        error_msg = f"Error in process_target: {str(e)}"
        print(error_msg)
        import traceback
        tb = traceback.format_exc()
        print(tb)

        # Send error information
        conn.send(json.dumps({
            'type': 'error',
            'data': {
                'error': str(e),
                'traceback': tb
            }
        }))
    finally:
        # Signal status thread to stop
        status_queue.put(None)
        status_thread.join()
        conn.close()