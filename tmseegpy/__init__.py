"""TMS-EEG Analysis Package"""

# tmseegpy/__init__.py

from .ica_selector_gui.ica_selector import (
    ICAComponentSelector,
    ICAComponentSelectorContinuous
)
from .ica_selector_gui.websocket_ica_selector import WebSocketICASelector
from .analyze import *
from .clean import *
from .pcist import *
from .preproc import *
from .preproc_vis import *
from .run import *
from .validate_tep import *
from .dataloader import *
from .cli_ica_selector import CLIICASelector, get_cli_ica_callback
from .server.server import run_server
from .cli import main as cli_main

__version__ = "0.1.8"

__all__ = [
    'cli_main',
    'run_server',
    'ICAComponentSelector',
    'ICAComponentSelectorContinuous',
    'CLIICASelector',
    'get_cli_ica_callback',
    'WebSocketICASelector',
]

# Define entry points for command line tools
def run_cli():
    """Entry point for the tmseegpy command line tool"""
    cli_main()

def run_server_cmd():
    """Entry point for the tmseegpy-server command"""
    run_server()