"""TMS-EEG Analysis Package"""

from .main import main
from .ica_selector_gui.ica_selector import (
    ICAComponentSelector,
    ICAComponentSelectorContinuous
)
from .analyze import *
from .clean import *
from .pcist import *
from .preproc import *
from .preproc_vis import *
from .run import *
from .validate_tep import *
from .dataloader import *
from .cli_ica_selector import CLIICASelector, get_cli_ica_callback


__version__ = "0.1.8"

__all__ = [
    'main',
    'ICAComponentSelector',
    'ICAComponentSelectorContinuous',
    'CLIICASelector',
    'get_cli_ica_callback'
]