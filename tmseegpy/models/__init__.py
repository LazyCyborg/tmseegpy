import os
import importlib.resources as pkg_resources


def get_model_path(filename):
    """
    Get the absolute path to a model file

    Parameters
    ----------
    filename : str
        Name of the file (e.g., 'model.keras' or 'encoder.pkl')

    Returns
    -------
    str
        Absolute path to the file
    """
    # Get the directory where this __init__.py file is located
    models_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(models_dir, filename)