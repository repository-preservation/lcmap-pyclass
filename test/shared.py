import numpy as np


def read_ccdresults(path):
    """
    Helper method to load sample CCD results from a npy file.

    Args:
        path: file path to load
    Returns:
        ndarray of dict objects
    """
    return np.load(path)
