""" Main bootstrap and configuration module for pyccd.  Any module that
requires configuration or services should import app and obtain the
configuration or service from here.
app.py enables a very basic but sufficient form of loose coupling
by setting names of services & configuration once and allowing other modules
that require these services/information to obtain them by name rather than
directly importing or instantiating.
Module level constructs are only evaluated once in a Python application's
lifecycle, usually at the time of first import. This pattern is borrowed
from Flask.
"""
import os

import yaml

import numpy as np


__paramfile_path = os.path.join(os.path.dirname(__file__), 'parameters.yaml')


def get_params(yaml_file=__paramfile_path):
    """
    Generates a Parameters object from a given yaml file path.

    Args:
        yaml_file: path to a yaml formatted file

    Returns:
        Parameters object
    """
    with open(yaml_file) as f:
        return yaml.load(f)


def gen_rng():
    """
    Helper method to create a new random number generator object.

    See:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html
        https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.random.set_state.html

    Returns:
        Numpy random number generating object
    """
    return np.random.RandomState()


def sorted_list(sort_dict):
    """
        Helper method to take a dictionary, and create a list of the values
        ordered on the key values.

        This exists because we cannot guarantee the order in which values in
        a yaml list stay ordered when going through the python transformation.

        Args:
            sort_dict: dictionary

        Returns:
            list
        """
    return [sort_dict[k] for k in sorted(sort_dict.keys())]
