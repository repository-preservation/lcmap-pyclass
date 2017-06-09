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


# Simple container object to hold parameters.
class Parameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        for kw, item in kwargs.items():
            if isinstance(item, dict):
                setattr(self, kw, Parameters(**kwargs[kw]))

    def __repr__(self):
        return repr(self.__dict__)


def get_params(yaml_file=__paramfile_path):
    """
    Generates a Parameters object from a given yaml file path.

    Args:
        yaml_file: path to a yaml formatted file

    Returns:
        Parameters object
    """
    with open(yaml_file) as f:
        return Parameters(**yaml.load(f))


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
