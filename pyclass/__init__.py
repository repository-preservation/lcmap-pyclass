"""
Main lead in methods
"""
import numpy as np

from pyclass import app, stats
log = app.logging.getLogger(__name__)


def __gen_rng():
    """
    Helper method to create a new random number generator object.

    See:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html
        https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.random.set_state.html

    Returns:
        Numpy random number generating object
    """
    return np.random.RandomState()


def train(trends, coefs, rmse, dem, aspect, slope, posidex, mpw, qa, random_state=None):
    """
    Main module entry point for training a new classification model.

    Implementation of the

    Args:
        trends:
        coefs:
        rmse:
        dem:
        aspect:
        slope:
        posidex:
        mpw:
        qa:
        random_state:

    Returns:

    """
    cloud_prob, snow_prob, water_prob = stats.quality_stats(qa)


def classify(coefs, rmse, dem, aspect, slope, posidex, mpw, qa):
    cloud_prob, snow_prob, water_prob = stats.quality_stats(qa)
