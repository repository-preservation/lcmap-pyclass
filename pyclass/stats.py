import numpy as np

from pyclass.app import logging, defaults, ensure_ndarray_input

log = logging.getLogger(__name__)


@ensure_ndarray_input
def quality_stats(quality, clear=defaults.QA_CLEAR, water=defaults.QA_WATER,
                  snow=defaults.QA_SNOW, cloud=defaults.QA_CLOUD,
                  fill=defaults.QA_FILL):
    """
    Take the initial quality information for the entire time series, then
    calculate the probabilities for cloud, snow, and water. These outputs are
    used for input parameters to both training and predicting.

    Args:
        quality: QA values from all available observations
        clear: int value representing clear
        water: int value representing water
        snow: int value representing snow
        cloud: int value representing cloud
        fill: int value representing fill/nodata

    Returns:
        1-d ndarray: cloud probability
        1-d ndarray: snow probability
        1-d ndarray: water probability

    """
    total_count = np.sum(quality != fill, axis=1)
    clear_water_count = np.sum((quality == clear) | (quality == water), axis=1)
    cloud_count = np.sum(quality == cloud, axis=1)
    snow_count = np.sum(quality == snow, axis=1)
    water_count = np.sum(quality == water, axis=1)

    cloud_prob = cloud_count / total_count * 100
    snow_prob = snow_count / (clear_water_count + snow_count + 0.01) * 100
    water_prob = water_count / (clear_water_count + 0.01) * 100

    return cloud_prob, snow_prob, water_prob
