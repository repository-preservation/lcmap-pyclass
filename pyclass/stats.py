import numpy as np

from pyclass.app import logging, defaults

log = logging.getLogger(__name__)


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
        float: cloud probability
        float: snow probability
        float: water probability

    """
    nofill = quality[quality != fill]
    clear_water_vw = nofill[quality == clear | quality == water]
    cloud_vw = nofill[quality == cloud]
    water_vw = nofill[quality == water]
    snow_vw = nofill[quality == snow]

    total_count = np.sum(nofill, axis=1)
    clear_water_count = np.sum(clear_water_vw, axis=1)
    cloud_count = np.sum(cloud_vw, axis=1)
    snow_count = np.sum(water_vw, axis=1)
    water_count = np.sum(snow_vw, axis=1)

    cloud_prob = cloud_count / total_count * 100
    snow_prob = snow_count / (clear_water_count + snow_count + 0.01) * 100
    water_prob = water_count / (clear_water_count + 0.01) * 100

    return cloud_prob, snow_prob, water_prob
