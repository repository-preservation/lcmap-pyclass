"""
Filters related to qa values associated with samples.
"""
import numpy as np

from pyclass.app import defaults


def unpackqa(quality, proc_params):
    """
    Transform the bit-packed QA values into their bit offset.

    Institute a hierarchy of qa values that may be flagged in the bitpacked
    value.

    fill > cloud > shadow > snow > water > clear

    Args:
        quality: 1-d array or list of bit-packed QA values
        proc_params: dictionary of processing parameters

    Returns:
        1-d ndarray
    """
    output = np.ones_like(quality)
    output.fill(proc_params.QA_NAN)

    # Reverse hierarchy, values later in the list will overwrite earlier values
    heirarchy = (proc_params.QA_CLEAR,
                 proc_params.QA_WATER,
                 proc_params.QA_SNOW,
                 proc_params.QA_SHADOW,
                 proc_params.QA_CLOUD,
                 proc_params.QA_FILL)

    for offset in heirarchy:
        output[(quality & 1 << offset) > 0] = offset

    check_unbitpack(output, quality, proc_params.QA_NAN)

    return output


def check_unbitpack(unpacked, packed, nan):
    """
    Check to insure that all values were unpacked.

    Args:
        unpacked: ndarray of ints that represent the unpacking
        packed: ndarray of ints that represent bit packed values
        nan: value used to represent NAN

    Raises:
        ValueError
    """
    if np.sum(unpacked == nan):
        raise ValueError('Received the following unknown bit packed values: {}'
                         .format(np.unique(packed[unpacked == nan])))


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
