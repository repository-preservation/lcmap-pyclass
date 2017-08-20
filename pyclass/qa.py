"""
ARD QA related methods.
"""
import numpy as np


def unpackqa(quality, qainfo):
    """
    Transform the bit-packed QA values into their bit offset.

    Institute a hierarchy of qa values that may be flagged in the bitpacked
    value.

    fill > cloud > shadow > snow > water > clear

    Args:
        quality: 1-d array or list of bit-packed QA values
        qainfo: dict of associated qa information

    Returns:
        ndarray: quality as thematic values
    """
    output = np.ones_like(quality)
    output.fill(qainfo['nan'])

    # We need to first account for special conditions for Landsat 8 pixelQA
    # inconsistencies.
    output[((quality & 1 << qainfo['cirrus1']) > 0) &
           ((quality & 1 << qainfo['cirrus2']) > 0)] = qainfo['clear']

    output[(quality & 1 << qainfo['occlusion']) > 0] = qainfo['clear']

    # Reverse hierarchy, values later in the list will overwrite earlier values
    heirarchy = (qainfo['clear'],
                 qainfo['water'],
                 qainfo['snow'],
                 qainfo['shadow'],
                 qainfo['cloud'],
                 qainfo['fill'])

    for offset in heirarchy:
        output[(quality & 1 << offset) > 0] = offset

    check_unbitpack(output, quality, qainfo['nan'])

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


# noinspection PyTypeChecker
def quality_stats(quality, qainfo, axis=1):
    """
    Take the initial quality information for the entire time series, then
    calculate the probabilities for cloud, snow, and water. These outputs are
    used for input parameters to both training and predicting.

    Args:
        quality: QA values from all available observations
        qainfo: dict of associated qa information

    Returns:
        1-d ndarray: cloud probability
        1-d ndarray: snow probability
        1-d ndarray: water probability
    """
    quality = unpackqa(quality, qainfo)

    total_count = np.sum(quality != qainfo['fill'], axis=axis)
    clear_water_count = np.sum((quality == qainfo['clear']) |
                               (quality == qainfo['water']), axis=axis)
    cloud_count = np.sum(quality == qainfo['cloud'], axis=axis)
    snow_count = np.sum(quality == qainfo['snow'], axis=axis)
    water_count = np.sum(quality == qainfo['water'], axis=axis)

    cloud_prob = cloud_count / total_count
    snow_prob = snow_count / (clear_water_count + snow_count + 0.01)
    water_prob = water_count / (clear_water_count + 0.01)

    return cloud_prob, snow_prob, water_prob
