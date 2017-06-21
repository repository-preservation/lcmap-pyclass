import logging
from collections import namedtuple

import numpy as np

ChangeModel = namedtuple('ChangeModel', 'start_day end_day coefs rmses')

log = logging.getLogger(__name__)


def filter_ccd(ccd, ccdinfo):
    """
    Filter the ccd results based on the time segments to ensure they fall
    within a given time range.

    Args:
        ccd: pyccd results
        ccdinfo: dict of CCD related processing parameters

    Returns:
        1-d ndarray
        1-d ndarray
    """
    models = unpack_ccd(ccd, ccdinfo)

    for model in models:
        if check_coverage(model, ccdinfo['begin_day'], ccdinfo['end_day']):
            return model.coefs, model.rmses


def unpack_ccd(ccd, ccdinfo):
    """
    Unpacks the curve fit information for all change models contained in a
    pyccd result.

    Args:
        ccd: pyccd results
        ccdinfo: dict of CCD related processing parameters

    Returns:
        list of ChangeModel namedtuple's
    """
    models = []

    for model in ccd['change_models']:
        curveinfo = extract_curve(model, ccdinfo['bands'], ccdinfo['coef_count'])

        models.append(ChangeModel(start_day=model['start_day'],
                                  end_day=model['end_day'],
                                  coefs=curveinfo[0],
                                  rmses=curveinfo[1]))

    return models


def extract_curve(model, bands, coef_count):
    """
    Helper method to pull out the curve fit information from a given model
    and return it as two flattened ndarrays.

    Args:
        model: pyccd change model as a dict
        bands: list of band names used by pyccd
        coef_count: default number of coefficients used by pyccd
    Returns:
        1-d ndarray
        1-d ndarray
    """
    coefs = np.zeros(shape=(len(bands), coef_count))
    rmse = np.full(shape=(len(bands),), fill_value=-1)

    for i, b in enumerate(bands):
        coefs[i] = model[b]['coefficients'] + [model[b]['intercept']]
        rmse[i] = model[b]['rmse']

    return coefs.flatten(), rmse


def check_coverage(model, begin_ord, end_ord):
    """
    Helper method to determine if a model covers a given time frame, in it's
    entirety.

    Args:
        model: pyccd result model as a namedtuple
        begin_ord: ordinal day
        end_ord: ordinal day
    Returns:
        bool
    """
    return model.start_day <= begin_ord & model.end_day >= end_ord


def band_list(bands_dict):
    """
    Helper method to take the band dictionary from the parameters, and create
    a list of the values ordered on the keys.

    Args:
        bands_dict: dictionary related the order of the band names

    Returns:
        list
    """
    return [bands_dict[k] for k in sorted(bands_dict.keys())]
