import logging
from collections import namedtuple

import numpy as np

ChangeModel = namedtuple('ChangeModel', 'start_day end_day coefs rmses')

log = logging.getLogger(__name__)


def filter_ccd(ccd, begin_date, end_date, bands, coef_count):
    """
    Filter the ccd results based on the time segments to ensure they fall
    within a given time range.

    Args:
        ccd: pyccd results
        begin_date: ordinal day
        end_date: ordinal day
        bands: list of band names used by pyccd
        coef_count: default number of coefficients used by pyccd

    Returns:
        1-d ndarray
        1-d ndarray
    """
    models = unpack_ccd(ccd, bands, coef_count)

    for model in models:
        if check_coverage(model, begin_date, end_date):
            return model.coefs, model.rmses


def unpack_ccd(ccd, bands, coef_count):
    """
    Unpacks the curve fit information for all change models contained in a
    pyccd result.

    Args:
        ccd: pyccd results
        bands: list of band names used by pyccd
        coef_count: default number of coefficients used by pyccd

    Returns:

    """
    models = []

    for model in ccd['change_models']:
        curveinfo = extract_curve(model, bands, coef_count)

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
