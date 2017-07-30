import logging
from collections import namedtuple

import numpy as np

from pyclass import app

ChangeModel = namedtuple('ChangeModel', 'start_day end_day coefs rmses')

log = logging.getLogger(__name__)


def filter_ccd(ccd, ccdinfo):
    """
    Filter the ccd results based on the time segments to ensure they fall
    within a given time range. Returns a list of coefficients, rmse's, and
    their index locations from the input array.

    Args:
        ccd: pyccd results
        ccdinfo: dict of CCD related processing parameters

    Returns:
        2-d ndarray coefficient values
        2-d ndarray rmse values
        1-d ndarray index locations
    """
    coefs = []
    rmse = []
    idx = []

    results = [filter_result(r, ccdinfo) if r else None for r in ccd]

    for i, res in enumerate(results):
        if res:
            coefs.append(res[0])
            rmse.append(res[1])
            idx.append(i)

    return np.array(coefs), np.array(rmse), np.array(idx)


def unpack_ccd(ccd, ccdinfo):
    """
    Unpack all the CCD curve fitted information for all the given models. There
    is no filtering done, all the change model segments are dumped out.

    The returned index array can be used to duplicate array locations in
    corresponding data sets.

    Args:
        ccd: pyccd results
        ccdinfo: dict of CCD related processing parameters

    Returns:
        2-d ndarray coefficient values
        2-d ndarray rmse values
        1-d ndarray index locations
    """
    coefs = []
    rmse = []
    idx = []

    for i, result in enumerate(ccd):
        models = unpack_result(result, ccdinfo)

        for model in models:
            coefs.append(model.coefs)
            rmse.append(model.rmses)
            idx.append(i)

    return np.array(coefs), np.array(rmse), np.array(idx)


def filter_result(result, ccdinfo):
    """
    Filter a CCD result for a pixel looking for a change model that meets the
    specified temporal criteria.

    Args:
        result: CCD result for a pixel
        ccdinfo: dict of CCD related processing parameters
    Returns:
        1-d ndarray
        1-d ndarray
    """
    models = unpack_result(result, ccdinfo)

    for model in models:
        if check_coverage(model, ccdinfo['begin_day'], ccdinfo['end_day']):
            return model.coefs, model.rmses


def unpack_result(result, ccdinfo):
    """
    Unpacks the curve fit information for all change models contained in a
    pyccd result.

    Args:
        result: pyccd results
        ccdinfo: dict of CCD related processing parameters

    Returns:
        list of ChangeModel namedtuple's
    """
    models = []

    for model in result['change_models']:
        curveinfo = extract_curve(model, ccdinfo)

        models.append(ChangeModel(start_day=model['start_day'],
                                  end_day=model['end_day'],
                                  coefs=curveinfo[0],
                                  rmses=curveinfo[1]))

    return models


def extract_curve(model, ccdinfo):
    """
    Helper method to pull out the curve fit information from a given model
    and return it as two flattened ndarrays.

    Args:
        model: pyccd change model as a dict
        ccdinfo: dict of CCD related processing parameters

    Returns:
        1-d ndarray
        1-d ndarray
    """
    bands = app.sorted_list(ccdinfo['bands'])

    coefs = np.zeros(shape=(len(bands), ccdinfo['coef_count']))
    rmse = np.zeros(shape=(len(bands),))

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
    return (model.start_day <= begin_ord) & (model.end_day >= end_ord)
