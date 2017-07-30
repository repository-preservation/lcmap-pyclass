from collections import namedtuple

import numpy as np

import pyclass
from pyclass import change
from test.shared import read_ccdresults


params = pyclass.app.get_params()


ccd = read_ccdresults('test/resources/results.npy')
ccdinfo = {'begin_day': 729755,
           'end_day': 730850,
           'coef_count': 8,  # includes intercept
           'bands': {0: 'blue',
                     1: 'green',
                     2: 'red',
                     4: 'swir1',
                     3: 'nir',
                     5: 'swir2',
                     6: 'thermal'}}


def test_band_list():
    """
    Ensure the list comes back in the correct order.
    """
    ans = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']
    bandls = pyclass.app.sorted_list(ccdinfo['bands'])

    assert np.array_equal(ans, bandls)


def test_unpack_result():
    """
    Make sure each model in a result comes back separately.
    """
    models = change.unpack_result(ccd[0], ccdinfo)

    assert len(ccd[0]['change_models']) == len(models)


def test_check_coverage():
    """
    """
    Model = namedtuple('Model', 'start_day end_day')

    good = Model(start_day=ccdinfo['begin_day'], end_day=ccdinfo['end_day'])
    bad1 = Model(start_day=ccdinfo['begin_day'] + 1, end_day=ccdinfo['end_day'])
    bad2 = Model(start_day=ccdinfo['begin_day'], end_day=ccdinfo['end_day'] - 1)

    assert change.check_coverage(good, ccdinfo['begin_day'], ccdinfo['end_day'])

    assert not change.check_coverage(bad1, ccdinfo['begin_day'], ccdinfo['end_day'])
    assert not change.check_coverage(bad2, ccdinfo['begin_day'], ccdinfo['end_day'])


def test_filter_result():
    """
    Filtering of the results tuples.
    """
    good_idx = 0
    bad_idx = 78

    assert change.filter_result(ccd[good_idx], ccdinfo) is not None
    assert change.filter_result(ccd[bad_idx], ccdinfo) is None


def test_extract_curve():
    """
    """
    coefs = list(range(ccdinfo['coef_count'] - 1))

    model = {'blue': {'coefficients': coefs,
                      'intercept': 0,
                      'rmse': 0},
             'green': {'coefficients': coefs,
                       'intercept': 0,
                       'rmse': 1},
             'red': {'coefficients': coefs,
                     'intercept': 0,
                     'rmse': 2},
             'nir': {'coefficients': coefs,
                     'intercept': 0,
                     'rmse': 3},
             'swir1': {'coefficients': coefs,
                       'intercept': 0,
                       'rmse': 4},
             'swir2': {'coefficients': coefs,
                       'intercept': 0,
                       'rmse': 5},
             'thermal': {'coefficients': coefs,
                         'intercept': 0,
                         'rmse': 6}}

    res_coefs, res_rmse = change.extract_curve(model, ccdinfo)
    ans_coefs = (coefs + [0]) * 7
    ans_rmse = list(range(7))

    assert np.array_equal(res_coefs, ans_coefs)
    assert np.array_equal(res_rmse, ans_rmse)


def test_filter_ccd():
    """
    """
    coefs, rmses, idxs = change.filter_ccd(ccd, ccdinfo)

    assert coefs.ndim == rmses.ndim == 2
    assert idxs.ndim == 1
    assert coefs.shape[1] == ccdinfo['coef_count'] * len(ccdinfo['bands'])
    assert rmses.shape[1] == len(ccdinfo['bands'])
    assert idxs.shape[0] == coefs.shape[0] == rmses.shape[0]


def test_unpack_ccd():
    coefs, rmses, idxs = change.unpack_ccd(ccd, ccdinfo)

    assert coefs.ndim == rmses.ndim == 2
    assert idxs.ndim == 1
    assert coefs.shape[1] == ccdinfo['coef_count'] * len(ccdinfo['bands'])
    assert rmses.shape[1] == len(ccdinfo['bands'])
    assert idxs.shape[0] == coefs.shape[0] == rmses.shape[0]
