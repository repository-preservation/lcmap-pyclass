import numpy as np

import pyclass
from pyclass import training

params = pyclass.app.get_params()


def test_class_stats():
    arr = np.arange(5)
    prct_ans = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    vals, prct = training.class_stats(arr)

    assert np.array_equal(arr, vals)
    assert np.array_equal(prct, prct_ans)


def test_sample():
    rng1 = np.random.RandomState(seed=1)

    arr = np.array(list(range(5)) * 5)

    indices = training.sample(arr, params['randomforest'], random_state=rng1)

    ans = np.array([11, 6, 21, 1, 16, 2, 12, 22, 17, 7, 13, 18, 8, 23, 3, 19, 4, 9, 14, 24])

    assert np.array_equal(ans, indices)


def test_train_randomforest():
    """
    Sanity check
    """
    xfile = r'test/resources/Xs_grid02.npy'
    yfile = r'test/resources/Ys_grid02.npy'

    xarr = np.load(xfile)
    yarr = np.load(yfile)

    rmse = xarr[:, :7]
    coefs = xarr[:, 7:63]
    aspect = xarr[:, 63]
    dem = xarr[:, 64]
    posidex = xarr[:, 65]
    slope = xarr[:, 66]
    mpw = xarr[:, 67]
    water_prob = xarr[:, 68]
    snow_prob = xarr[:, 69]
    cloud_prob = xarr[:, 70]

    independent = np.hstack((coefs,
                             rmse,
                             dem[:, np.newaxis],
                             aspect[:, np.newaxis],
                             slope[:, np.newaxis],
                             posidex[:, np.newaxis],
                             mpw[:, np.newaxis],
                             cloud_prob[:, np.newaxis],
                             snow_prob[:, np.newaxis],
                             water_prob[:, np.newaxis]))

    rs = np.random.RandomState(seed=1)

    model, metrics = training.train_randomforest(independent,
                                                 yarr,
                                                 params['randomforest'],
                                                 random_state=rs)
