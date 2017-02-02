import numpy as np

from pyclass import training, classifier


def test_reclass_target():
    arr = np.arange(5)

    tgt_from = [2, 1]
    tgt_to = [3, 3]

    ans = np.array([0, 3, 3, 3, 4])

    res = training.reclass_target(arr, tgt_from, tgt_to)

    assert np.array_equal(ans, res)


def test_class_stats():
    arr = np.arange(5)
    prct_ans = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    vals, prct = training.class_stats(arr)
    print(vals)
    print(prct)

    assert np.array_equal(arr, vals)
    assert np.array_equal(prct, prct_ans)


def test_sample():
    rng1 = np.random.RandomState(seed=1)

    arr = np.array(list(range(5)) * 5)

    indices = training.sample(arr, minimum=1, maximum=3, random_state=rng1)

    ans = [10, 5, 20, 1, 11, 21, 12, 17, 7, 18, 3, 8, 19, 4, 9]

    assert ans == indices


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

    model = training.train_randomforest(independent, yarr, random_state=rs)

    classes, results = classifier.rf_predict(model, independent[0])

    assert np.array_equal(classes, np.array([0, 1, 2, 5, 6, 7, 8, 9, 10]))

    assert np.array_equal(results, np.array([[0.12, 0, 0.15, 0.02, 0.07,
                                              0.37, 0.05, 0.04, 0.18]]))


