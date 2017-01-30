import numpy as np

from pyclass import training


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

    assert np.array_equal(arr, vals)
    assert np.array_equal(prct, prct_ans)


def test_sample():
    pass
    # rng1 = np.random.RandomState()
    # seed = rng1.get_state()
    #
    # rng2 = np.random.RandomState
    # rng2.set_state(seed)
    #
    # arr = np.array(list(range(5)) * 10)
    #
    # indices = training.sample(arr, random_state=rng2)


def test_train_randomforest():
    pass
