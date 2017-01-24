import numpy as np

from pyclass import training


def test_reclass_target():
    arr = np.arange(5)

    tgt_from = [2, 1]
    tgt_to = [3, 3]

    ans = np.array([0, 3, 3, 3, 4])

    res = training.reclass_target(arr, tgt_from, tgt_to)

    assert np.array_equal(ans, res)
