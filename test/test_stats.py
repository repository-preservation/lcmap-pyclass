import numpy as np

from pyclass import stats


def test_quality_stats():
    qa = [[4, 4, 4, 4],
          [4, 4, 255, 255],
          [3, 3, 3, 3],
          [1, 1, 1, 1],
          [0, 4, 4, 0],
          [0, 1, 3, 4]]

    cloud_prob = np.array([100,
                           100,
                           0,
                           0,
                           50,
                           25])

    snow_prob = np.array([0,
                          0,
                          4 / 4.01 * 100,
                          0,
                          0,
                          1 / 3.01 * 100])

    water_prob = np.array([0,
                           0,
                           0,
                           4 / 4.01 * 100,
                           0,
                           1 / 2.01 * 100])

    cloud, snow, water = stats.quality_stats(qa)

    assert np.array_equal(cloud_prob, cloud)
    assert np.array_equal(snow_prob, snow)
    assert np.array_equal(water_prob, water)
