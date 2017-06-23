import numpy as np

import pyclass


params = pyclass.app.get_params()


def test_quality_stats():
    qainfo = params['qa']

    qa = [[32, 32, 32, 32],
          [32, 32, 1, 1],
          [16, 16, 16, 16],
          [4, 4, 4, 4],
          [2, 32, 32, 2],
          [2, 4, 16, 32]]

    cloud_prob = np.array([1,
                           1,
                           0,
                           0,
                           .50,
                           .25])

    snow_prob = np.array([0,
                          0,
                          4 / 4.01,
                          0,
                          0,
                          1 / 3.01])

    water_prob = np.array([0,
                           0,
                           0,
                           4 / 4.01,
                           0,
                           1 / 2.01])

    cloud, snow, water = pyclass.qa.quality_stats(np.array(qa), qainfo)

    assert np.array_equal(cloud_prob, cloud)
    assert np.array_equal(snow_prob, snow)
    assert np.array_equal(water_prob, water)
