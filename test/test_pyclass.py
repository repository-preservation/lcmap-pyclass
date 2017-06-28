import numpy as np

import pyclass
from test.shared import read_ccdresults

params = pyclass.app.get_params()
ccd = read_ccdresults('test/resources/results.npy')


def test_train():
    """
    WiP
    """
    xfile = r'test/resources/Xs_grid02.npy'
    yfile = r'test/resources/Ys_grid02.npy'

    xarr = np.load(xfile)
    yarr = np.load(yfile)
