"""
Main lead in methods
"""
import logging

import numpy as np

from pyclass import app, training, classifier, change, qa
from .version import __version__
from .version import __algorithm__ as algorithm
from .version import __name

log = logging.getLogger(__name__)


def __attach_trainmetadata(model, random_seed, sample_cts):
    """
    Helper method to attach information and format processing information.

    {algorithm: string,
     rf_model: sklearn.ensemble.RandomForestClassifier object,
     random_seed: tuple,
     sample_cts: string,
     messages: list of strings}

    Args:
        model: sklearn random forest object
        random_seed: tuple used to initialize a numpy RandomState object
        metrics: string of performance metrics and information of the model
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

    Returns:
        dict
    """
    return {'algorithm': algorithm,
            'rf_model': model,
            'random_seed': random_seed,
            'sample counts': sample_cts}


def train(trends, ccd, dem, aspect, slope, posidex, mpw, cloud_prob, snow_prob,
          water_prob, random_seed=None, proc_params=app.get_params()):
    """
    Main module entry point for training a new classification model.

    Implements the classification methodology for the LCMAP Project's
    Change Detection and Classification (CCDC) as outlined by the principle
    algorithm investigator Zhe Zhu.

    When passing in the coefs and rmse values, it is important to be
    consistent with how they are passed into the classify method.

    References:
        http://www.sciencedirect.com/science/article/pii/S0924271616302829

    Pertinent numpy random module information:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.get_state.html
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.set_state.html

    Args:
        trends: 1-d ndarray. Representative values that are trying
            to be predicted given other predictors. Dependent variable.
        ccd: 1-d array of dict like structure conforming to the pyccd output
            structure. https://github.com/USGS-EROS/lcmap-pyccd
        dem: 1-d ndarray. Digital elevation model values.
        aspect: 1-d ndarray. DEM derived product.
        slope: 1-d ndarray. DEM derived product.
        posidex: 1-d ndarray. DEM derived product.
        mpw: 1-d ndarray. Max potential wetland product.
        cloud_prob: 1-d ndarray. Probably of cloud from QA.
        snow_prob: 1-d ndarray. Probably of snow from QA.
        water_prob: 1-d ndarray. Probably of water from QA.
        random_seed: tuple used to initialize a numpy RandomState object or None
        proc_params: python dictionary to change module wide processing
            parameters

    Returns:
        dict
    """
    qainfo = proc_params['qa']
    ccdinfo = proc_params['ccd']
    rfinfo = proc_params['randomforest']

    # Make sure we have proper governance in place for random number generation
    random_state = app.gen_rng()

    if random_seed is None:
        random_seed = random_state.get_state()
    else:
        random_state.set_state(random_seed)

    coefs, rmse, idx = change.filter_ccd(ccd, ccdinfo)

    # Stack the independent arrays into a single cohesive block.
    independent = np.hstack((coefs,
                             rmse,
                             dem[idx, np.newaxis],
                             aspect[idx, np.newaxis],
                             slope[idx, np.newaxis],
                             posidex[idx, np.newaxis],
                             mpw[idx, np.newaxis],
                             cloud_prob[idx, np.newaxis],
                             snow_prob[idx, np.newaxis],
                             water_prob[idx, np.newaxis]))

    # Where we need to get to.
    # TODO makes this configurable
    model, sample_cts = training.train_randomforest(independent,
                                                    trends,
                                                    rfinfo,
                                                    random_state=random_state)

    return __attach_trainmetadata(model, random_seed, sample_cts)


def classify(model, ccd, dem, aspect, slope, posidex, mpw, cloud_prob,
             snow_prob, water_prob, proc_params=app.get_params()):
    """
    Main module entry point for classifying a sample or series of samples.

    The return is a list of dicts, with each dict corresponding to each
    time segment in the ccd result.

    Args:
        model: Trained classifier model object.
        ccd: 1-d array of dict like structure conforming to the pyccd output
            structure. https://github.com/USGS-EROS/lcmap-pyccd
        dem: 1-d ndarray. Digital elevation model values.
        aspect: 1-d ndarray. DEM derived product.
        slope: 1-d ndarray. DEM derived product.
        posidex: 1-d ndarray. DEM derived product.
        mpw: 1-d ndarray. Max potential wetland product.
        cloud_prob: 1-d ndarray. Probably of cloud from QA.
        snow_prob: 1-d ndarray. Probably of snow from QA.
        water_prob: 1-d ndarray. Probably of water from QA.
        proc_params: python dictionary to change module wide processing
            parameters

    Returns:
        list of dicts
    """
    qainfo = proc_params['qa']
    ccdinfo = proc_params['ccd']
    rfinfo = proc_params['randomforest']

    # Stack the independent arrays into a single cohesive block.
    aux = np.hstack((dem,
                     aspect,
                     slope,
                     posidex,
                     mpw,
                     cloud_prob,
                     snow_prob,
                     water_prob))

    # TODO make this configurable
    return classifier.classify_ccd(model, ccd, aux, ccdinfo)
