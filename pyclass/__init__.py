"""
Main lead in methods
"""
import logging

import numpy as np

from pyclass import app, stats, training, classifier, change, qa

log = logging.getLogger(__name__)


def train(trends, ccd, dem, aspect, slope, posidex, mpw, quality, random_seed=None,
          proc_params=app.get_params()):
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
        trends: 1-d array or list. Representative values that are trying
            to be predicted given other predictors. Dependent variable.
        ccd: 1-d array of dict like structure conforming to the pyccd output
            structure. https://github.com/USGS-EROS/lcmap-pyccd
        coefs: 2-d array or list. The various coefficients generated from the
            Continuous Change Detection module.
        rmse: 2-d array or list. RMSE associated with the models that were
            generated during the Continuous Change Detection module.
        dem: 1-d array or list. Digital elevation model values.
        aspect: 1-d array or list. DEM derived product.
        slope: 1-d array or list. DEM derived product.
        posidex: 1-d array or list. DEM derived product.
        mpw: 1-d array or list.
        quality: 2-d array or list. Observation quality values for the entire
            history of the sample.
        random_seed: tuple used to initialize a numpy RandomState object or None


    Returns:
        SKLearn RandomForestClassifier object, prediction model
        tuple used for setting the state of a numpy RandomState object

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

    # Turn the QA values into requisite three probabilities
    cloud_prob, snow_prob, water_prob = qa.quality_stats(quality, qainfo)

    coefs, rmse = change.filter_ccd(ccd, ccdinfo)

    # Stack the independent arrays into a single cohesive block.
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

    # Where we need to get to.
    # TODO makes this configurable
    model = training.train_randomforest(independent, trends, random_state=random_state)

    return model, random_seed


def classify(ccd, dem, aspect, slope, posidex, mpw, quality,
             proc_params=app.get_params()):
    """
    Main module entry point for classifying a sample or series of samples.

    Args:
        model: Trained classifier model object.
        ccd: 1-d array of dict like structure conforming to the pyccd output
            structure. https://github.com/USGS-EROS/lcmap-pyccd
        coefs: 2-d array or list. The various coefficients generated from the
            Continuous Change Detection module.
        rmse: 2-d array or list. RMSE associated with the models that were
            generated during the Continuous Change Detection module.
        dem: 1-d array or list. Digital elevation model values.
        aspect: 1-d array or list. DEM derived product.
        slope: 1-d array or list. DEM derived product.
        posidex: 1-d array or list. DEM derived product.
        mpw: 1-d array or list.
        quality: 2-d array or list. Observation quality values for the entire
            history of the sample.

    Returns:
        1-d ndarray of ordered class values represented in the model
        2-d ndarray of the probability for each class within each sample
    """
    qainfo = proc_params['qa']
    ccdinfo = proc_params['ccd']
    rfinfo = proc_params['randomforest']

    cloud_prob, snow_prob, water_prob = qa.quality_stats(quality, qainfo)

    # Stack the independent arrays into a single cohesive block.
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

    # TODO make this configurable
    return classifier.rf_predict(model, independent)
