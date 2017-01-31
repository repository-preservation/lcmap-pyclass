"""
Main lead in methods
"""
import numpy as np

from pyclass import app, stats, training, classifier
log = app.logging.getLogger(__name__)


def train(trends, coefs, rmse, dem, aspect, slope, posidex, mpw, qa, random_seed=None):
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
        coefs: 2-d array or list. The various coefficients generated from the
            Continuous Change Detection module.
        rmse: 2-d array or list. RMSE associated with the models that were
            generated during the Continuous Change Detection module.
        dem: 1-d array or list. Digital elevation model values.
        aspect: 1-d array or list. DEM derived product.
        slope: 1-d array or list. DEM derived product.
        posidex: 1-d array or list. DEM derived product.
        mpw: 1-d array or list.
        qa: 2-d array or list. Observation quality values for the entire
            history of the sample.
        random_seed: tuple used to initialize a numpy RandomState object or None


    Returns:
        prediction model for classified
        initialzed numpy RandomState object

    """
    # Make sure we have proper governance in place for random number generation
    random_state = app.gen_rng()

    if random_seed is None:
        random_seed = random_state.get_state()
    else:
        random_state.set_state(random_seed)

    # Turn the QA values into requisite three probabilities
    cloud_prob, snow_prob, water_prob = stats.quality_stats(qa)

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


def classify(model, coefs, rmse, dem, aspect, slope, posidex, mpw, qa):
    """
    Main module entry point for classifying a sample or series of samples.

    Args:
        model: Trained classifier model object.
        coefs: 2-d array or list. The various coefficients generated from the
            Continuous Change Detection module.
        rmse: 2-d array or list. RMSE associated with the models that were
            generated during the Continuous Change Detection module.
        dem: 1-d array or list. Digital elevation model values.
        aspect: 1-d array or list. DEM derived product.
        slope: 1-d array or list. DEM derived product.
        posidex: 1-d array or list. DEM derived product.
        mpw: 1-d array or list.
        qa: 2-d array or list. Observation quality values for the entire
            history of the sample.

    Returns:
        probability for each class within each sample
    """

    cloud_prob, snow_prob, water_prob = stats.quality_stats(qa)

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
