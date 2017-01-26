"""
Main lead in methods
"""
import numpy as np

from pyclass import app, stats, training
log = app.logging.getLogger(__name__)


def train(trends, coefs, rmse, dem, aspect, slope, posidex, mpw, qa, random_state=None):
    """
    Main module entry point for training a new classification model.

    Implementation of the

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
        random_state: initialized numpy RandomState object or None

    Returns:
        prediction model for classified
        initialzed numpy RandomState object

    """
    if random_state is None:
        random_state = app.gen_rng()

    # Turn the QA values into requisite three probabilities
    cloud_prob, snow_prob, water_prob = stats.quality_stats(qa)

    # TODO
    # Stack the independent arrays into a single cohesive block.
    independent = np.array()

    # Where we need to get to.
    # TODO makes this configurable
    model = training.train_randomforest(independent, trends, random_state=random_state)

    return model, random_state


def classify(coefs, rmse, dem, aspect, slope, posidex, mpw, qa):
    cloud_prob, snow_prob, water_prob = stats.quality_stats(qa)
