from pyclass import app
log = app.logging.getLogger(__name__)


def rf_predict(forest, X):
    """
    Takes in a trained classfier model and predicts the samples that are passed
    to it.

    Args:
        forest: Trained SKlearn random forest classifier object.
        X: Samples to be classified.

    Returns:
        probability for each class within each sample
    """
    return forest.predict_proba(X)
