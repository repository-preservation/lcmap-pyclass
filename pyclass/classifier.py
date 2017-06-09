import logging

log = logging.getLogger(__name__)


def rf_predict(forest, X):
    """
    Takes in a trained classfier model and predicts the samples that are passed
    to it.

    Args:
        forest: Trained SKlearn random forest classifier object.
        X: Samples to be classified.

    Returns:
        class values
        probability for each class within each sample
    """
    # Will get a deprecation warning on a single sample if it is not
    # 2-d
    if len(X.shape) < 2:
        X = X.reshape(1, -1)

    return forest.classes_, forest.predict_proba(X)
