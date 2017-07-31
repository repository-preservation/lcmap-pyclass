import logging

import numpy as np

from pyclass import change

log = logging.getLogger(__name__)


def result_template():
    return {'start_day': 0,
            'end_day': 0,
            'class_vals': [],
            'class_probs': []}


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
    # Will get a deprecation warning on a single sample if it is not
    # 2-d
    if len(X.shape) < 2:
        X = X.reshape(1, -1)

    return forest.predict_proba(X)


def classify_ccd(model, ccd, aux, ccdinfo):
    """
    Classify each time segment contained in the pyccd result.

    Args:
        model: scikit-learn random forest model
        ccd: pyccd result
        aux: 1-d ndarray auxillery data set values
        ccdinfo: ccd related processing parameters

    Returns:
        list of dicts
    """
    class_vals = tuple(model.classes_)
    ccd_models = change.unpack_result(ccd, ccdinfo)

    ret = []
    for ccd_model in ccd_models:
        result = result_template()
        probs = rf_predict(model, np.hstack([ccd_model.coefs, ccd_model.rmses, aux]))

        result['start_day'] = ccd_model.start_day
        result['end_day'] = ccd_model.end_day
        result['class_vals'] = class_vals
        result['class_probs'] = probs

        ret.append(result)

    return ret
