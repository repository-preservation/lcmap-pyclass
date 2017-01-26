import numpy as np
from sklearn.ensemble import RandomForestClassifier

from pyclass.app import logging, defaults, gen_rng


log = logging.getLogger(__name__)


def reclass_target(vector, tgt_from=defaults.TARGET_FROM, tgt_to=defaults.TARGET_TO):
    """
    Handle classes in the target data set that need to be combined or recoded.

    Args:
        vector: 1-d ndarray.
        tgt_from: list of ints, must be the same size as tgt_to
        tgt_to: list of ints, must be the same size as tgt_from

    Returns:
        1-d ndarray of recoded values.
    """
    # Let's try to mitigate any mutability issues
    out = vector[:]

    for i in range(len(tgt_from)):
        out[out == tgt_from[i]] = tgt_to[i]

    return out


def class_stats(dependent):
    """
    Helper method to determine the unique class values and the percentage
    that each class represents in the data set.

    Args:
        dependent: 1-d ndarry of int values

    Returns:
        1-d ndarray of unqiue values
        1-d ndarray percentage of represenatation
    """
    class_values = np.unique(dependent)

    # Find the percentage that each class makes up in the data set
    prct, _ = np.histogram(dependent, class_values.shape[0])
    prct /= np.sum(prct, dtype=float)

    return class_values, prct


def sample(dependent, minimum=defaults.CLASS_MIN, maximum=defaults.CLASS_MAX,
           total=defaults.TOTAL_SAMPLE_SIZE, random_state=None):
    """
    Since we have a maximum number of samples that we want to hit

    Args:
        dependent: 1-d int ndarray.
        minimum: minimum number of samples we want of a single class
            currently doesn't matter with the prescribed methodology
        maximum: maximum number of samples we want of a single class
        total: total number of samples we want to hit, across all classes
        random_state: numpy random state object

    Returns:
        array of index locations to use for training purposes
    """
    if random_state is None:
        random_state = gen_rng()

    class_values, percent = class_stats(dependent)

    # Adjust the target counts that we are wanting based on the percentage
    # that each one represents in the base data set
    adj_counts = np.ceil(total * percent)
    adj_counts[adj_counts > maximum] = maximum
    # adj_counts[adj_counts < minimum] = minimum

    selected_indices = []
    for cls, count in zip(class_values, adj_counts):
        # Index locations of values
        indices = np.where(dependent == cls)[0]

        # Randomize the order of the index locations
        indices = random_state.permutation(indices)

        # Add the index locations up to the count
        selected_indices.extend(indices[:count])

    return selected_indices


def train_randomforest(independent, dependent,
                       n_estimators=defaults.RANDOM_FOREST_ESTIMATORS,
                       random_state=None):
    """
    Train a random forest with a given set independent variables and an
    array of target/dependent values.

    For more information see:
    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    Args:
        independent: 2-d ndarray of the independent predictor values
            where each row is a sample and each column is an attribute
        dependent: 1-d ndarray of dependent values, what we are trying to
            predict
        n_estimators: int. number of trees/estimators for the random forest
        random_state: initialized numpy RandomState class

    Returns:
        SKLearn RandomForestClassifier class
    """
    # First we need to determine which samples we want to use, based on the
    # distribution of the target/dependant data set.
    indices = sample(dependent, random_state=random_state)

    # Grab the samples that we want from the data sets.
    X = independent[indices]
    y = dependent[indices]

    # Initialize the RandomForestClassifier then produce a fit for
    # the data sets.
    rfmodel = RandomForestClassifier(random_state=random_state,
                                     n_estimators=n_estimators)
    rfmodel.fit(X, y)

    return rfmodel, random_state
