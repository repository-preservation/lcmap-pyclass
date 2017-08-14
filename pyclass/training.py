import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


log = logging.getLogger(__name__)


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
    prct = np.true_divide(prct, np.sum(prct))

    return class_values, prct


def sample(dependent, rfinfo, random_state=None):
    """
    Since we have a maximum number of samples that we want to hit

    Args:
        dependent: 1-d int ndarray.
        rfinfo: dict of random forest related processing parameters
        random_state: numpy random state object

    Returns:
        array of index locations to use for training purposes
    """
    exclude = mask_exclusions(dependent, rfinfo)
    class_values, percent = class_stats(dependent[~exclude])

    # Adjust the target counts that we are wanting based on the percentage
    # that each one represents in the base data set.
    adj_counts = np.ceil(rfinfo['target_samples'] * percent)
    adj_counts[adj_counts > rfinfo['class_max']] = rfinfo['class_max']
    adj_counts[adj_counts < rfinfo['class_min']] = rfinfo['class_min']

    selected_indices = []
    for cls, count in zip(class_values, adj_counts):
        # Index locations of values
        indices = np.where(dependent == cls)[0]

        # Randomize the order of the index locations
        indices = random_state.permutation(indices)

        # Add the index locations up to the count
        selected_indices.extend(indices[:int(count)])

    return np.array(selected_indices)


def mask_exclusions(dependent, rfinfo):
    """
    Mask the incoming dependant array to mark classes that we don't
    actually train on. True being the values to exclude from training.

    Args:
        dependent: 1-d ndarray of ints
        rfinfo: dict of random forest related processing parameters

    Returns:
        ndarray bool
    """
    return np.isin(dependent, rfinfo['train_exclude'])


def train_randomforest(independent, dependent, rfinfo, random_state=None):
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
        rfinfo: dict of random forest related processing parameters
        random_state: initialized numpy RandomState class

    Returns:
        SKLearn RandomForestClassifier class
    """
    # First we need to determine which samples we want to use, based on the
    # distribution of the target/dependant data set.
    indices = sample(dependent, rfinfo, random_state=random_state)
    mask = np.zeros_like(dependent, dtype=np.bool)
    mask[indices] = 1
    log.debug('Number of samples for training %s' % indices.shape[0])
    log.debug('Number of samples for testing %s' % (dependent.shape[0] - indices.shape[0]))

    # Grab the samples that we want from the data sets.
    train_X = independent[mask]
    train_y = dependent[mask]

    # Initialize the RandomForestClassifier then produce a fit for
    # the data sets.
    log.debug('Training the Classifier')
    rfmodel = RandomForestClassifier(random_state=random_state,
                                     n_estimators=rfinfo['estimators'])
    rfmodel.fit(train_X, train_y)

    # Number of samples used for each class.
    log.debug('Creating sample counts')
    sample_counts = {}
    for val in rfmodel.classes_:
        sample_counts[val] = np.sum(train_y == val)

    return rfmodel, sample_counts
