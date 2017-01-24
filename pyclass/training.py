from pyclass.app import logging, defaults


log = logging.getLogger(__name__)


def reclass_target(target, tgt_from=defaults.TARGET_FROM, tgt_to=defaults.TARGET_TO):
    """
    Handle classes in the target data set that need to be combined or recoded.

    Args:
        target: 1-d ndarray of discrete target values.
        tgt_from: list of ints, must be the same size as tgt_to
        tgt_to: list of ints, must be the same size as tgt_from

    Returns:
        1-d ndarray of recoded values.
    """
    for i in range(len(tgt_from)):
        target[tgt_from[i]] = tgt_to[i]

    return target
