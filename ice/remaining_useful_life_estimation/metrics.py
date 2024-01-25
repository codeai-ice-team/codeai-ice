import numpy as np


def rmse(pred: list, target: list) -> float:
    """
    Root mean squared error between real and predicted remaining useful life values.

    Args:
        pred (list): numpy prediction values.
        target (list): numpy target values.

    Returns:
        float: rmse
    """
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def nonsimmetric_function(value):
    """
    Exponent calculation function depending on the relative deviation from the true value

    Args:
        value (float): division result between predicted and real values.

    Returns:
        float: calculation result
    """

    return float(np.exp((-value / 13)) - 1 if value < 0 else np.exp((value / 10)) - 1)


def score(pred: list, target: list) -> float:
    """
    Non-simmetric metric proposed in the original dataset paper.
    DOI: 10.1109/PHM.2008.4711414

    Args:
        pred (list): numpy prediction values.
        target (list): numpy target values.

    Returns:
        float: cmapss score function
    """
    function = np.vectorize(nonsimmetric_function)
    return float(np.sum(function(pred - target)))
