import numpy as np


def mse(pred: list, target: list) -> float:
    """
    Mean squared error between real and predicted wear.

    Args:
        pred (list): numpy prediction values.
        target (list): numpy target values.

    Returns:
        float: rmse
    """
    return float(np.mean((pred - target) ** 2))

def rmse(pred: list, target: list) -> float:
    """
    Mean squared error between real and predicted wear.

    Args:
        pred (list): numpy prediction values.
        target (list): numpy target values.

    Returns:
        float: rmse
    """
    return float(np.sqrt(np.mean((pred - target) ** 2)))
