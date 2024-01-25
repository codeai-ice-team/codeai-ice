import numpy as np
from sklearn.metrics import confusion_matrix

def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Accuracy of the classification is the number of true positives divided by 
    the number of examples.

    Args:
        pred (np.ndarray): predictions.
        target (np.ndarray): target values.
    
    Returns:
        float: accuracy.
    """
    return sum(pred == target) / len(pred)


def correct_daignosis_rate(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Correct Diagnosis Rate is the total number of correctly diagnosed faulty 
    samples divided by the number of detected faulty samples.

    Args:
        pred (np.ndarray): predictions.
        target (np.ndarray): target values.
    
    Returns:
        float: value of Correct Diagnosis Rate.
    """
    cm = confusion_matrix(target, pred, labels=np.arange(target.max() + 1))
    correct = cm[1:, 1:].diagonal().sum()
    true_positive = cm[1:, 1:].sum()
    return correct / true_positive


def true_positive_rate(pred: np.ndarray, target: np.ndarray) -> np.ndarray[float]:
    """
    True Positive Rate is the number of detected faults i divided by the 
    number of faults i.

    Args:
        pred (np.ndarray): predictions.
        target (np.ndarray): target values.
    
    Returns:
        list: list of float values with true positive rate for each fault.
    """
    cm = confusion_matrix(target, pred, labels=np.arange(target.max() + 1))
    correct = cm[1:, 1:].diagonal()
    return list(correct / cm[1:].sum(axis=1))


def false_positive_rate(pred: np.ndarray, target: np.ndarray) -> np.ndarray[float]:
    """
    False Positive Rate, aka False Alarm Rate is the number of false alarms i 
    divided by the number of normal samples.

    Args:
        pred (np.ndarray): predictions.
        target (np.ndarray): target values.
    
    Returns:
        list: list of float values with true positive rate for each fault.
    """
    cm = confusion_matrix(target, pred, labels=np.arange(target.max() + 1))
    return list(cm[0, 1:] / cm[0].sum())
