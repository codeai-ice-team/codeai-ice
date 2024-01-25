def accuracy(pred: list, target: list) -> float:
    """
    Accuracy of the classification is the number of true positives divided by 
    the number of examples.

    Args:
        pred (list): predictions.
        target (list): target values.
    
    Returns:
        float: accuracy
    """
    return sum(pred == target) / len(pred)
