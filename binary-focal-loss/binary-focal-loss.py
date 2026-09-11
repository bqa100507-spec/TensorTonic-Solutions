import math
import numpy as np

def binary_focal_loss(predictions: list, targets: list, alpha: float, gamma: float) -> float:
    """
    Returns the mean binary focal loss as a float.
    """
    pred, label = np.array(predictions), np.array(targets)
    pred[label == 0] = 1 - pred[label == 0]
    loss = np.mean(-alpha*((1- pred)**gamma)*np.log(pred))
    return loss.item()