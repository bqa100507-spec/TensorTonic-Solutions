import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    rng = range(y_pred.shape[0])
    y_prob = y_pred[rng, y_true]
    return -np.mean(np.log(y_prob))