import numpy as np

def contrastive_loss(a: list, b: list, y: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    a, b, y = np.array(a), np.array(b), np.array(y)
    if len(y) == 1:
        a = np.expand_dims(a, axis = 0)
        b = np.expand_dims(b, axis = 0)

    d = np.linalg.norm(a-b, axis=1)
    d = np.atleast_1d(np.squeeze(d))

    l = y*(d**2) + (1 - y)*(np.maximum(0, margin - d)**2)

    if reduction == "mean":
        return np.mean(l).item()
    else:
        return np.sum(l).item()