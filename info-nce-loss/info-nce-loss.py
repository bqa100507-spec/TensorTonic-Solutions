import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    """
    Returns the loss as a float.
    """
    z1, z2 = np.array(Z1), np.array(Z2)
    s = (z1@(z2.T))/temperature
    s = s - np.max(s, axis=1, keepdims = True)
    s = np.exp(s)

    diag = np.diag(s)
    sum = np.sum(s, axis = 1)

    loss = - np.mean(np.log(diag / sum))
    return loss.item()