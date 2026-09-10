import numpy as np

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    anchor, positive, negative = np.array(anchor), np.array(positive), np.array(negative)
    dp = anchor - positive
    dn = anchor - negative

    if np.ndim(dp) == 1:
        dp = np.expand_dims(dp, axis=0)
        dn = np.expand_dims(dn, axis=0)

    dp = np.sum(dp**2, axis=1)
    dn = np.sum(dn**2, axis=1)

    loss = np.maximum(0.0, dp - dn + margin)
    return np.mean(loss).item()