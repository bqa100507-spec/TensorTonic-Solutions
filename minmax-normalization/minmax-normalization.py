import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """
    X = np.array(X)
    max = np.max(X, axis=axis, keepdims = True)
    min = np.min(X, axis=axis, keepdims = True)
    scale = max - min
    scale = np.where(scale <= eps, 1.0, scale)
    X = (X - min)/scale
    return X
    