import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    X = np.array(X)
    std = np.std(X, axis = axis, keepdims = True)
    div = np.where(std <= eps, 1.0, std)
    z = (X - np.mean(X, axis = axis, keepdims = True))/div
    return z