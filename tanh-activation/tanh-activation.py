import numpy as np

def tanh(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.array(x)
    tanh_x = (1 - np.exp(-2*x))/(1 + np.exp((-2*x)))
    return tanh_x