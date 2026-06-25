import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    p = np.ndim(x)-1
    x = x - np.max(x, axis=p, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis = p, keepdims=True)
    return x