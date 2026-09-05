import math
import numpy as np

def gelu(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.array(x)
    erf = np.vectorize(math.erf)
    x = x/2*(1 + erf(x / math.sqrt(2)))
    return x