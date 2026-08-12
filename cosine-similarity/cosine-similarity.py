import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.array(a)
    b = np.array(b)
    euclid_a = np.sqrt(np.sum(a**2))
    euclid_b = np.sqrt(np.sum(b**2))
    if (euclid_a == 0 or euclid_b == 0):
        return 0
    return (a@b)/(euclid_a*euclid_b)