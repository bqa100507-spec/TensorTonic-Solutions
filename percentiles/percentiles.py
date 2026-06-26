import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """

    x = np.array(x)
    q = np.array(q)
    x = np.sort(x)
    res = np.percentile(x, q)
    return res
        
        