import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x)
    pick = None
    if rng is not None :
        pick = rng.random(x.shape)
    else :
        pick = np.random.random(x.shape)
    
    dropout_pattern = np.where(pick < (1-p), 1, 0)*(1.0/(1-p))
    
    return (x*dropout_pattern, dropout_pattern)