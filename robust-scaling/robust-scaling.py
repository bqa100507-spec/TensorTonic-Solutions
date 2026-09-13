import numpy as np

def robust_scaling(values: list) -> list:
    """
    Returns values centered by the median and scaled by the interquartile range.
    """
    values = np.array(values)
    med = np.median(values)

    lower = values[values < med]
    upper = values[values > med]
    scale = np.median(upper) - np.median(lower)

    values = (values - med)
    if scale > 1e-9:
        values = values/scale
    
    return values.tolist()