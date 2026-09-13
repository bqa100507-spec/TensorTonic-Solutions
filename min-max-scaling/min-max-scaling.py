import numpy as np

def min_max_scaling(data: list) -> list:
    """
    Returns each data column scaled to the range from 0 through 1.
    """

    data = np.array(data)
    max = np.max(data, axis = 0)
    min = np.min(data, axis = 0)
    scale = max - min
    scale = np.where(scale < 1e-9, 1.0, scale)
    data = (data - min)/scale
    return data.tolist()