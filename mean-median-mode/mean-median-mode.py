import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.array(x)
    x = np.sort(x)
    mean = np.mean(x)
    med = np.median(x)
    count = Counter(x)
    common = count.most_common(1)
    mode = common[0][0]
    mean, med = mean.item(), med.item()
    return (mean, med, mode)