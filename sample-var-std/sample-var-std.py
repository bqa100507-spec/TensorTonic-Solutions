import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x = np.array(x)
    mean = np.mean(x)
    var = np.sum((x - mean)**2) / (np.size(x) - 1)
    std = np.sqrt(var)
    var, std = var.item(), std.item()
    return (var, std)