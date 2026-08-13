import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    x = np.array(x)
    if rng is None:
        rng = np.random.default_rng()

    bs_mean = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        bss = rng.choice(x, size=x.shape)
        bs_mean[i] = np.mean(bss)

    alpha = (1 - ci)/2
    lower = np.quantile(bs_mean, alpha)
    upper = np.quantile(bs_mean, 1-alpha)
    lower, upper = lower.item(), upper.item()
    return (bs_mean, lower, upper)
    
