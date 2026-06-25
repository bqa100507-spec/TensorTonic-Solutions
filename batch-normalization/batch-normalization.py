import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)
    if x.ndim == 2:
        mui = np.mean(x, axis=0, keepdims=True)
        sigma = np.mean((x - mui)**2, axis=0, keepdims=True)
        xhat = (x - mui)/np.sqrt(sigma + eps)
        return gamma*xhat + beta
    else:
        mui = np.mean(x, axis=(0, 2, 3), keepdims=True)
        sigma = np.mean((x - mui)**2, axis=(0, 2, 3), keepdims=True)
        xhat = (x - mui)/np.sqrt(sigma + eps)
        gamma = gamma[np.newaxis, :, np.newaxis, np.newaxis]
        beta = beta[np.newaxis, :, np.newaxis, np.newaxis]
        return gamma*xhat + beta
    return x