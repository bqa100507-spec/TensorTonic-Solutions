import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)
    W = np.zeros(X.shape[1])
    b = 0.0

    for epoch in range(steps):
        old_b = b
        b = b - lr * np.mean(_sigmoid(X@W + old_b) - y)
        W = W - lr * np.transpose(X) @ (_sigmoid(X@W + old_b) - y)/(X.shape[0])

    W = W.tolist()
    return (W, b)