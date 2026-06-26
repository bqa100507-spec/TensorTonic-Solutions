import numpy as np

def conv2d(X, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    X = np.asarray(X)
    W = np.asarray(W)
    b = np.asarray(b)

    N, c_in, Hin, Win = X.shape
    c_out, c_in, KH, KW = W.shape
    H_out = Hin - KH + 1
    W_out = Win - KW + 1
    y = np.zeros((N, c_out, H_out, W_out))
    
    for n in range(N):
        for c in range(c_out):
            for i in range(H_out):
                for j in range(W_out):
                    y[n, c, i, j] = np.sum(X[n, :, i:i+KH, j:j+KW] * W[c]) + b[c]

    return y