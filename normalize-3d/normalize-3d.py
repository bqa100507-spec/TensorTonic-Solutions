import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v = np.array(v, dtype=float)
    if np.ndim(v) == 1:
        norm_v = np.sqrt(np.sum(v**2))
        if norm_v < 1e-10:
            return 0
        return v/norm_v
    else:
        norm_v = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
        mask = (norm_v > 1e-10).flatten()
        v[mask] = v[mask] / norm_v[mask]
        return v