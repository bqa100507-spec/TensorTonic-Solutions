import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """

    a = np.array(a)
    b = np.array(b)
    dot = a@b
    if dot == 0:
        return 0.0
    cos = (a@b)/(np.linalg.norm(a) * np.linalg.norm(b))
    return cos.item()