import math

def cosine_embedding_loss(x1: list, x2: list, label: int, margin: float) -> float:
    """
    Returns the cosine embedding loss as a float.
    """
    cos = sum(x*y for x, y in zip(x1, x2))
    if cos != 0:
        lx1 = math.sqrt(sum(x*x for x in x1))
        lx2 = math.sqrt(sum(x*x for x in x2))
        cos = cos / (lx1 * lx2)

    loss = 0
    if label == 1:
        loss = 1 - cos
    else:
        loss = max(0, cos - margin)

    return loss