import numpy as np


def pinvert_np(V, h):
    weight = np.argmin(
        np.linalg.norm(np.tile(h, (V.shape[0], 1)) - np.repeat(V, h.shape[0], axis=0),
                       axis=1, ord=2).reshape((-1, h.shape[0])).T,
        axis=1)

    return weight
