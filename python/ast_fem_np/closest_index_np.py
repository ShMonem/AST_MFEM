import numpy as np


def closest_index(V, P):
    # This looks quirky, but leverages the maximum parallelization that CuPy or numpy can offer by avoiding loops
    out_inds = np.argmin(
        np.linalg.norm(np.tile(P, (V.shape[0], 1)) - np.repeat(V, P.shape[0], axis=0),
                       axis=1, ord=2).reshape((-1, P.shape[0])),
        axis=1)

    return out_inds
