import numpy as np
import cupy as cp
import config


def closest_index_cp(V, P):
    # This looks quirky, but leverages the maximum parallelization that CuPy or numpy can offer by avoiding loops
    out_inds = cp.argmin(
        cp.linalg.norm(cp.tile(P, (V.shape[0], 1)) - cp.repeat(V, P.shape[0], axis=0),
                       axis=1, ord=2).reshape((-1, P.shape[0])),
        axis=1)

    return out_inds


def closest_index_np(V, P):
    # This looks quirky, but leverages the maximum parallelization that CuPy or numpy can offer by avoiding loops
    out_inds = np.argmin(
        np.linalg.norm(np.tile(P, (V.shape[0], 1)) - cp.repeat(V, P.shape[0], axis=0),
                       axis=1, ord=2).reshape((-1, P.shape[0])),
        axis=1)

    return out_inds
