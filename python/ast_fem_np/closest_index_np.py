# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0
import numpy as np


def closest_index(V, P):
    # This looks quirky, but leverages the maximum parallelization that CuPy or numpy can offer by avoiding loops
    out_inds = np.argmin(
        np.linalg.norm(np.tile(P, (V.shape[0], 1)) - np.repeat(V, P.shape[0], axis=0),
                       axis=1, ord=2).reshape((-1, P.shape[0])),
        axis=1)

    return out_inds


def point_to_line_distance(V, handles, hier):
    x1 = handles[hier[hier != 0] - 1, :]
    x2 = handles[hier != 0, :]

    n = x1.shape[0]

    out_ind = ((np.sum((np.tile(x1, (V.shape[0], 1)) - np.repeat(V, n, axis=0)) * (
            np.tile(x2, (V.shape[0], 1)) - np.repeat(V, n, axis=0)), axis=1))).reshape((-1, n))
    out_ind = np.argmin(out_ind, axis=1)

    u, c = np.unique(hier, return_counts=True)
    dup = u[c > 1]
    for i in range(hier.shape[0]):
        if hier[i] in dup:
            out_ind[out_ind == i - 1] = hier[i] - 1

    return out_ind

# Unfinished - DO NOT USE!
def point_to_segment_distance(V, handles, hier):
    x1 = handles[hier[hier != 0] - 1, :]
    x2 = handles[hier != 0, :]

    n = x1.shape[0]

    out_ind = ((np.sum((np.tile(x1, (V.shape[0], 1)) - np.repeat(V, n, axis=0)) * (
            np.tile(x2, (V.shape[0], 1)) - np.repeat(V, n, axis=0)), axis=1))).reshape((-1, n))
    out_ind = np.argmin(out_ind, axis=1)

    u, c = np.unique(hier, return_counts=True)
    dup = u[c > 1]
    for i in range(hier.shape[0]):
        if hier[i] in dup:
            out_ind[out_ind == i - 1] = hier[i] - 1

    return out_ind
