import numpy as np

def linear_tet3dmesh_arap_ds(V, T, s, mu):

    n = T.shape[0]
    g = 2 * mu * (s - np.tile(np.array([1, 1, 1, 0, 0, 0]), (n, 1)).reshape(6*n, 1)).reshape(6*n, 1)
    return g
