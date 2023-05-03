import numpy as np
import scipy.sparse as sp


def linear_tet3dmesh_corot_ds2(mu, la, T: np.ndarray):
    n = T.shape[0]
    H123 = np.array([[la + 2 * mu, la, la], [la, la + 2 * mu, la], [la, la, la + 2 * mu]])
    H456 = np.array([[4*mu, 0, 0], [0, 4*mu, 0], [0, 0, 4*mu]])
    block = sp.block_diag((H123, H456))

    def repeat(num, b):
        H = list()
        for i in range(num):
            H.append(b)

        return H

    return sp.block_diag(repeat(n, block))


def linear_tet3dmesh_corot_ds(mu, la, T: np.ndarray):
    n = T.shape[0]
    vec = -np.array([3*la+2*mu, 3*la+2*mu, 3*la+2*mu, 0, 0, 0])

    return np.tile(vec.T, n)
