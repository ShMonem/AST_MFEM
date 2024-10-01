# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

import numpy as np
from scipy.sparse import diags
import scipy.sparse as sp


def linear_tet3dmesh_arap_ds(V, T, s, mu):
    mu = np.repeat(mu, 6, axis=0)
    n = T.shape[0]
    g = 2 * mu * (s - np.tile(np.array([1, 1, 1, 0, 0, 0]), (n, 1)).reshape(6 * n, 1)).reshape(6 * n, 1)
    return g


def linear_tet3dmesh_arap_ds2(V, T, s, mu):
    mu = np.repeat(mu, 6, axis=0)
    n = T.shape[0]
    repeated = mu * np.tile(np.array([1, 1, 1, 2, 2, 2]).reshape(-1, 1), (n, 1))
    H = 2 * diags(repeated.ravel(), 0, shape=(6 * n, 6 * n))
    return H


def linear_tet3dmesh_corot_ds2(mu, la, T: np.ndarray):
    n = T.shape[0]
    H = list()

    for i in range(n):
        H123 = np.array([[la[i, 0] + 2 * mu[i, 0], la[i, 0], la[i, 0]], [la[i, 0], la[i, 0] + 2 * mu[i, 0], la[i, 0]],
                         [la[i, 0], la[i, 0], la[i, 0] + 2 * mu[i, 0]]])
        H456 = np.array([[4 * mu[i, 0], 0, 0], [0, 4 * mu[i, 0], 0], [0, 0, 4 * mu[i, 0]]])
        block = sp.block_diag((H123, H456))
        H.append(block)

    return sp.block_diag(H)


def linear_tet3dmesh_corot_ds(mu, la, T: np.ndarray):
    n = T.shape[0]
    k = -2 * np.repeat(mu, 3, axis=1) - 3 * np.repeat(la, 3, axis=1)
    grad = np.hstack((k, np.zeros_like(k))).reshape(-1)

    return np.expand_dims(grad, axis=1)
