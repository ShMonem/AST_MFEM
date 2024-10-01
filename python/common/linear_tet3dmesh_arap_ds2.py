# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

import numpy as np
from scipy.sparse import diags

def linear_tet3dmesh_arap_ds2(V, T, s, mu):

    n = T.shape[0]
    repeated = np.tile(np.array([1, 1, 1, 2, 2, 2]).reshape(-1, 1), (n, 1))
    H = 2 * mu * diags(repeated.ravel(), 0, shape=(6 * n, 6 * n))
    return H