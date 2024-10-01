# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

import numpy as np


def pinvert(V, h):
    weight = np.argmin(
        np.linalg.norm(np.tile(h, (V.shape[0], 1)) - np.repeat(V, h.shape[0], axis=0),
                       axis=1, ord=2).reshape((-1, h.shape[0])).T,
        axis=1)

    return weight
