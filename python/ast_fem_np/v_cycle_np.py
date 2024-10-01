# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

from scipy.sparse.linalg import spsolve
import numpy as np
from python.ast_fem_np.gauss_seidel_np import gauss_seidel, A_L_sum_U
from time import time


def v_cycle(A, U, L, b, UTAU, Ub, l, itr, x_init, debug=False):
    start = time()
    sol = gauss_seidel(U, L, b, itr, x_init)
    end = time()
    if debug:
        print("GS took {0} seconds".format(end-start))

    # compute residual
    r = b - A.dot(sol)
    # restrict to lower res grid
    red_res = Ub[l].T.dot(r)
    e = np.zeros((red_res.shape[0], 1))
    if l == (len(Ub) -1 ): # reached the last reduction matrix in the list
        start = time()
        e = spsolve(UTAU[l], red_res)
        end = time()
        if debug:
            print("SPSolve took {0} seconds".format(end-start))
    else:
        e = v_cycle(A, U, L, b, UTAU, Ub, l+1, itr, e)

    sol = sol + Ub[l].dot(e)
    if l == 0:
        sol = gauss_seidel( U, L, b, itr, sol)
    else:
        U_utau , L_utau = A_L_sum_U(UTAU[l])
        sol = gauss_seidel(U_utau , L_utau, b, itr, sol)
    return sol
