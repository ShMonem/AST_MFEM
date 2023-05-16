import numpy as np
from scipy.sparse import csr_matrix, tril
from scipy.sparse.linalg import spsolve_triangular


def gauss_seidel(U, L, b, itr, sol):
    x = sol
    for i in range(itr):
        # x_old = x.copy()
        x_old = x

        # x = L \ b - U x_old
        x = b - U.dot(x_old)
        x = spsolve_triangular(L, x, lower=True)

    return x


def A_L_sum_U(A):
    L_np = tril(A).astype(np.float64)
    U_np = (A - L_np).astype(np.float64)
    # convert to sparse format for fast triplets fill
    L_np = csr_matrix(L_np)
    U_np = csr_matrix(U_np)

    return U_np, L_np
