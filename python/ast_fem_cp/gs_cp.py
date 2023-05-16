from cupyx.scipy.sparse import csr_matrix, tril
from cupyx.scipy.sparse.linalg import spsolve_triangular
import cupy as cp


def gauss_seidel_cp(U, L, b, itr, sol):
    x = sol
    for i in range(itr):
        x_old = x
        x = b - U.dot(x_old)
        x = spsolve_triangular(L, x, lower=True)
    return x


def A_L_sum_U_cp(A):
    L_np = tril(A).astype(cp.float64)
    U_np = (A - L_np).astype(cp.float64)
    # convert to sparse format for fast triplets fill
    L_np = csr_matrix(L_np)
    U_np = csr_matrix(U_np)

    return  U_np, L_np
