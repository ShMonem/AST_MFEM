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
    L = csr_matrix(tril(A).astype(cp.float32))
    U = (A - L).astype(cp.float32)

    return U, L
