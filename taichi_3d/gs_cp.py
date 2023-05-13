import numpy as np
from cupyx.scipy.sparse import csr_matrix, tril
from cupyx.scipy.sparse.linalg import spsolve_triangular
import cupy as cp
import cupyx as cpx


def gauss_seidel_cp(U, L, b, itr, sol):
    x = sol
    for i in range(itr):
        x_old = x
        x = b - U.dot(x_old)
        x = spsolve_triangular(L, x, lower=True)
    return x


def A_L_sum_U_cp(A):
    L = csr_matrix(tril(A).astype(cp.float64))
    U = (A - L).astype(cp.float64)

    return U, L


"""
## numerical example
A = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]], dtype=np.float32)
b = np.array([15, 10, 10, 10], dtype=np.float32)
sol = np.array([0, 0, 0, 0], dtype=np.float32)
itr = 25

#U, solver = A_L_sum_U(A)
x = ti.field(dtype=ti.f32, shape=(sol.shape[0],))
b_field = ti.field(dtype=ti.f32, shape=(b.shape[0],))

x_old = ti.field(dtype=ti.f32, shape=(x.shape[0],))
result = ti.field(dtype=ti.f32, shape=x.shape[0])

#fill_field(x, sol)
fill_field(x_old, sol)
fill_field(b_field, b)

U, L_solver = A_L_sum_U(A)
x = gauss_seidel(U, L_solver, b_field, x_old, itr, x_old)


print(x)

"""





