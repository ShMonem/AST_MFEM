import cupy as cp
from cupyx.scipy.sparse.linalg import spsolve
from python.ast_fem_cp.gs_cp import gauss_seidel, A_L_sum_U
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
    e = cp.zeros((red_res.shape[0], 1))
    if l == (len(Ub) -1 ): # reached the last reduction matrix in the list
        start = time()
        e = spsolve(UTAU[l], red_res)
        end = time()
        if debug:
            print("SPSolve took {0} seconds".format(end-start))
    else:
        e = v_cycle_cp(A, U, L, b, UTAU, Ub, l+1, itr, e)

    sol = sol + Ub[l].dot(e)
    if l == 0:
        sol = gauss_seidel( U, L, b, itr, sol)
    else:
        U_utau , L_utau = A_L_sum_U(UTAU[l])
        sol = gauss_seidel(U_utau , L_utau, b, itr, sol)
    return sol

"""

# Example usage:
from build_U import *
A = np.array([[10, -1, 0, 0], [-1, 4, -18, 0], [5, -1, 42, -1], [0, 0, -1, 3]], dtype=np.float32)
b = np.array([15, 10, 10, 10], dtype=np.float32)
sol_init = np.array([0, 0, 0, 0], dtype=np.float32)
n = 25
U1t = np.eye(b.shape[0], dtype=np.float32)[:, :3]
NN = np.zeros(U1t.shape[1], dtype=np.float32)
Ut = []  # list of numpy reduction matrices
Ut.append(csr_matrix(U1t)) 

Ub = [] # list of taichi reduction matrices
Ub_shapes = []
UTAU = []
UTAU_shapes =[]
projA_solvers = []## list of UTAU solvers

U_toTichi(Ut, A, Ub, UTAU, projA_solvers, NN)


A_field = ti.field(dtype=ti.f32, shape=(A.shape[0], A.shape[1]))
b_field = ti.field(dtype=ti.f32, shape=(b.shape[0],))

fill_U(A_field, A)
fill_field(b_field, b)

sol_init = np.zeros(b.shape, dtype=np.float32)
x = ti.field(dtype=ti.f32, shape=(sol_init.shape[0],))
x_old = ti.field(dtype=ti.f32, shape=(x.shape[0],))
result = ti.field(dtype=ti.f32, shape=x.shape[0])
fill_field(x, sol_init)


l =0
itr = 24
U, L_solver = A_L_sum_U(A)

sol = v_cycle(A_field, b_field, UTAU, projA_solvers, Ub, l, \
              U, L_solver, itr, x_old)
print(sol)
"""


