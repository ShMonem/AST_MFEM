import taichi as ti
import numpy as np
from scipy.sparse import kron, eye, find, csr_matrix
# gauss_seidel_solver is a python function, and is defined in GS
from GS import *
from build_U import *



placed_fields = False

@ti.kernel
def compute_residual(A: ti.template(), b: ti.template(), x: ti.template(), r: ti.template()):
    for i in range(A.shape[0]):
        sum = 0.0
        for j in range(A.shape[1]):
            sum += A[i, j] * x[j]
        r[i] = b[i] - sum

@ti.kernel
def update_sol_with_e(sol: ti.template(), U: ti.template(), e: ti.ext_arr()):
    for i in range(sol.shape[0]):
        sum = 0.0
        for j in range(U.shape[1]):
            sum += U[i, j] * e[j]
        sol[i] += sum
@ti.kernel
def restrict(Ub_l: ti.template(), res: ti.template(), red_res: ti.template()):
    ## Ub_l: ti.field
    # project error to lower resolution grid
    #red_res = Ub_l.transpose() @ res  ## works only wehen Ub_l is ti.linalg.SparseMatrix
    for j in range(Ub_l.shape[1]):
        sum = 0.0
        for i in range(Ub_l.shape[0]):
            sum += Ub_l[i, j] * res[i]
        red_res[j] += sum


def atom_add(x: ti.template(), y:ti.template()):
    ti.atomic_add(x, y)
def v_cycle(A_ti: ti.template(), b: ti.template(), UTAU, projA_solvers: ti.template(), Ub, l, \
            U:  ti.template(), L_solver: ti.template,\
                 itr: ti.i32, x_old: ti.template()):
    
    sol = gauss_seidel(U, L_solver, b, x_old, itr, x_old)  ## at beginning x = x_old, x_old = x_old
    
    r = ti.field(dtype=ti.f32, shape=(A_ti.shape[0],))
    compute_residual(A_ti, b, sol, r)

    red_res= ti.field(dtype=ti.f32, shape=(Ub[l].shape[1],))
    restrict(Ub[l], r, red_res)
    
    e = np.zeros(red_res.shape)
    if l == (len(Ub) -1 ): # reached the last reduction matrix in the list
        e = projA_solvers[l].solve(red_res)
    else:
        e = v_cycle(A_ti, b, UTAU, projA_solvers, Ub, l+1, \
              U, L_solver, itr, x_old, result)

    update_sol_with_e(sol, Ub[l], e)
   
    if l == 0:
        sol = gauss_seidel(U, L_solver, b, sol, itr, x_old)
    else:
        sol = gauss_seidel(UTAU[l], b, sol, itr, x_old)
    return sol


"""

# Example usage:
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


