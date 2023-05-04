import taichi as ti
import numpy as np

# gauss_seidel_solver is a python function, and is defined in GS
from GS import *

ti.init()


placed_fields = False

@ti.kernel
def compute_residual(A: ti.template(), b: ti.template(), x: ti.template(), r: ti.template()):
    for i in range(A.shape[0]):
        sum = ti.cast(0.0, ti.f32)
        for j in range(A.shape[1]):
            sum += A[i, j] * x[j]
        r[i] = b[i] - sum

@ti.kernel
def update_sol_with_e(sol: ti.template(), U: ti.template(), e: ti.ext_arr()):
    for i in range(sol.shape[0]):
        sum = ti.cast(0.0, ti.f32)
        for j in range(U.shape[1]):
            sum += U[i, j] * e[j]
        sol[i] += sum

def restrict(Ub_l: ti.template(), res: ti.template(), red_res: ti.template()):
    ## Ub_l: class 'taichi.linalg.sparse_matrix.SparseMatrix'
    # project error to lower resolution grid
    red_res = Ub_l.transpose() @ res

def atom_add(x: ti.template(), y:ti.template()):
    ti.atomic_add(x, y)
def v_cycle(A_ti: ti.template(), b: ti.template(), UTAU, Ub, l, \
            A:  ti.template(), x: ti.template(),\
                 itr: ti.i32, x_old: ti.template(), \
                 result: ti.template()):
    
    sol = gauss_seidel(A, b, x, itr, x_old, result)
    
    r = ti.field(dtype=ti.f32, shape=(A.shape[0],))
    compute_residual(A_ti, b, sol, r)

    red_res= ti.field(dtype=ti.f32, shape=(Ub[l].shape[0],))
    restrict(Ub[l], r, red_res)

    e = np.zeros(red_res.shape)

    if l == (len(Ub) -1 ): # reached the last reduction matrix in the list
        e = basis_solvers[l].solve(red_res)
    else:
        e = v_cycle(A_ti, b, UTAU, Ub, l+1, \
              A, x, itr, x_old, result, UTAU_shapes)
      
    atom_add(sol, Ub[l]@ e)  ## TODO: has "sol" been updated?


    if l == 0:
        sol = gauss_seidel(A, b, sol, itr, x_old, result)
    else:
        sol = gauss_seidel(UTAU[l], b, sol, itr, x_old, result)
    return sol


"""
# Example usage:
A = np.array([[10, -1, 0, 0], [-1, 4, -18, 0], [5, -1, 42, -1], [0, 0, -1, 3]], dtype=np.float32)
b = np.array([15, 10, 10, 10], dtype=np.float32)
sol_init = np.array([0, 0, 0, 0], dtype=np.float32)
n = 25
U1t = np.eye(b.shape[0], dtype=np.float32)
Ut = []  # list of numpy reduction matrices
Ut.append(U1t) 

Ub = [] # list of taichi reduction matrices
Ub_shapes = []
for l in range(len(Ut)):
    U_l = ti.linalg.SparseMatrix(n=Ut[l].shape[0], m=Ut[l].shape[1], dtype=ti.f32) 
    triplets_u_l = ti.Vector.ndarray(n=3, dtype=ti.f32, shape= Ut[l].shape[0])
    fill_triplets(triplets_u_l, Ut[l])
    U_l.build_from_ndarray(triplets_u_l) ## TODO: not filled properly
    Ub.append(U_l)  # basis list
    Ub_shapes.append(Ut[l].shape)

UTAU = []
UTAU_shapes =[]
for l in range(len(Ut)):
    UTAUt_l = Ut[l].T @ A @ Ut[l] 
    UTAU_l = ti.linalg.SparseMatrix(n=UTAUt_l.shape[0], m=UTAUt_l.shape[1], dtype=ti.f32)
    triplets_utau_l = ti.Vector.ndarray(n=3, dtype=ti.f32, shape= UTAUt_l.shape[0])
    fill_triplets(triplets_utau_l, UTAUt_l)
    UTAU_l.build_from_ndarray(triplets_utau_l) ## TODO: not filled properly
    UTAU.append(UTAU_l) # projected basis list
    UTAU_shapes.append(UTAUt_l.shape)

basis_solvers = [] ## list of UTAU solvers
for l in range(len(UTAU)):
    solver_l = ti.linalg.SparseSolver()
    solver_l.compute(UTAU[l])
    basis_solvers.append(solver_l)

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
sol = v_cycle(A_field, b_field, UTAU, Ub, l, \
              A, x, itr, x_old, result)
print(sol)

"""
