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
def update_sol_with_e(sol: ti.template(), U: ti.template(), e: ti.template()):
    for i in range(sol.shape[0]):
        sum = ti.cast(0.0, ti.f32)
        for j in range(U.shape[1]):
            sum += U[i, j] * e[j]
        sol[i] += sum

def v_cycle(A, b, UTAU, U, n, init, U_fields, r_fields, l=0):
    A_field = ti.field(dtype=ti.f32, shape=(A.shape[0], A.shape[1]))
    b_field = ti.field(dtype=ti.f32, shape=(b.shape[0],))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_field[i, j] = A[i, j]
        b_field[i] = b[i]

    sol = gauss_seidel_solver(A_field, b_field, n, init)

    sol_field = ti.field(dtype=ti.f32, shape=(A_field.shape[0],))
    for i in range(sol_field.shape[0]):
        sol_field[i] = sol[i]
    
    r = ti.field(dtype=ti.f32, shape=(A_field.shape[0],))
    compute_residual(A_field, b_field, sol_field, r)

    global placed_fields

    if not placed_fields:
        for i in range(len(U)):
            ti.root.dense(ti.i, U[i].shape[0]).place(U_fields[i])
            ti.root.dense(ti.i, A.shape[0]).place(r_fields[i])
        placed_fields = True
        
    # Initialize U_field and r_field with U[l] and r
    initialize_U_and_r(U_fields[l], r_fields[l], U[l], r)

    # Replace U[l][j, i] * r[j] with U_field[j, i] * r_field[j]
    rhs = ti.field(dtype=ti.f32, shape=(U[l].shape[1],))
    for i in range(rhs.shape[0]):
        sum = ti.cast(0.0, ti.f32)
        for j in range(U[l].shape[0]):
            sum += U_field[j, i] * r_field[j]
        rhs[i] = sum
    
    e = np.zeros(rhs.shape[0], dtype=np.float32)
    
    if l == len(U) - 1:
        e = gauss_seidel_solver(UTAU[l], rhs, n, e)
    else:
        e = v_cycle(UTAU[l], rhs, UTAU, U, n, e, l + 1)
    
    update_sol_with_e(sol, U[l], e)

    if l == 0:
        sol = gauss_seidel_solver(A_field, b_field, n, sol_field)
    else:
        sol = gauss_seidel_solver(UTAU[l - 1], b_field, n, sol_field)

    return sol

# Define other necessary functions and data here

# Example usage:
A = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]], dtype=np.float32)
b = np.array([15, 10, 10, 10], dtype=np.float32)
sol_init = np.array([0, 0, 0, 0], dtype=np.float32)
n = 25

UTAU = [np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]], dtype=np.float32)]
U = [np.eye(4, dtype=np.float32)]

# create Taichi fields
U_fields = [ti.Matrix.field(n=U[i].shape[1], m=U[i].shape[0], dtype=ti.f32) for i in range(len(U))]
r_fields = [ti.field(dtype=ti.f32, shape=(A.shape[0],)) for _ in range(len(U))]


sol = v_cycle(A, b, UTAU, U, n, sol_init, U_fields, r_fields, l=0)
print(sol)

