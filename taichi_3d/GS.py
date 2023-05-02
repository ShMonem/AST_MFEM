import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

@ti.kernel
def gauss_seidel(L: ti.template(), U: ti.template(), b: ti.template(), x: ti.template(), n: int):
    
    for iter in range(n):
        for i in range(x.shape[0]):
            sum = ti.cast(0.0, ti.f32)
            for j in range(x.shape[0]):
                if i != j:
                    sum += L[i, j] * x[j]
            x[i] = (b[i] - sum) / L[i, i]

def gauss_seidel_solver(A, b, n, sol):
    # create an ndarray fron the tichi field A:
    A_np = np.zeros((A.shape[0], A.shape[1]), dtype=np.float32)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_np[i, j] = A[i, j]

    L_np = np.tril(A_np)
    U_np = A_np - L_np
    
    L = ti.field(dtype=ti.f32, shape=(A.shape[0], A.shape[1]))
    U = ti.field(dtype=ti.f32, shape=(A.shape[0], A.shape[1]))

    @ti.kernel
    def initialize_L_and_U(L: ti.template(), U: ti.template(), L_np: ti.ext_arr(), U_np: ti.ext_arr()):
        for i, j in L:
            L[i, j] = L_np[i, j]
            U[i, j] = U_np[i, j]

    initialize_L_and_U(L, U, L_np, U_np)

    x = ti.field(dtype=ti.f32, shape=(sol.shape[0],))
    for i in range(sol.shape[0]):
        x[i] = sol[i]

    b_field = ti.field(dtype=ti.f32, shape=(b.shape[0],))
    for i in range(b.shape[0]):
        b_field[i] = b[i]

    gauss_seidel(L, U, b_field, x, n)

    x_np = np.array([x[i] for i in range(sol.shape[0])])
    return x_np


"""
A = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]], dtype=np.float32)
b = np.array([15, 10, 10, 10], dtype=np.float32)
sol = np.array([0, 0, 0, 0], dtype=np.float32)
n = 25

x = gauss_seidel_solver(A, b, n, sol)
print(x)
"""

