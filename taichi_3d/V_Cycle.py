import taichi as ti
import numpy as np

from GS import *

ti.init(arch=ti.cpu)

def V_Cycle(A, b, n, sol, level):
    sol = gauss_seidel_solver(A, b, n, sol)

    r = b - A @ sol
    rhs = U[level].transpose() @ r
    e = ti.Vector.zero(ti.f32, n)

    if level == l - 1:
        e = ti.Matrix.solve(UTAU[level], rhs)
    else:
        e = V_Cycle(UTAU[level], rhs, n, e, level + 1)

    sol = sol + U[level] @ e
    if level == 0:
        sol = gauss_seidel_solver(A, b, n, sol)
    else:
        sol = gauss_seidel_solver(UTAU[level - 1], b, n, sol)

    return sol


@ti.kernel
def v_cycle_kernel(A: ti.template(), b: ti.template(), n: int, init: ti.template(), l: int):
    sol[None] = V_Cycle(A, b, n, init, l)



if __name__ == "__main__":

    # Set dimensions for matrices and vectors
    n = 64
    l = 3

    # Create Taichi fields
    A = ti.field(ti.f32, (n, n), needs_grad=False)
    b = ti.field(ti.f32, (n,), needs_grad=False)
    UTAU = [ti.field(ti.f32, (n, n), needs_grad=False) for _ in range(l)]
    U = [ti.field(ti.f32, (n, n), needs_grad=False) for _ in range(l)]
    init = ti.Vector.field(n, dtype=ti.f32, shape=(), needs_grad=False)
    sol = ti.Vector.field(n, dtype=ti.f32, shape=(), needs_grad=False)

    # Fill A, b, UTAU, and U with example data
    A_np = np.random.rand(n, n).astype(np.float32)
    b_np = np.random.rand(n).astype(np.float32)
    UTAU_np = [np.random.rand(n, n).astype(np.float32) for _ in range(l)]
    U_np = [np.random.rand(n, n).astype(np.float32) for _ in range(l)]
    init_np = np.random.rand(n).astype(np.float32)
    sol_np = np.zeros(n, dtype=np.float32)

    A.from_numpy(A_np)
    b.from_numpy(b_np)
    for i in range(l):
        UTAU[i].from_numpy(UTAU_np[i])
        U[i].from_numpy(U_np[i])
    init.from_numpy(init_np)

    # Validate the length of the list U
    if len(U) != l:
        raise ValueError(f"The length of the list U ({len(U)}) must match the parameter l ({l}).")

    v_cycle_kernel(A, b, n, init, len(U))

    sol_np = sol[None].to_numpy()
    print(sol_np)
