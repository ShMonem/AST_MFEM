import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# Set dimensions for matrices and vectors
n = 64
num_levels = 3

# Create Taichi fields
A = ti.field(ti.f32, (n, n), needs_grad=False)
b = ti.field(ti.f32, (n,), needs_grad=False)
UTAU = [ti.field(ti.f32, (n, n), needs_grad=False) for _ in range(num_levels)]
U = [ti.field(ti.f32, (n, n), needs_grad=False) for _ in range(num_levels)]
init = ti.field(ti.f32, (n,), needs_grad=False)
sol = ti.field(ti.f32, (n,), needs_grad=False)


@ti.func
def GS(A, b, n, x):
    for i in range(n):
        sum = 0.0
        for j in range(n):
            if i != j:
                sum += A[i, j] * x[j]
        x[i] = (b[i] - sum) / A[i, i]
    return x


@ti.func
def V_Cycle(A, b, UTAU, U, n, init, l):
    sol = GS(A, b, n, init)

    r = b - A @ sol
    rhs = U[l].transpose() @ r
    e = ti.field(ti.f32, (n,), needs_grad=False)
    e.fill(0)

    if l == num_levels - 1:
        e = ti.Matrix.solve(UTAU[l], rhs)
    else:
        e = V_Cycle(UTAU[l], rhs, UTAU, U, n, e, l + 1)

    sol = sol + U[l] @ e
    if l == 0:
        sol = GS(A, b, n, sol)
    else:
        sol = GS(UTAU[l - 1], b, n, sol)

    return sol


@ti.kernel
def v_cycle_kernel(A: ti.template(), b: ti.template(), UTAU: ti.template(), U: ti.template(), n: int, init: ti.template(), sol: ti.template()):
    sol = V_Cycle(A, b, UTAU, U, n, init, 0)


if __name__ == "__main__":
    # Fill A, b, UTAU, and U with example data
    A_np = np.random.rand(n, n).astype(np.float32)
    b_np = np.random.rand(n).astype(np.float32)
    UTAU_np = [np.random.rand(n, n).astype(np.float32) for _ in range(num_levels)]
    U_np = [np.random.rand(n, n).astype(np.float32) for _ in range(num_levels)]
    init_np = np.random.rand(n).astype(np.float32)
    sol_np = np.zeros(n, dtype=np.float32)

    A.from_numpy(A_np)
    b.from_numpy(b_np)
    for i in range(num_levels):
        UTAU[i].from_numpy(UTAU_np[i])
        U[i].from_numpy(U_np[i])
    init.from_numpy(init_np)

    v_cycle_kernel(A, b, UTAU, U, n, init, sol)

    sol_np =
