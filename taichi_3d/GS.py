import taichi as ti
import numpy as np
from scipy import sparse

ti.init(arch=ti.cpu)

#@ti.kernel
def gauss_seidel(b: ti.template(), x: ti.template(),\
                 itr: ti.i32, xdim: ti.i32, x_old: ti.template(), \
                 result: ti.template(), solver: ti.template()):
    

    for iter in range(itr):
        for i in range(xdim):
            x_old[i] = x[i]
        """
        for i in range(Udim1):
            res = 0.0
            for j in range(Udim2):
                res += U[i, j] * x[j]
            result[i] = b[i] - res
        """
        x = solver.solve(result) ## cannot be called in a taichi function: return self.solver.solve(b.to_numpy())


def gauss_seidel_solver(A, b, n, sol):
    A_field = ti.field(dtype=ti.f32, shape=A.shape)
    A_field.from_numpy(A)

    tribletsNum = sparse.csr_matrix(A).nnz

    A_sparse_ti = ti.linalg.SparseMatrix(n=A.shape[0], m=A.shape[1], dtype=ti.f32) 
    triplets = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=A.shape[0])

    def fill_triplets(triplets: ti.types.ndarray(), A: ti.ext_arr()):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                triplets[i] = ti.Vector([i, j, A[i,j]], dt=ti.f32)
    
    fill_triplets(triplets, A)

    A_sparse_ti.build_from_ndarray(triplets)
    """
    L_np = np.tril(A)
    U_np = A - L_np
    
    L = ti.field(dtype=ti.f32, shape=L_np.shape)
    U = ti.field(dtype=ti.f32, shape=U_np.shape)
    
    for i in range(L_np.shape[0]):
        for j in range(L_np.shape[1]):
            L[i, j] = L_np[i, j]
            U[i, j] = U_np[i, j]

    print(L_np.shape)
    ti_field = ti.field(dtype=ti.f32, shape=L_np.shape)
    ti_field.from_numpy(L_np)
    """
    x = ti.field(dtype=ti.f32, shape=(sol.shape[0],))
    
    for i in range(sol.shape[0]):
        x[i] = sol[i]

    b_field = ti.field(dtype=ti.f32, shape=(b.shape[0],))
    for i in range(b.shape[0]):
        b_field[i] = b[i]
    
    result = ti.field(dtype=ti.f32, shape=x.shape[0])
    x_old = ti.field(dtype=ti.f32, shape=(x.shape[0],))

    for i in range(x.shape[0]):
        x_old[i] = x[i]


    solver = ti.linalg.SparseSolver()
    solver.compute(A_sparse_ti)

    gauss_seidel(b_field, x, itr, x.shape[0], x_old, result, solver)

    x_np = np.array([x[i] for i in range(sol.shape[0])])
    return x_np


"""
A = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]], dtype=np.float32)
b = np.array([15, 10, 10, 10], dtype=np.float32)
sol = np.array([0, 0, 0, 0], dtype=np.float32)
itr = 25

x = gauss_seidel_solver(A, b, itr, sol)
print(x)
"""
