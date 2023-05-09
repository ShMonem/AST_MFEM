import taichi as ti
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, isspmatrix
from scipy.sparse.linalg import spsolve_triangular

#ti.init(arch=ti.cpu)

@ti.kernel
def gauss_seidel_itr(U:  ti.template(), b: ti.template(), x: ti.template(),\
                 x_old: ti.template()):
    Udim1 = U.shape[0]
    Udim2 = U.shape[1]  
    xdim = x.shape[0]      
    for i in range(xdim):
        x_old[i] = x[i]   # x_old = x
    for i in range(Udim1):
        res = 0.0
        for j in range(Udim2):
            res += U[i, j] * x_old[j]   # x = b - U x_old
        x[i] = b[i] - res


## python function
def gauss_seidel_ti(U:  ti.ext_arr(), solver : ti.template(), b: ti.template(), x: ti.template(),\
                 itr: ti.i32, x_old: ti.template()):
    for iter in range(itr):
        print(iter)
        gauss_seidel_itr(U, b, x, x_old)
        # x = L \ b - U x_old
        xtemp = solver.solve(x).astype(np.float32) ## cannot be called in a taichi function: return self.solver.solve(b.to_numpy())
        fill_field(x, xtemp)  ## xtemp is a numpy array. After each itr, we convert it to ti.template
                              ## so that it be used by the ti.kernel gauss_seidel_itr
    return x

def gauss_seidel_py(U_field: ti.template(), U:  ti.ext_arr(), L:  ti.ext_arr(), b: ti.template(),\
                 itr: ti.i32, x: ti.template(), x_old: ti.template()):
    for iter in range(itr):
        print(iter)
        gauss_seidel_itr(U_field, b, x, x_old)
        # x = L \ b - U x_old
        xtemp = spsolve_triangular(L, x.to_numpy(), lower=True).astype(np.float32)
        fill_field(x, xtemp)
    return x


@ti.kernel
def fill_U(U: ti.template(), U_np: ti.ext_arr()):
    
    Udim1 = U_np.shape[0]
    Udim2 = U_np.shape[1]
    for i in range(Udim1):
        for j in range(Udim2):
            U[i, j] = U_np[i, j]
@ti.kernel
def fill_U_Sparse(U: ti.template(), U_np_i: ti.ext_arr(), U_np_j: ti.ext_arr(), U_np_data: ti.ext_arr()):

    for i in range(U_np_i.shape[0]):
        U[U_np_i[i], U_np_j[i]] = U_np_data[i]


@ti.kernel
def fill_triplets(triplets: ti.types.ndarray(), A: ti.ext_arr()):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            triplets[i*A.shape[1]+ j] = ti.Vector([i, j, A[i,j]], dt=ti.f32)
            #print(triplets[i])

#@ti.kernel
def fill_tripletsSparse(triplets: ti.types.ndarray(), At_i: ti.ext_arr(), At_j: ti.ext_arr(), At_data: ti.ext_arr()):
     for i in range(At_i.shape[0]):
        triplets[i] = ti.Vector([int(At_i[i]), int(At_j[i]), At_data[i]], dt=[ti.i32, ti.i32, ti.f32])
       # print(triplets[i])


@ti.kernel
def fill_field(x: ti.template(),  sol: ti.ext_arr()):
    for i in range(sol.shape[0]):
        x[i] = sol[i]


def A_L_sum_U_ti(A):  ## TODO: ti.kernel
    L_np = np.tril(A).astype(np.float32)
    U_np = (A - L_np).astype(np.float32)
    
    # convert to sparse format for fast triplets fill
    L_np = csr_matrix(L_np)
    U_np = csr_matrix(U_np)
    
    rows, cols = L_np.nonzero()
    rows_, cols_ = U_np.nonzero()

    U = ti.field(dtype=ti.f32, shape=U_np.shape)
    fill_U_Sparse(U, rows_, cols_, U_np.data)

    L = ti.linalg.SparseMatrix(n=L_np.shape[0], m=L_np.shape[1], dtype=ti.f32)
    triplets = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=L_np.nnz)
    fill_tripletsSparse(triplets, rows, cols, L_np.data)
    
    L.build_from_ndarray(triplets)  

    solver = ti.linalg.SparseSolver()
    solver.compute(L)
    return U, solver

def A_L_sum_U_py(A):  ## TODO: ti.kernel
    L_np = np.tril(A).astype(np.float32)
    U_np = (A - L_np).astype(np.float32)
    # convert to sparse format for fast triplets fill
    L_np = csr_matrix(L_np)
    U_np = csr_matrix(U_np)
    
    U_field = ti.field(dtype=ti.f32, shape=U_np.shape)
    rows_, cols_ = U_np.nonzero()
    fill_U_Sparse(U_field, rows_, cols_, U_np.data)
    return  U_field, U_np, L_np


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





