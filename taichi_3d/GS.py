import taichi as ti
import numpy as np
from scipy import sparse

ti.init(arch=ti.cpu)

@ti.kernel
def gauss_seidel_itr(U:  ti.template(), b: ti.template(), x: ti.template(),\
                 x_old: ti.template(), result: ti.template()):
    Udim1 = U.shape[0]
    Udim2 = U.shape[1]  
    xdim = x.shape[0]      
    for i in range(xdim):
        x_old[i] = x[i]
    
    for i in range(Udim1):
        res = 0.0
        for j in range(Udim2):
            res += U[i, j] * x[j]
        result[i] = b[i] - res

## python function
def gauss_seidel(A:  ti.ext_arr(), b: ti.template(), x: ti.template(),\
                 itr: ti.i32, x_old: ti.template(), \
                 result: ti.template()):
    
    U, solver = A_L_sum_U(A)
    
    x_itr = ti.field(dtype=ti.f32, shape=(x.shape[0],))
    #fill_field(x_itr, x)?
    for iter in range(itr):
        print(iter)
        gauss_seidel_itr(U, b, x_itr, x_old, result)
        xtemp = solver.solve(result) ## cannot be called in a taichi function: return self.solver.solve(b.to_numpy())
        fill_field(x_itr, xtemp)  ## xtemp is a numpy array. After each itr, we convert it to ti.template
                              ## so that it be used by the ti.kernel gauss_seidel_itr
    return x_itr

@ti.kernel
def fill_U(U: ti.template(), U_np: ti.ext_arr()):
    
    Udim1 = U_np.shape[0]
    Udim2 = U_np.shape[1]
    for i in range(Udim1):
        for j in range(Udim2):
            """
            try:
                U_np[i, j]
                U[i, j] = U_np[i, j]
            except Exception as e:
                print(f"An error occurred: {e}")
                print(i, j)
            """
            U[i, j] = U_np[i, j]
    
@ti.kernel
def fill_triplets(triplets: ti.types.ndarray(), A: ti.ext_arr()):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            triplets[i*A.shape[1]+ j] = ti.Vector([i, j, A[i,j]], dt=ti.f32)
            #print(triplets[i])
@ti.kernel
def fill_field(x: ti.template(),  sol: ti.ext_arr()):

    for i in range(sol.shape[0]):
        x[i] = sol[i]

def A_L_sum_U(A):  ## TODO: ti.kernel
    L_np = np.tril(A)
    U_np = A - L_np
    
    U = ti.field(dtype=ti.f32, shape=U_np.shape)
    fill_U(U, U_np)
    
    L = ti.linalg.SparseMatrix(n=L_np.shape[0], m=L_np.shape[1], dtype=ti.f32) 
    triplets = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=L_np.shape[0]*2)
    fill_triplets(triplets, L_np)
    L.build_from_ndarray(triplets)  ## not filled correctly

    solver = ti.linalg.SparseSolver()
    solver.compute(L)
    return U, solver

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

fill_field(x, sol)
fill_field(x_old, sol)
fill_field(b_field, b)

x = gauss_seidel(A, b_field, x, itr, x_old, result)

x_np = np.array([x[i] for i in range(sol.shape[0])])

print(x)
"""





