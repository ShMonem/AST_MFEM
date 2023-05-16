import taichi as ti
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, isspmatrix

ti.init()

N = 4
M = 4
At = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]], dtype=np.float32)
At = csc_matrix(At)

print(At.nnz)
triplets = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=At.nnz)

@ti.kernel
def fill_tripletsSparse(triplets: ti.types.ndarray(), At_i: ti.ext_arr(), At_j: ti.ext_arr(), At_data: ti.ext_arr()):
     for i in range(At_i.shape[0]):
        triplets[i] = ti.Vector([At_i[i], At_j[i], At_data[i]], dt=ti.f32)
       # print(triplets[i])

rows, cols = At.nonzero()
fill_tripletsSparse(triplets, rows, cols, At.data,)
A = ti.linalg.SparseMatrix(n=N, m=N, dtype=ti.f32) 
A.build_from_ndarray(triplets)
print(At)
print(A)
