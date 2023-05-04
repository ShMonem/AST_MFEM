import taichi as ti
import numpy as np
ti.init()

N = 4
M = 4
At = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]], dtype=np.float32)
print(At)
triplets = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=2*N)
@ti.kernel
def fill(triplets: ti.types.ndarray(), At: ti.ext_arr()):
     for i in range(N):
        for j in range(M):
            triplets[i*M+j] = ti.Vector([i, j, At[i, j]], dt=ti.f32)
            print(triplets[i])
fill(triplets, At)
A = ti.linalg.SparseMatrix(n=N, m=N, dtype=ti.f32) 
A.build_from_ndarray(triplets)
print(At)
print(A)
