import taichi as ti
import numpy as np

@ti.kernel
def closest_index(V: ti.template(), P: ti.template(), weight: ti.template(), D: ti.template()):  
    for ii in range(V.shape[0]):
        for jj in ti.static(range(P.shape[0])):
            D[ii][jj] = (V[ii] - P[jj]).norm()
        
        min_dist_index = -1
        min_dist = float('inf')
        for jj in ti.static(range(P.shape[0])):
            if D[ii][jj] < min_dist:
                min_dist = D[ii][jj]
                min_dist_index = jj
        weight[ii] = min_dist_index
        #print(weight[ii])


"""
# Example usage:
Vt = np.random.rand(10, 3)
Pt = np.random.rand(5, 3)

ti.init(arch=ti.cpu)
V = ti.Vector.field(3, dtype=ti.f32, shape=Vt.shape[0])
P = ti.Vector.field(3, dtype=ti.f32, shape=Pt.shape[0])

V.from_numpy(Vt)
P.from_numpy(Pt)

weight = ti.field(ti.f32, shape=V.shape[0])
D = ti.Vector.field(P.shape[0], dtype=float, shape=V.shape[0])

closest_index(V, P, weight, D)
print(weight)
"""