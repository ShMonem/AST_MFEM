import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

@ti.kernel
def pinvert(V: ti.template(), h: ti.template(), weight: ti.template(), D: ti.template()):
    n_vertices = V.shape[0]
    n_handles = h.shape[0]

    # Compute the distance between each vertex to each handle
    for i in range(n_vertices):
        for j in range(n_handles):
            dist_sq = 0.0
            for k in range(3):
                dist_sq += (V[i, k] - h[j, k]) ** 2
            D[i, j] = ti.sqrt(dist_sq)

    # Get the index of handles with min distance
    for i in range(n_vertices):
        min_dist_idx = 0
        for j in range(1, n_handles):
            if D[i, j] < D[i, min_dist_idx]:
                min_dist_idx = j
        weight[i] = min_dist_idx



"""
# Example usage
V_np = np.array([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]])

h_np = np.array([[1.5, 2.5, 3.5],
                 [5.5, 6.5, 7.5]])

V = ti.field(ti.f32, shape=(3, 3))
h = ti.field(ti.f32, shape=(2, 3))
weight = ti.field(ti.i32, shape=(3,))
D = ti.field(ti.f32, shape=(V_np.shape[0], h_np.shape[0]))


V.from_numpy(V_np)
h.from_numpy(h_np)

pinvert(V, h, weight, D)

weight_np = weight.to_numpy()

print("Weight:")
print(weight_np)
"""

