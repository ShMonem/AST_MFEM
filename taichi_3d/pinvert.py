import taichi as ti
import numpy as np

@ti.kernel
def pinvert(ndim: int, V: ti.template(), h: ti.template()):
    for ii in range(h.shape[0]):
        Dii = 0.0
        for j in range(ndim):
            Dii += (V - h[ii][:])**2[j]
        D[:, ii] = ti.sqrt(Dii)
    return D

# Define the input arrays
V = np.random.rand(100, 3)
h = np.random.rand(10, 3)
Vn = V.shape[0]
hn = h.shape[0]
ndim = 3
# Create Taichi arrays from the input arrays
ti.init(arch=ti.cpu)
V_t = ti.Vector.field(3, dtype=ti.f32, shape=Vn)
h_t = ti.Vector.field(3, dtype=ti.f32, shape=hn)
V_t.from_numpy(V)
h_t.from_numpy(h)

# Compute the output using the pinvert function

D = ti.field(dtype=ti.f32, shape=(Vn, hn))
D = pinvert(ndim, V_t, h_t)

# Convert the output to a NumPy array and print it
D_np = D.to_numpy()
print(D_np)

