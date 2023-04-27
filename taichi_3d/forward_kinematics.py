import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

vecl = ti.Matrix.field(1, 100000, dtype=ti.f32, shape=())

@ti.func
def compute_rotation(alpha, beta, gamma):
    R = ti.Vector([
        ti.cos(alpha) * ti.cos(beta),
        ti.sin(alpha) * ti.cos(beta),
        -ti.sin(beta),
        ti.cos(alpha) * ti.sin(beta) * ti.sin(gamma) - ti.sin(alpha) * ti.cos(gamma),
        ti.sin(alpha) * ti.sin(beta) * ti.sin(gamma) + ti.cos(alpha) * ti.cos(gamma),
        ti.cos(beta) * ti.sin(gamma),
        ti.cos(alpha) * ti.sin(beta) * ti.cos(gamma) + ti.sin(alpha) * ti.sin(gamma),
        ti.sin(alpha) * ti.sin(beta) * ti.cos(gamma) - ti.cos(alpha) * ti.sin(gamma),
        ti.cos(beta) * ti.cos(gamma)
    ])
    return R

@ti.kernel
def forward_kinematics(handles: ti.ext_arr(), hier: ti.ext_arr(), eulers: ti.ext_arr(), T: ti.ext_arr(), new_handles: ti.ext_arr(), newR: ti.ext_arr()):
    n = handles.shape[0]
    
    for i in range(n):
        lo = compute_rotation(eulers[i, 0], eulers[i, 1], eulers[i, 2])
        
        if hier[i, 0] == 0:
            for j in range(T.shape[1]):
                T[i, j] = lo[j]
            for k in range(handles.shape[1]):
                new_handles[i, k] = handles[i, k]
        else:
            for j in range(T.shape[1]):
                newR[i - 1, j] = T[int(hier[i, 0] - 1), j]
            
            Tp = ti.Matrix([[T[int(hier[i, 0] - 1), 0], T[int(hier[i, 0] - 1), 1], T[int(hier[i, 0] - 1), 2]],
                            [T[int(hier[i, 0] - 1), 3], T[int(hier[i, 0] - 1), 4], T[int(hier[i, 0] - 1), 5]],
                            [T[int(hier[i, 0] - 1), 6], T[int(hier[i, 0] - 1), 7], T[int(hier[i, 0] - 1), 8]]])
            
            
            for l in range(handles.shape[1]):
                vecl[0, l] = (handles[i, l] - handles[int(hier[i, 0] - 1), l])
            for l in range(handles.shape[1]):
                vec = new_handles[int(hier[i, 0] - 1), l] + Tp @ vecl
                new_handles[i, l] = vec

            lo_matrix = ti.Matrix([[lo[0], lo[1], lo[2]], [lo[3], lo[4], lo[5]], [lo[6], lo[7], lo[8]]])
            temp = Tp @ lo_matrix
            for j in range(T.shape[1]):
                T[i, j] = temp[j]




# Example input data
handles = np.array([
    [0.0, 0.0, 0.0],  # Root joint
    [1.0, 0.0, 0.0],  # First child joint
    [2.0, 0.0, 0.0]   # Second child joint
])

hierarchy = np.array([
    [0, 0],
    [1, 0],
    [2, 1]
])

eulers = np.array([
    [0.0, 0.0, 0.0],      # Root joint
    [0.0, np.pi/2, 0.0],  # First child joint - rotate 90 degrees around Y-axis
    [0.0, 0.0, np.pi/2]   # Second child joint - rotate 90 degrees around Z-axis
])
# Create the T array outside the kernel
n = handles.shape[0]
T = np.tile(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]), (n, 1))
# Output arrays

new_handles = np.zeros_like(handles)
newR = np.zeros((handles.shape[0] - 1, 9))


# Run the forward_kinematics function
forward_kinematics(handles, hierarchy, eulers, T, new_handles, newR)

print("New handles:")
print(new_handles)
print("\nNew rotations:")
print(newR)
