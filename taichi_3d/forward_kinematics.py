import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

@ti.func
def compute_rotation(alpha, beta, gamma):
    R = ti.Matrix([
        [ti.cos(alpha) * ti.cos(beta), ti.sin(alpha) * ti.cos(beta), -ti.sin(beta)],
        [ti.cos(alpha) * ti.sin(beta) * ti.sin(gamma) - ti.sin(alpha) * ti.cos(gamma),
         ti.sin(alpha) * ti.sin(beta) * ti.sin(gamma) + ti.cos(alpha) * ti.cos(gamma),
         ti.cos(beta) * ti.sin(gamma)],
        [ti.cos(alpha) * ti.sin(beta) * ti.cos(gamma) + ti.sin(alpha) * ti.sin(gamma),
         ti.sin(alpha) * ti.sin(beta) * ti.cos(gamma) - ti.cos(alpha) * ti.sin(gamma),
         ti.cos(beta) * ti.cos(gamma)]
    ])
    return R

@ti.kernel
def forward_kinematics(handles: ti.ext_arr(), hier: ti.ext_arr(), eulers: ti.ext_arr(), T: ti.ext_arr(), new_handles: ti.ext_arr(), newR: ti.ext_arr()):
    n = handles.shape[0]
    for i in range(n):    
        lo = compute_rotation(eulers[i, 0], eulers[i, 1], eulers[i, 2]) # (3, 3)
        if hier[i] == 0:  # what does this case mean?
            for j in range(3):
                for k in range(3):
                    T[i, j * 3 + k] = ti.cast(lo[j, k], ti.f32) # F order
            for k in range(3):
                new_handles[i, k] = handles[i, k]
                
        else:
            
            for j in range(9):
                newR[i - 1, j] = T[int(hier[i] - 1), j]  # only 
                #print(newR[i - 1, j])
                print("newR",newR[i - 1, j])
            Tp = ti.Matrix([   # printing shows Tp is always the identity
                [T[int(hier[i] - 1), 0], T[int(hier[i] - 1), 3], T[int(hier[i] - 1), 6]],
                [T[int(hier[i] - 1), 1], T[int(hier[i] - 1), 4], T[int(hier[i] - 1), 7]],
                [T[int(hier[i] - 1), 2], T[int(hier[i] - 1), 5], T[int(hier[i] - 1), 8]],
            ])
            print("Tp", Tp)
            vecl = ti.Vector([handles[i, l] - handles[int(hier[i] - 1), l] for l in range(3)]) #(3,)
            vec = ti.Vector([0.0, 0.0, 0.0])
            for l in range(3):
                vec[l] = ti.cast(new_handles[int(hier[i] - 1), l] + Tp[l, :] @ vecl, ti.f32) #(3,)
            print("vec",vec)
            for l in range(3):
                new_handles[i, l] = vec[l]
                print(new_handles[i, l])
            print("lo", lo)
            temp = Tp @ lo
            print("temp", temp)
            for j in range(3):
                for k in range(3):
                    T[i, j * 3 + k] = ti.cast(temp[j, k], ti.f32)
                    print("T", T[i, j * 3 + k] )



"""
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
# Create the newR array outside the kernel
n_newR = handles.shape[0] - 1
newR = np.tile(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]), (n_newR, 1))


# Run the forward_kinematics function
forward_kinematics(handles, hierarchy, eulers, T, new_handles, newR)

print("New handles:")
print(new_handles.shape)
print("\nNew rotations:")
print(newR.shape)

"""