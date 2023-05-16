import numpy as np


def compute_rotation(alpha, beta, gamma):
    R = np.array([
        [np.cos(alpha) * np.cos(beta),
         np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma),
         np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)],

        [np.sin(alpha) * np.cos(beta),
         np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
         np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)],

        [-np.sin(beta),
         np.cos(beta) * np.sin(gamma),
         np.cos(beta) * np.cos(gamma)]
    ])
    return R.astype(float)


def forward_kinematics(handles, hier, eulers):
    n = handles.shape[0]
    new_handles = np.zeros_like(handles)
    n_newR = handles.shape[0] - 1
    newR = np.zeros((n_newR, 9))
    newR[:, [0, 4, 8]] = 1
    T = np.zeros((n_newR + 1, 9))
    T[:, [0, 4, 8]] = 1
    for i in range(n):
        loc_rot_mtx = compute_rotation(eulers[i, 0], eulers[i, 1], eulers[i, 2])  # (3, 3)
        if hier[i] == 0:  # what does this case mean? # this is the root node
            for j in range(3):
                for k in range(3):
                    T[i, j * 3 + k] = loc_rot_mtx[j, k]  # F order
            for k in range(3):
                new_handles[i, k] = handles[i, k]
        else:
            for j in range(9):
                newR[i - 1, j] = T[int(hier[i] - 1), j]
            Tp = np.array([  # printing shows Tp is always the identity
                [T[int(hier[i] - 1), 0], T[int(hier[i] - 1), 3], T[int(hier[i] - 1), 6]],
                [T[int(hier[i] - 1), 1], T[int(hier[i] - 1), 4], T[int(hier[i] - 1), 7]],
                [T[int(hier[i] - 1), 2], T[int(hier[i] - 1), 5], T[int(hier[i] - 1), 8]],
            ])
            vecl = np.array([handles[i, l] - handles[int(hier[i] - 1), l] for l in range(3)])
            vec = np.array([0.0, 0.0, 0.0])
            for l in range(3):
                vec[l] = new_handles[int(hier[i] - 1), l] + Tp[l, :] @ vecl
            for l in range(3):
                new_handles[i, l] = vec[l]
            temp = Tp @ loc_rot_mtx
            temp = temp.astype(float)
            for j in range(3):
                for k in range(3):
                    T[i, j * 3 + k] = temp[k, j]
    return newR, new_handles
