# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

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
    newR = np.zeros((n-1, 3, 3))
    newR[:, :] = np.eye(3)
    T = np.zeros((n, 3, 3))
    T[:, :] = np.eye(3)
    for i in range(n):
        loc_rot_mtx = compute_rotation(eulers[i, 0], eulers[i, 1], eulers[i, 2])  # (3, 3)
        if hier[i] == 0:  # what does this case mean? # this is the root node
            T[i] = loc_rot_mtx  # F order
            new_handles[i] = handles[i]
        else:
            newR[i - 1] = T[int(hier[i] - 1)]
            Tp = T[int(hier[i] - 1)].T
            vecl = handles[i] - handles[int(hier[i] - 1)]
            vec = new_handles[int(hier[i] - 1)] + Tp @ vecl
            new_handles[i, :] = vec
            temp = (Tp @ loc_rot_mtx).astype(float)
            T[i] = temp.T
    return newR.reshape((-1,9)), new_handles
