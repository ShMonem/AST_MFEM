# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

import scipy
import numpy as np
import scipy.sparse as sps
from time import time
from python.ast_fem_np.forward_kinematics_np import forward_kinematics
from python.ast_fem_np.block_r3d_np import block_r3d
from python.common.igl2bart import igl2bart
from python.ast_fem_np.compute_j_np import compute_J_SVD
from python.ast_fem_np.v_cycle_np import v_cycle
from python.ast_fem_np.gauss_seidel_np import A_L_sum_U
from python.ast_fem_np.fem_data_np import FEMData
import python.ast_fem_np.config as config


def solve(obj_data):
    debug = config.DEBUG
    start_global = time()

    # We move the handles here, so let's consider that we start the sim from around this point
    if debug:
        start_all_sim = time()

    if debug:
        start = time()
    if obj_data.eulers is not None:
        newR, new_handles = forward_kinematics(obj_data.handles_pos, obj_data.hier, obj_data.eulers)
        # newR = np.load(r'C:\Users\DimitryKachkovski\git\personal\AST_MFEM\data\human\newR.npy')
        # new_handles = np.load(r'C:\Users\DimitryKachkovski\git\personal\AST_MFEM\data\human\new_handles.npy')
    else:
        newR = obj_data.handles_rot
        new_handles = obj_data.handles_pos
    if debug:
        end = time()
        print(f"Running FWD Kinematics took {end-start} seconds.")
    rows_T = obj_data.tets.shape[0]
    R = np.zeros((rows_T, 9))
    # assigning rotations to each tet according to fAssign info
    for i in range(rows_T):
        R[i, :] = newR[obj_data.tet_assign[i], :]

    if debug:
        start = time()
    R_mat_py, R_mat_blocks = block_r3d(R, blocks=True)  # python function
    if debug:
        end = time()
        print(f"block_r3d took {end-start} seconds.")

    pinned_mat = sps.lil_matrix((3 * obj_data.Ht.shape[0], 3 * obj_data.verts.shape[0]))
    for i in range(obj_data.Ht.shape[0]):
        pinned_mat[3 * i:3 * (i + 1), 3 * int(obj_data.pin_verts[i]):3 * int(obj_data.pin_verts[i])+3] = np.eye(3)

    midpoints = np.zeros((new_handles.shape[0] - 1, 3))
    for i in range(new_handles.shape[0]):
        if obj_data.hier[i] == 0:
            continue
        else:
            midpoints[i - 1, :] = (new_handles[i, :] + new_handles[obj_data.hier[i] - 1, :]) / 2

    new_new_handles = np.zeros((new_handles.shape[0] + midpoints.shape[0], 3))
    new_new_handles[:new_handles.shape[0], :] = new_handles
    new_new_handles[new_handles.shape[0]:, :] = midpoints

    pinned_b = igl2bart(new_new_handles)

    if debug:
        start = time()
    J = compute_J_SVD(R_mat_py, obj_data.B, r_mat_blocks=R_mat_blocks)  # cupy/numpy function
    if debug:
        end = time()
        print(f"J computed in {end-start} seconds.")
    A = obj_data.k_bc * pinned_mat.T @ pinned_mat + J.T @ obj_data.hess @ J
    b = obj_data.k_bc * pinned_mat.T @ pinned_b - J.T @ obj_data.grad
    b = np.squeeze(b)

    if config.USE_MG:
        UTAU = []
        for i in range(len(obj_data.Ut)):
            if i == 0:
                UTAU.append(obj_data.Ut[i].T.dot(A).dot(obj_data.Ut[i]) + obj_data.NN)
            else:
                UTAU.append(obj_data.Ut[i].T.dot(UTAU[i - 1]).dot(obj_data.Ut[i]))

        normVal = float('inf')
        tol = 1e-5
        sol = np.zeros(b.shape)
        itr_num = 1
        l = len(obj_data.Ut) - 1
        U, L = A_L_sum_U(A)  # python
        if debug:
            print("Using the MG solver...")
            start_solve = time()
        while normVal > tol:
            sol_old = sol
            sol = v_cycle(A, U, L, b, UTAU, obj_data.Ut, l, itr_num, sol_old, debug=False)
            normVal = np.linalg.norm(b - A.dot(sol))
            if debug:
                print("error: ", normVal)
    else:
        if debug:
            print("Using the direct solver...")
            start_solve = time()
        sol = scipy.sparse.linalg.spsolve(A, b)
    if debug:
        end_solve = time()
        print(f"The solve took {end_solve-start_solve} seconds.")
        print(f"The complete sim took {end_solve-start_all_sim} seconds.")
        print(f"The entire process from startup took {end_solve-start_global} seconds.")

    return sol, obj_data.tets