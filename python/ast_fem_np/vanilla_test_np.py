import scipy
import numpy as np
import scipy.sparse as sps
from time import time
from python.common.def_grad3d import def_grad3d
from python.common.linear_tet3dmesh_arap_ds2 import linear_tet3dmesh_arap_ds2
from python.common.linear_tet3dmesh_arap_ds import linear_tet3dmesh_arap_ds
from python.ast_fem_np.build_u_np import build_u
from python.ast_fem_np.closest_index_np import closest_index
from python.ast_fem_np.tet_assignment_np import tet_assignment
from python.ast_fem_np.forward_kinematics_np import forward_kinematics
from python.ast_fem_np.block_r3d_np import block_r3d
from python.ast_fem_np.pinvert_np import pinvert
from python.common.igl2bart import igl2bart
from python.ast_fem_np.compute_j_np import compute_J_SVD
from python.ast_fem_np.v_cycle_np import v_cycle
from python.ast_fem_np.gauss_seidel_np import A_L_sum_U
from python.ast_fem_np.fem_data_np import FEMData
import python.ast_fem_np.config as config


def main(obj_name):
    if obj_name not in ['beam', 'human']:
        raise Exception(f"The element {obj_name} is not supported!")
    debug = config.DEBUG
    start_global = time()
    if debug:
        start = time()
    obj_data = FEMData(obj_name)

    if debug:
        end = time()
        print(f"Reading data took {end-start} seconds.")
        print("Vertices: ", len(obj_data.verts), obj_data.verts.shape)
        print("Tets: ", len(obj_data.tets), obj_data.tets.shape)

    ########## ONE TIME SETUP ##########
    if debug:
        print("Starting one time setup process...")
        start_setup = time()

    if debug:
        start = time()
    B = def_grad3d(obj_data.verts, obj_data.tets)
    if debug:
        end = time()
        print(f"Getting grad took {end-start} seconds.")
    mu = 100  # material properties
    k_bc = 10000  # stiffness
    s = np.zeros((6 * obj_data.tets.shape[0], 1))
    if debug:
        start = time()
    grad = linear_tet3dmesh_arap_ds(obj_data.verts, obj_data.tets, s, mu)
    hess = linear_tet3dmesh_arap_ds2(obj_data.verts, obj_data.tets, s, mu)
    if debug:
        end = time()
        print(f"Getting energy took {end-start} seconds.")

    l = obj_data.b_levels.shape[0]

    if debug:
        start = time()
    weight = closest_index(obj_data.verts, obj_data.P)
    if debug:
        end = time()
        print(f"Computing closest index took {end-start} seconds.")
    if debug:
        start = time()
    Ut, NN = build_u(weight, obj_data.b_levels, l, obj_data.P, obj_data.verts)
    if debug:
        end = time()
        print(f"Building the U matrix took {end-start} seconds.")

    midpoints_np = np.zeros((obj_data.handles_pos.shape[0] - 1, 3))
    for i in range(obj_data.handles_pos.shape[0]):
        if obj_data.hier[i] == 0:
            continue
        else:
            midpoints_np[i - 1, :] = \
                (obj_data.handles_pos[i, :] + obj_data.handles_pos[int(obj_data.hier[i]) - 1, :]) / 2

    if debug:
        start = time()
    fAssign = tet_assignment(obj_data.verts, obj_data.tets, midpoints_np)
    if debug:
        end = time()
        print(f"Computing tet assignments took {end-start} seconds.")

    num_handles = obj_data.handles_pos.shape[0]
    num_midpoints = midpoints_np.shape[0]
    Ht = np.zeros((num_handles + num_midpoints, 3))
    Ht[:num_handles, :] = obj_data.handles_pos
    Ht[num_handles:, :] = midpoints_np

    if debug:
        start = time()
    vAssign = pinvert(obj_data.verts, Ht)
    if debug:
        end = time()
        print(f"Computing the pinned verts took {end-start} seconds.")
    if debug:
        print(f"Finished the one time setup.\nThe setup took {end-start_setup} seconds.")

    # We move the handles here, so let's consider that we start the sim from around this point
    if debug:
        start_all_sim = time()

    if debug:
        start = time()
    newR, new_handles = forward_kinematics(obj_data.handles_pos, obj_data.hier, obj_data.eulers)
    if debug:
        end = time()
        print(f"Running FWD Kinematics took {end-start} seconds.")
    rows_T = obj_data.tets.shape[0]
    R = np.zeros((rows_T, 9))
    # assigning rotations to each tet according to fAssign info
    for i in range(rows_T):
        R[i, :] = newR[int(fAssign[i]), :]

    if debug:
        start = time()
    R_mat_py, R_mat_blocks = block_r3d(R, blocks=True)  # python function
    if debug:
        end = time()
        print(f"block_r3d took {end-start} seconds.")

    I = np.eye(3)
    q = igl2bart(obj_data.verts)

    pinned_mat = sps.lil_matrix((3 * Ht.shape[0], 3 * obj_data.verts.shape[0]))
    for i in range(Ht.shape[0]):
        pinned_mat[3 * i:3 * (i + 1), 3 * int(vAssign[i]):3 * int(vAssign[i])+3] = I

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

    nq = q.shape[0]
    ns = s.shape[0]
    nlambda = 9 * obj_data.tets.shape[0]
    if debug:
        start = time()
    J = compute_J_SVD(R_mat_py, B, r_mat_blocks=R_mat_blocks)  # cupy/numpy function
    if debug:
        end = time()
        print(f"J computed in {end-start} seconds.")
    A = k_bc * pinned_mat.T @ pinned_mat + J.T @ hess @ J
    b = k_bc * pinned_mat.T @ pinned_b - J.T @ grad
    b = np.squeeze(b)

    if config.USE_MG:
        UTAU = []
        for i in range(len(Ut)):
            if i == 0:
                UTAU.append(Ut[i].T.dot(A).dot(Ut[i]) + NN)
            else:
                UTAU.append(Ut[i].T.dot(UTAU[i - 1]).dot(Ut[i]))

        normVal = float('inf')
        itr = 0
        tol = 1e-5
        sol = np.zeros(b.shape)
        rows, cols = A.nonzero()
        itr_num = 1
        l = len(Ut) - 1
        U, L = A_L_sum_U(A)  # python
        if debug:
            print("Using the MG solver...")
            start_solve = time()
        while normVal > tol:
            sol_old = sol
            sol = v_cycle(A, U, L, b, UTAU, Ut, l, itr_num, sol_old, debug=False)
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