import scipy
import numpy as np
import scipy.sparse as sps
import polyscope as ps
from time import time
import meshio
from def_grad3D import def_grad3D
from linear_tet3dmesh_arap_ds2 import linear_tet3dmesh_arap_ds2
from linear_tet3dmesh_arap_ds import linear_tet3dmesh_arap_ds
from build_U import build_U
from closest_index import closest_index_np
from tetAssignment import tetAssignment_py
from forward_kinematics import forward_kinematics_np
from block_R3d import block_R3d
from pinvert import pinvert_np
from igl2bart import igl2bart
from compute_J import compute_J_SVD, compute_J
from V_Cycle import v_cycle_py
from GS import A_L_sum_U_np
from fill_euler import fill_euler
import config


if __name__ == '__main__':
    print("Reading data...")

    ########## BEAM DATA ##############
    # read beam model
    mesh = meshio.read("../data/beam.mesh")
    Vt = mesh.points
    Tt = mesh.cells[1].data.astype("int32")
    # make beam skeleton
    hier = np.array([0, 1, 2])
    handles = np.array([[-4.5, 0.0, 0.0], [0.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
    # level of beam
    b_levels = np.array([[30]]).astype(int)
    # load beam samples
    py_P = scipy.io.loadmat("../data/P_beam.mat")['P']
    py_PI = scipy.io.loadmat("../data/PI_beam.mat")['PI']
    # beam eulers
    eulers = np.array([[0.0, 0.0, 0.0], [-np.pi / 3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # eulers = np.array([[0.0, 0.0, 0.0], [0.0, -np.pi/2, 0.0], [0.0, 0.0, 0.0]])
    ########## BEAM DATA END ##############

    ########## HUMAN DATA ##############
    # # read human model
    # mesh = meshio.read("../data/output_mfem1.msh")
    # Vt = mesh.points
    # Tt = mesh.cells[0].data.astype("int32")
    # # load human skeleton
    # handlest = scipy.io.loadmat('../data/handles.mat')
    # hiert = scipy.io.loadmat("../data/hierarchy.mat")
    # handles = handlest['position']  ## to access the mat from the dict use: handles['position'] (numHandels, 3)
    # hier = hiert['hierarchy'][:, 1]
    # # level of human
    # b_levels = np.array([[50]]).astype(int)
    # # load human samples
    # py_P = scipy.io.loadmat("../data/P.mat")['P']  ## to access the mat from the dict use: Pt['P']
    # py_PI = scipy.io.loadmat("../data/PI.mat")['PI']
    # # load human eulers
    # eulers = fill_euler(handles)
    ########## HUMAN DATA END ##############

    print("Vertices: ", len(Vt), Vt.shape)
    print("Tets: ", len(Tt), Tt.shape)

    B = def_grad3D(Vt, Tt)
    mu = 100  # material properties
    k_bc = 10000  # stiffness
    s = np.zeros((6 * Tt.shape[0], 1))
    grad = linear_tet3dmesh_arap_ds(Vt, Tt, s, mu)
    hess = linear_tet3dmesh_arap_ds2(Vt, Tt, s, mu)

    l = b_levels.shape[0]

    weight = closest_index_np(Vt, py_P)
    Ut, NN = build_U(weight, b_levels, l, py_P, Vt)

    midpoints_np = np.zeros((handles.shape[0] - 1, 3))
    for i in range(handles.shape[0]):
        if hier[i] == 0:
            continue
        else:
            midpoints_np[i - 1, :] = (handles[i, :] + handles[int(hier[i]) - 1, :]) / 2

    fAssign = tetAssignment_py(Vt, Tt, midpoints_np)

    # We move the handles here, so let's consider that we start the sim from around this point
    start_all_sim = time()

    newR, new_handles = forward_kinematics_np(handles, hier, eulers)
    rows_T = Tt.shape[0]
    R = np.zeros((rows_T, 9))
    # assigning rotations to each tet according to fAssign info
    for i in range(rows_T):
        R[i, :] = newR[int(fAssign[i]), :]

    R_mat_py = block_R3d(R)  # python function

    num_handles = handles.shape[0]
    num_midpoints = midpoints_np.shape[0]
    Ht = np.zeros((num_handles + num_midpoints, 3))
    Ht[:num_handles, :] = handles
    Ht[num_handles:, :] = midpoints_np

    vAssign = pinvert_np(Vt, Ht)
    I = np.eye(3)
    q = igl2bart(Vt)

    pinned_mat = sps.lil_matrix((3 * Ht.shape[0], 3 * Vt.shape[0]))
    for i in range(Ht.shape[0]):
        pinned_mat[3 * i:3 * (i + 1), 3 * int(vAssign[i]):3 * int(vAssign[i])+3] = I

    midpoints = np.zeros((new_handles.shape[0] - 1, 3))
    for i in range(new_handles.shape[0]):
        if hier[i] == 0:
            continue
        else:
            midpoints[i - 1, :] = (new_handles[i, :] + new_handles[hier[i] - 1, :]) / 2

    new_new_handles = np.zeros((new_handles.shape[0] + midpoints.shape[0], 3))
    new_new_handles[:new_handles.shape[0], :] = new_handles
    new_new_handles[new_handles.shape[0]:, :] = midpoints

    pinned_b = igl2bart(new_new_handles)

    nq = q.shape[0]
    ns = s.shape[0]
    nlambda = 9 * Tt.shape[0]
    start_j = time()
    if config.USE_SVD:
        J = compute_J_SVD(R_mat_py, B)  # cupy/numpy function
    else:
        J = compute_J(R_mat_py, B)
    end_j = time()
    print("J computed in {0} seconds".format(end_j-start_j))
    A = k_bc * pinned_mat.T @ pinned_mat + J.T @ hess @ J
    b = k_bc * pinned_mat.T @ pinned_b - J.T @ grad
    b = np.squeeze(b)

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
    itr_num = 3
    l = len(Ut) - 1
    U, L = A_L_sum_U_np(A)  # python
    while normVal > tol:
        sol_old = sol
        start = time()
        sol = v_cycle_py(A, U, L, b, UTAU, Ut, l, itr_num, sol_old)
        normVal = np.linalg.norm(b - A.dot(sol))
        end = time()
        print('V_cycle, itr', itr, 'in ', round(end - start, 3), 's')
        print("error: ", normVal)

    end_all_sim = time()

    print(np.linalg.norm(b - A.dot(sol)))
    print(sol.shape)
    print("The complete sim took {0} seconds.".format(end_all_sim - start_all_sim))

    # visualization
    ps.set_program_name("mfem")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    ps_vol = ps.register_volume_mesh("test volume mesh", sol.reshape(-1, 3), tets=Tt)
    ps.show()