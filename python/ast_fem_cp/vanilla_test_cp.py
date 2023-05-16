import scipy
import numpy as np
import cupy as cp
import cupyx as cpx
import scipy.sparse as sps
import polyscope as ps
from time import time
import meshio
from python.common.def_grad3d import def_grad3d
from python.common.linear_tet3dmesh_arap_ds2 import linear_tet3dmesh_arap_ds2
from python.common.linear_tet3dmesh_arap_ds import linear_tet3dmesh_arap_ds
from python.ast_fem_cp.build_u_cp import build_u_cp
from python.ast_fem_cp.closest_index_cp import closest_index_np
from python.ast_fem_cp.tet_assignment_cp import tet_assignment_np
from python.common.igl2bart import igl2bart
from forward_kinematics_cp import forward_kinematics
from block_r3d_cp import block_r3d
from pinvert_cp import pinvert
from compute_j_cp import compute_J_SVD
from v_cycle_cp import v_cycle
from gs_cp import A_L_sum_U
from fill_euler_cp import fill_euler_cp
import config


if __name__ == '__main__':
    print("Reading data...")
    start_data = time()
    ########## BEAM DATA ##############
    # # read beam model
    # mesh = meshio.read("../data/beam.mesh")
    # Vt = mesh.points
    # Tt = mesh.cells[1].data.astype("int32")
    # # make beam skeleton
    # hier = np.array([0, 1, 2])
    # handles = np.array([[-4.5, 0.0, 0.0], [0.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
    # # level of beam
    # b_levels = np.array([[30]]).astype(int)
    # # load beam samples
    # py_P = scipy.io.loadmat("../data/P_beam.mat")['P']
    # py_PI = scipy.io.loadmat("../data/PI_beam.mat")['PI']
    # # beam eulers
    # eulers = np.array([[0.0, 0.0, 0.0], [-np.pi / 3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # # eulers = np.array([[0.0, 0.0, 0.0], [0.0, -np.pi/2, 0.0], [0.0, 0.0, 0.0]])
    ########## BEAM DATA END ##############

    ########## HUMAN DATA ##############
    # read human model
    mesh = meshio.read("../../data/output_mfem1.msh")
    Vt_np = mesh.points
    Tt_np = mesh.cells[0].data.astype("int32")
    Vt = cp.asarray(Vt_np)
    Tt = cp.asarray(Tt_np)
    # load human skeleton
    handlest = scipy.io.loadmat('../../data/handles.mat')
    hiert = scipy.io.loadmat("../../data/hierarchy.mat")
    handles = cp.asarray(handlest['position'])  ## to access the mat from the dict use: handles['position'] (numHandels, 3)
    hier = cp.asarray(hiert['hierarchy'][:, 1])
    # level of human
    b_levels = cp.array([[50]]).astype(int)
    # load human samples
    P_np = scipy.io.loadmat("../../data/P.mat")['P']
    PI_np = scipy.io.loadmat("../../data/PI.mat")['PI']
    P = cp.asarray(P_np)  ## to access the mat from the dict use: Pt['P']
    PI = cp.asarray(PI_np)
    # load human eulers
    eulers = fill_euler_cp(handles)
    ########## HUMAN DATA END ##############

    end_data = time()
    print("Done reading data...")
    print("Reading data took {0} seconds.".format(end_data - start_data))
    print("Vertices: ", len(Vt), Vt.shape)
    print("Tets: ", len(Tt), Tt.shape)

    print("Getting grad...")
    # We have to use the original numpy data to compute B, and only then we go onto the GPU
    B = cpx.scipy.sparse.csc_matrix(def_grad3d(Vt_np, Tt_np))
    mu = 100  # material properties
    k_bc = 10000  # stiffness
    s = np.zeros((6 * Tt.shape[0], 1))
    print("Getting energy...")
    grad = cp.asarray(linear_tet3dmesh_arap_ds(Vt_np, Tt_np, s, mu))
    hess = cpx.scipy.sparse.dia_matrix(linear_tet3dmesh_arap_ds2(Vt_np, Tt_np, s, mu))

    l = b_levels.shape[0]

    print("Computing closest index...")
    # Kernel initialization takes some time, and we only compute the closest indexes once, so it is much faster to use
    # a NumPy implementation to get this, and then simply send it to the GPU.
    weight = cp.asarray(closest_index_np(mesh.points, P_np))
    print("Building the U matrix...")
    Ut, NN = build_u_cp(weight, b_levels, l, P, Vt)

    midpoints_np = cp.zeros((handles.shape[0] - 1, 3))
    for i in range(handles.shape[0]):
        if hier[i] == 0:
            continue
        else:
            midpoints_np[i - 1, :] = (handles[i, :] + handles[int(hier[i]) - 1, :]) / 2

    print("Computing tet assignments...")
    # Same kernel initialization consideration here - using numpy instead of CuPy to be faster
    fAssign = cp.asarray(tet_assignment_np(Vt, Tt, midpoints_np))

    # We move the handles here, so let's consider that we start the sim from around this point
    start_all_sim = time()

    print("Running FWD Kinematics...")
    newR, new_handles = forward_kinematics(handles, hier, eulers)
    rows_T = Tt.shape[0]
    R = np.zeros((rows_T, 9))
    # assigning rotations to each tet according to fAssign info
    for i in range(rows_T):
        R[i, :] = newR[int(fAssign[i]), :]

    print("Getting the R_Mat...")
    R_mat_py = block_r3d(R)  # python function

    num_handles = handles.shape[0]
    num_midpoints = midpoints_np.shape[0]
    Ht = np.zeros((num_handles + num_midpoints, 3))
    Ht[:num_handles, :] = handles
    Ht[num_handles:, :] = midpoints_np

    print("Computing the pinned verts...")
    vAssign = pinvert(Vt, Ht)
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
    J = compute_J_SVD(R_mat_py, B)  # cupy/numpy function
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
    U, L = A_L_sum_U(A)  # python
    print("Starting the multigrid solve...")
    # convert the elements we will use to cupy and thus send to the GPU
    sol_cp = cp.asarray(sol)
    A_cp = cpx.scipy.sparse.csr_matrix(A)
    U_cp, L_cp = A_L_sum_U(A_cp)
    b_cp = cp.asarray(b)
    UTAU_cp = [cpx.scipy.sparse.csc_matrix(u_m) for u_m in UTAU]
    Ut_cp = [cpx.scipy.sparse.csr_matrix(u_m) for u_m in Ut]
    start_mg = time()
    while normVal > tol:
        sol_old_cp = sol_cp
        sol_cp = v_cycle(A_cp, U_cp, L_cp, b_cp, UTAU_cp, Ut_cp, l, itr_num, sol_old_cp, debug=False)
        normVal = np.linalg.norm(b_cp - A_cp.dot(sol_cp))
        print("error: ", normVal)
    sol = cp.asnumpy(sol_cp)
    print("Finished the multigrid solve")
    end_all_sim = time()
    print(np.linalg.norm(b - A.dot(sol)))
    print(sol.shape)
    print("The multigrid solve took {0} seconds.".format(end_all_sim-start_mg))
    print("The complete sim took {0} seconds.".format(end_all_sim - start_all_sim))

    # visualization
    ps.set_program_name("mfem")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    ps_vol = ps.register_volume_mesh("test volume mesh", sol.reshape(-1, 3), tets=Tt)
    ps.show()