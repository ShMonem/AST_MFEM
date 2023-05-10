import taichi as ti
import igl
import scipy
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import polyscope as ps
from scipy.sparse.linalg import svds, lsqr, spsolve
from scipy.sparse import csr_matrix, csc_matrix, isspmatrix
from scipy.linalg import pinv
import sys
sys.path.append("../Bartels/python/build")
import bartelspy as bt
import meshio
from def_grad3D import *
from linear_tet3dmesh_arap_ds2 import *
from linear_tet3dmesh_arap_ds import *
from build_U import *
from closest_index import *
from tetAssignment import *
from forward_kinematics import *
from block_R3d import *
from pinvert import *
from igl2bart import *
from compute_J import *
from V_Cycle import *


if __name__ == '__main__':
    mesh = meshio.read("../data/beam.mesh")
    Vt = mesh.points
    Tt = mesh.cells[1].data.astype("int32")

    print("Reading data...")
    print("Vertices: ", len(Vt), Vt.shape)
    print("Tets: ", len(Tt), Tt.shape)

    B = def_grad3D(Vt, Tt)
    mu = 100  # material properties
    k_bc = 1000  # stiffness
    s = np.zeros((6 * Tt.shape[0], 1))
    grad = linear_tet3dmesh_arap_ds(Vt, Tt, s, mu)
    hess = linear_tet3dmesh_arap_ds2(Vt, Tt, s, mu)

    # making a skeleton
    hier = np.array([0, 1, 2])
    handles = np.array([[-4.5, 0.0, 0.0], [0.0, 0.0, 0.0], [4.5, 0.0, 0.0]])

    b_levels = np.array([[30]]).astype(int)
    l = b_levels.shape[0]
    py_P = scipy.io.loadmat("../data/P_beam.mat")['P']
    py_PI = scipy.io.loadmat("../data/PI_beam.mat")['PI']


    # translate everything to taichi
    ti.init(arch=ti.cpu)
    V = ti.Vector.field(3, dtype=ti.f32, shape=Vt.shape[0])
    T = ti.Vector.field(3, dtype=ti.i32, shape=Tt.shape[0])
    V.from_numpy(Vt)
    T.from_numpy(Tt)
    weight = ti.field(ti.i32, shape=Vt.shape[0])
    D = ti.Vector.field(py_P.shape[0], dtype=ti.f32, shape=Vt.shape[0])
    P = ti.Vector.field(3, dtype=ti.f32, shape=py_P.shape[0])
    P.from_numpy(py_P)
    closest_index(V, P, weight, D)
    Ut, NN = build_U(weight.to_numpy(), b_levels, l, P, Vt)

    midpointst = np.zeros((handles.shape[0] - 1, 3))
    for i in range(handles.shape[0]):
        if hier[i] == 0:
            continue
        else:
            midpointst[i - 1, :] = (handles[i, :] + handles[int(hier[i]) - 1, :]) / 2
    midpoints = ti.Vector.field(3, dtype=ti.f32, shape=midpointst.shape[0])
    midpoints.from_numpy(midpointst)

    fAssign = ti.field(dtype=ti.i32, shape=Tt.shape[0])
    tetAssignment(V, T, midpoints, fAssign)

    eulers = np.array([[0.0, 0.0, 0.0], [0.0, np.pi / 4, 0.0], [0.0, 0.0, 0.0]])
    # eulers = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    n_newR = handles.shape[0] - 1
    n_newR = handles.shape[0] - 1
    Tdummy = np.zeros((n_newR + 1, 9))
    newR = np.zeros((n_newR, 9))
    Tdummy[:, [0, 4, 8]] = 1
    newR[:, [0, 4, 8]] = 1
    new_handles = np.zeros_like(handles)
    forward_kinematics(handles, hier, eulers, Tdummy, new_handles, newR)
    rows_T = Tt.shape[0]
    py_fAssign = fAssign.to_numpy()
    R = np.zeros((rows_T, 9))
    for i in range(rows_T):
        R[i, :] = newR[py_fAssign[i], :]
    R_mat_ti, R_mat_py = block_R3d(R)

    num_handles = handles.shape[0]
    num_midpoints = midpointst.shape[0]
    Ht = np.zeros((num_handles + num_midpoints, 3))
    Ht[:num_handles, :] = handles
    Ht[num_handles:, :] = midpoints.to_numpy()
    vAssign = ti.field(ti.i32, shape=(Ht.shape[0],))
    Dt = ti.field(ti.f32, shape=(Vt.shape[0], Ht.shape[0]))
    H = ti.field(ti.f32, shape=(Ht.shape[0], 3))
    H.from_numpy(Ht)
    V = ti.field(ti.f32, shape=(Vt.shape[0], 3))
    V.from_numpy(Vt)
    pinvert(V, H, vAssign, Dt)

    I = np.eye(3)
    q = igl2bart(Vt)

    pinned_mat = sps.lil_matrix((3 * H.shape[0], 3 * V.shape[0]))
    py_vAssign = vAssign.to_numpy()
    for i in range(H.shape[0]):
        pinned_mat[3 * i:3 * (i + 1), 3 * py_vAssign[i]:3 * py_vAssign[i]+3] = I

    midpoints = np.zeros((new_handles.shape[0] - 1, 3))
    for i in range(new_handles.shape[0]):
        if hier[i] == 0:
            continue
        else:
            midpoints[i - 1, :] = (new_handles[i, :] + new_handles[hier[i], :]) / 2

    new_new_handles = np.zeros((new_handles.shape[0] + midpoints.shape[0], 3))
    new_new_handles[:new_handles.shape[0], :] = new_handles
    new_new_handles[new_handles.shape[0]:, :] = midpoints

    pinned_b = igl2bart(new_new_handles)
    # scipy.io.savemat("../data/pinned_b.mat", {"mat": pinned_b})

    pinned_b = scipy.io.loadmat("../data/pinned_b.mat")['pinned_b']

    nq = q.shape[0]
    ns = s.shape[0]
    nlambda = 9 * Tt.shape[0]

    J = compute_J(R_mat_py, B)
    A = k_bc * pinned_mat.T @ pinned_mat + J.T @ hess @ J
    b = k_bc * pinned_mat.T @ pinned_b - J.T @ grad

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
    A_field = ti.field(dtype=ti.f32, shape=A.shape)
    fill_U_Sparse(A_field, rows, cols, A.data)

    b_field = ti.field(dtype=ti.f32, shape=(b.shape[0],))
    fill_field(b_field, np.squeeze(b))

    sol_init = np.zeros(b.shape, dtype=np.float32)
    x = ti.field(dtype=ti.f32, shape=(sol_init.shape[0],))
    x_old = ti.field(dtype=ti.f32, shape=(x.shape[0],))
    result = ti.field(dtype=ti.f32, shape=x.shape[0])
    fill_field(x, np.squeeze(sol_init))
    itr_num = 100
    l = len(Ut) - 1
    U_field, U, L = A_L_sum_U_py(A)
    while normVal > tol:
        sol_old = sol
        start = time()
        # sol = v_cycle_ti(A_field, b_field, UTAU, projA_solvers, Ub,  l, U, L_solver, itr_num,  x_old)
        sol = v_cycle_py(A, U, L, b, UTAU, Ut, l, itr_num, sol_old)
        end = time()
        # print('V_cycle, itr', itr, 'in ', round(end - start, 3),'s')
        normVal = npla.norm(b - A.dot(sol))
        # itr = itr + 1
        print("error: ", normVal)

    print(sol.shape)



    # visualization
    # ps.set_program_name("mfem")
    # ps.set_ground_plane_mode("shadow_only")
    # ps.init()
    # ps_vol = ps.register_volume_mesh("test volume mesh", Vt, tets=Tt)
    #
    # points = Vt[py_vAssign, :]
    # ps_cloud = ps.register_point_cloud("my points", points)
    # ps.show()