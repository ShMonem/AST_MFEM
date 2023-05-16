import scipy
import numpy as np
import scipy.sparse as sps
import polyscope as ps
from time import time
import meshio
from def_grad3D import def_grad3D
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
import taichi as ti
from enum import Enum


class MaterialModel(Enum):
    arap = 1


@ti.data_oriented
class MFEMSolver:
    def __init__(self, model_file, skeleton_file, hierarchy_file, mu, levels):
        self.levels = ti.ndarray(dtype=ti.int32, shape=(levels.shape[0]))
        self.levels.from_numpy(levels)
        self.material_model = MaterialModel.arap
        self.mu = mu
        mesh = meshio.read(model_file)
        self.V = ti.Vector.field(3, dtype=ti.f32, shape=(mesh.points.shape[0],))
        self.V.from_numpy(mesh.points)
        self.num_V = self.V.shape[0]
        self.T = ti.Vector.field(4, dtype=int, shape=(mesh.cells[1].data.shape[0]))
        self.T.from_numpy(mesh.cells[1].data)
        self.num_T = self.T.shape[0]
        bone_hierarchy_np = np.load(hierarchy_file)
        self.num_bone = bone_hierarchy_np.shape[0]
        self.bone_hierarchy = ti.field(dtype=int, shape=self.num_bone)
        self.bone_hierarchy.from_numpy(bone_hierarchy_np)
        skeleton_np = np.load(skeleton_file)
        self.num_skeleton = skeleton_np.shape[0]
        self.skeleton = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_skeleton,))
        self.skeleton.from_numpy(skeleton_np)
        self.eulers = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_skeleton,))
        self.Hessian = ti.field(ti.f32, shape=(self.num_T * 6, self.num_T * 6))
        self.fill_Hessian()
        self.Grad = ti.field(ti.f32, shape=(self.num_T * 6,))
        self.fill_Grad()
        B_np = def_grad3D(mesh.points, mesh.cells[1].data.astype("int32")).tocoo()
        K = ti.linalg.SparseMatrixBuilder(9 * self.num_T, 3 * self.num_V, max_num_triplets=B_np.col.shape[0])
        self.fill_B(K, B_np.row, B_np.col, B_np.data)
        self.B = K.build()


    @ti.kernel
    def fill_eulers(self, eus: ti.types.ndarray(dtype=ti.math.vec3, ndim=1)):
        for i in range(self.eulers.shape[0]):
            self.eulers[i] = ti.Vector([eus[i][0], eus[i][1], eus[i][2]])

    @ti.kernel
    def fill_Hessian(self):
        for i in range(self.num_T):
            if self.material_model == MaterialModel.arap:
                self.Hessian[6 * i, 6 * i] = 2 * self.mu * 1
                self.Hessian[6 * i + 1, 6 * i + 1] = 2 * self.mu * 1
                self.Hessian[6 * i + 2, 6 * i + 2] = 2 * self.mu * 1
                self.Hessian[6 * i + 3, 6 * i + 3] = 2 * self.mu * 2
                self.Hessian[6 * i + 4, 6 * i + 4] = 2 * self.mu * 2
                self.Hessian[6 * i + 5, 6 * i + 5] = 2 * self.mu * 2

    @ti.kernel
    def fill_Grad(self):
        for i in range(self.num_T):
            if self.material_model == MaterialModel.arap:
                self.Grad[6 * i] = -2 * self.mu * 1
                self.Grad[6 * i + 1] = -2 * self.mu * 1
                self.Grad[6 * i + 2] = -2 * self.mu * 1
                self.Grad[6 * i + 3] = 0
                self.Grad[6 * i + 4] = 0
                self.Grad[6 * i + 5] = 0

    @ti.kernel
    def fill_B(self, K: ti.types.sparse_matrix_builder(), row: ti.types.ndarray(), col: ti.types.ndarray(), val: ti.types.ndarray()):
        for i in range(col.shape[0]):
            K[row[i], col[i]] += val[i]


if __name__ == '__main__':
    ti.init(arch=ti.cpu)
    levels = np.array([30])
    mfem = MFEMSolver(config.MODEL_PATH,
                      config.SKELETON_PATH,
                      config.HIERARCHY_PATH,
                      config.MU,
                      levels)
    # print(mfem.B[0, 9151])

# if __name__ == '__main__':
#     print("Reading data...")
#     start_data = time()
#     ########## BEAM DATA ##############
#     # read beam model
#     mesh = meshio.read("../data/beam.mesh")
#     Vt = mesh.points
#     print(mesh.points.shape[0])
#     print(mesh.cells[1].data.shape[0])
#     Tt = mesh.cells[1].data.astype("int32")
#     # make beam skeleton
#     hier = np.array([0, 1, 2])
#     handles = np.array([[-4.5, 0.0, 0.0], [0.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
#     # level of beam
#     b_levels = np.array([[30]]).astype(int)
#     # load beam samples
#     py_P = scipy.io.loadmat("../data/P_beam.mat")['P']
#     py_PI = scipy.io.loadmat("../data/PI_beam.mat")['PI']
#     # beam eulers
#     eulers = np.array([[0.0, 0.0, 0.0], [-np.pi / 3, 0.0, 0.0], [0.0, 0.0, 0.0]])
#     # eulers = np.array([[0.0, 0.0, 0.0], [0.0, -np.pi/2, 0.0], [0.0, 0.0, 0.0]])
#     ########## BEAM DATA END ##############
#
#     ########## HUMAN DATA ##############
#     # # read human model
#     # mesh = meshio.read("../data/output_mfem1.msh")
#     # Vt = mesh.points
#     # Tt = mesh.cells[0].data.astype("int32")
#     # # load human skeleton
#     # handlest = scipy.io.loadmat('../data/handles.mat')
#     # hiert = scipy.io.loadmat("../data/hierarchy.mat")
#     # handles = handlest['position']  ## to access the mat from the dict use: handles['position'] (numHandels, 3)
#     # hier = hiert['hierarchy'][:, 1]
#     # # level of human
#     # b_levels = np.array([[50]]).astype(int)
#     # # load human samples
#     # py_P = scipy.io.loadmat("../data/P.mat")['P']  ## to access the mat from the dict use: Pt['P']
#     # py_PI = scipy.io.loadmat("../data/PI.mat")['PI']
#     # # load human eulers
#     # eulers = fill_euler(handles)
#     ########## HUMAN DATA END ##############
#
#     end_data = time()
#     print("Done reading data...")
#     print("Reading data took {0} seconds.".format(end_data - start_data))
#     print("Vertices: ", len(Vt), Vt.shape)
#     print("Tets: ", len(Tt), Tt.shape)
#
#     print("Getting grad...")
#     B = def_grad3D(Vt, Tt)
#     mu = 100  # material properties
#     k_bc = 10000  # stiffness
#     s = np.zeros((6 * Tt.shape[0], 1))
#     print("Getting energy...")
#     grad = linear_tet3dmesh_arap_ds(Vt, Tt, s, mu)
#     hess = linear_tet3dmesh_arap_ds2(Vt, Tt, s, mu)
#
#     l = b_levels.shape[0]
#
#     print("Computing closest index...")
#     weight = closest_index_np(Vt, py_P)
#     print("Building the U matrix...")
#     Ut, NN = build_U(weight, b_levels, l, py_P, Vt)
#
#     midpoints_np = np.zeros((handles.shape[0] - 1, 3))
#     for i in range(handles.shape[0]):
#         if hier[i] == 0:
#             continue
#         else:
#             midpoints_np[i - 1, :] = (handles[i, :] + handles[int(hier[i]) - 1, :]) / 2
#
#     print("Computing tet assignments...")
#     fAssign = tetAssignment_py(Vt, Tt, midpoints_np)
#
#     # We move the handles here, so let's consider that we start the sim from around this point
#     start_all_sim = time()
#
#     print("Running FWD Kinematics...")
#     newR, new_handles = forward_kinematics_np(handles, hier, eulers)
#     rows_T = Tt.shape[0]
#     R = np.zeros((rows_T, 9))
#     # assigning rotations to each tet according to fAssign info
#     for i in range(rows_T):
#         R[i, :] = newR[int(fAssign[i]), :]
#
#     print("Getting the R_Mat...")
#     R_mat_py = block_R3d(R)  # python function
#
#     num_handles = handles.shape[0]
#     num_midpoints = midpoints_np.shape[0]
#     Ht = np.zeros((num_handles + num_midpoints, 3))
#     Ht[:num_handles, :] = handles
#     Ht[num_handles:, :] = midpoints_np
#
#     print("Computing the pinned verts...")
#     vAssign = pinvert_np(Vt, Ht)
#     I = np.eye(3)
#     q = igl2bart(Vt)
#
#     pinned_mat = sps.lil_matrix((3 * Ht.shape[0], 3 * Vt.shape[0]))
#     for i in range(Ht.shape[0]):
#         pinned_mat[3 * i:3 * (i + 1), 3 * int(vAssign[i]):3 * int(vAssign[i])+3] = I
#
#     midpoints = np.zeros((new_handles.shape[0] - 1, 3))
#     for i in range(new_handles.shape[0]):
#         if hier[i] == 0:
#             continue
#         else:
#             midpoints[i - 1, :] = (new_handles[i, :] + new_handles[hier[i] - 1, :]) / 2
#
#     new_new_handles = np.zeros((new_handles.shape[0] + midpoints.shape[0], 3))
#     new_new_handles[:new_handles.shape[0], :] = new_handles
#     new_new_handles[new_handles.shape[0]:, :] = midpoints
#
#     pinned_b = igl2bart(new_new_handles)
#
#     nq = q.shape[0]
#     ns = s.shape[0]
#     nlambda = 9 * Tt.shape[0]
#     start_j = time()
#     if config.USE_SVD:
#         J = compute_J_SVD(R_mat_py, B)  # cupy/numpy function
#     else:
#         J = compute_J(R_mat_py, B)
#     end_j = time()
#     print("J computed in {0} seconds".format(end_j-start_j))
#     A = k_bc * pinned_mat.T @ pinned_mat + J.T @ hess @ J
#     b = k_bc * pinned_mat.T @ pinned_b - J.T @ grad
#     b = np.squeeze(b)
#
#     UTAU = []
#     for i in range(len(Ut)):
#         if i == 0:
#             UTAU.append(Ut[i].T.dot(A).dot(Ut[i]) + NN)
#         else:
#             UTAU.append(Ut[i].T.dot(UTAU[i - 1]).dot(Ut[i]))
#
#     normVal = float('inf')
#     itr = 0
#     tol = 1e-5
#     sol = np.zeros(b.shape)
#     rows, cols = A.nonzero()
#     itr_num = 3
#     l = len(Ut) - 1
#     U, L = A_L_sum_U_np(A)  # python
#     print("Starting the multigrid solve...")
#     if config.USE_CUPY:
#         print("Running the CuPy MG solver...")
#         import cupy as cp
#         import cupyx as cpx
#         from gs_cp import A_L_sum_U_cp
#         from v_cycle_cp import v_cycle_cp
#         # convert the elements we will use to cupy and thus send to the GPU
#         sol_cp = cp.asarray(sol)
#         A_cp = cpx.scipy.sparse.csr_matrix(A)
#         U_cp, L_cp = A_L_sum_U_cp(A_cp)
#         b_cp = cp.asarray(b)
#         UTAU_cp = [cpx.scipy.sparse.csc_matrix(u_m) for u_m in UTAU]
#         Ut_cp = [cpx.scipy.sparse.csr_matrix(u_m) for u_m in Ut]
#         start_mg = time()
#         while normVal > tol:
#             sol_old_cp = sol_cp
#             sol_cp = v_cycle_cp(A_cp, U_cp, L_cp, b_cp, UTAU_cp, Ut_cp, l, itr_num, sol_old_cp, debug=False)
#             normVal = np.linalg.norm(b_cp - A_cp.dot(sol_cp))
#             print("error: ", normVal)
#         sol = cp.asnumpy(sol_cp)
#     else:
#         print("Running the NumPy MG solver...")
#         start_mg = time()
#         while normVal > tol:
#             sol_old = sol
#             # start = time()
#             sol = v_cycle_py(A, U, L, b, UTAU, Ut, l, itr_num, sol_old, debug=False)
#             normVal = np.linalg.norm(b - A.dot(sol))
#             # end = time()
#             # print('V_cycle, itr', itr, 'in ', round(end - start, 3), 's')
#             print("error: ", normVal)
#     print("Finished the multigrid solve")
#     end_all_sim = time()
#     print(np.linalg.norm(b - A.dot(sol)))
#     print(sol.shape)
#     print("The multigrid solve took {0} seconds.".format(end_all_sim-start_mg))
#     print("The complete sim took {0} seconds.".format(end_all_sim - start_all_sim))
#
#     # visualization
#     ps.set_program_name("mfem")
#     ps.set_ground_plane_mode("shadow_only")
#     ps.init()
#     ps_vol = ps.register_volume_mesh("test volume mesh", sol.reshape(-1, 3), tets=Tt)
#     ps.show()
