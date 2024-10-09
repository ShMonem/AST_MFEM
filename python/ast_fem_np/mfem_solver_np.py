# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

import scipy
import numpy as np
import scipy.sparse as sps
from time import time
from python.ast_fem_np.block_r3d_np import block_r3d
from python.common.igl2bart import igl2bart
from python.ast_fem_np.compute_j_np import compute_J_SVD
from python.ast_fem_np.v_cycle_np import v_cycle
from python.ast_fem_np.gauss_seidel_np import A_L_sum_U
from python.utils.utils import compute_mesh_volume
from python.ast_fem_np.pinvert_np import read_pinvert
class MFEMSolver:
    def __init__(self, fem_data_node, use_mg=False, debug=False):
        self.obj_data = fem_data_node
        self.use_mg = use_mg
        self.debug = debug
        self.curr_frame = None
        if self.obj_data.pin_inds_file is not None:
            self.pinned_inds = read_pinvert(self.obj_data.pin_inds_file)
            self.pinned_points_pos = self.obj_data.verts[self.pinned_inds]  # positions of pinned verts, if user defined pinned verts is used

    def solve(self):
        if self.debug:
            start_all_sim = time()

        joint_rots = self.obj_data.handles_rot
        joint_pos = self.obj_data.handles_pos
        J = self.compute_J(joint_rots)
        pinned_mat = self.build_pinned_mat()
        # transformed_vert_offsets = np.matmul(some_local_rots, self.obj_data.pinned_vert_offsets, axis=1)
        if self.obj_data.pin_inds_file is not None:
            pinned_b = self.compute_pinned_b_w_given_pinning(joint_pos)
        else:
            pinned_b = self.compute_pinned_b(joint_pos)
        # Factorized A
        A = self.obj_data.k_bc * pinned_mat.T @ pinned_mat + J.T @ self.obj_data.hess @ J
        b = np.squeeze(self.obj_data.k_bc * pinned_mat.T @ pinned_b - J.T @ self.obj_data.grad)

        if self.use_mg:
            self.obj_data.init_multi_grid()
            sol = self.multi_grid_solve(A, b)
        else:
            sol = self.direct_solve(A, b)
        if self.debug:
            end_all_sim = time()
            print(f"The complete sim took {end_all_sim-start_all_sim} seconds.")

        if self.obj_data.pin_inds_file is not None:
            self.pinned_points_pos = sol.reshape(-1, 3)[self.pinned_inds]
        print(compute_mesh_volume(sol.reshape(-1, 3), self.obj_data.faces))
        return sol

    def build_pinned_mat(self):
        pinned_mat = sps.lil_matrix((3 * self.obj_data.init_pinned_pos.shape[0], 3 * self.obj_data.verts.shape[0]))
        for i in range(self.obj_data.init_pinned_pos.shape[0]):
            r_start, r_end = (3 * i, 3 * (i + 1))
            col_start = 3 * int(self.obj_data.init_pin_verts[i])
            col_end = 3 * int(self.obj_data.init_pin_verts[i]) + 3
            pinned_mat[r_start:r_end, col_start:col_end] = np.eye(3)
        return pinned_mat

    def compute_pinned_b(self, joint_pos):
        midpoints = np.zeros((joint_pos.shape[0] - 1, 3))
        for i in range(joint_pos.shape[0]):
            if self.obj_data.hier[i] == 0:
                continue
            else:
                midpoints[i - 1, :] = (joint_pos[i, :] + joint_pos[self.obj_data.hier[i] - 1, :]) / 2

        pinned_positions = np.zeros((joint_pos.shape[0] + midpoints.shape[0], 3))
        pinned_positions[:joint_pos.shape[0], :] = joint_pos
        pinned_positions[joint_pos.shape[0]:, :] = midpoints

        # We have to account for the initial offsets when we pin certain verts. Thus, we have to skin them to the
        # relevant joints and midpoints used in the pinned_positions.
        if self.obj_data.handles_rot.shape[0] != midpoints.shape[0]:
            local_rots = self.obj_data.get_parent_rots(self.obj_data.handles_rot)
            offsets = self.obj_data.pinned_vert_offsets.copy()
            offsets[:self.obj_data.handles_pos.shape[0]] = np.einsum('...ij,...j',
                                                    self.obj_data.handles_rot.reshape(-1, 3, 3).transpose((0, 2, 1)),
                                                    offsets[:self.obj_data.handles_pos.shape[0]])
            offsets[self.obj_data.handles_pos.shape[0]:] = np.einsum('...ij,...j',
                                                                local_rots.reshape(-1, 3, 3).transpose((0, 2, 1)),
                                                                offsets[self.obj_data.handles_pos.shape[0]:])
        else:
            main_rots = np.matmul(np.transpose(self.obj_data.skeleton.inv_rest_skel[:, :3, :3], axes=(0, 1, 2)),
                                               self.obj_data.skeleton.get_rotations())
            offsets = self.obj_data.pinned_vert_offsets.copy()
            offsets[:self.obj_data.handles_pos.shape[0]] = np.einsum('...ij,...j',
                                                                     main_rots.transpose((0, 2, 1)),
                                                                     offsets[:self.obj_data.handles_pos.shape[0]])
            offsets[self.obj_data.handles_pos.shape[0]:] = np.einsum('...ij,...j',
                                                    self.obj_data.handles_rot.reshape(-1, 3, 3).transpose((0, 2, 1)),
                                                    offsets[self.obj_data.handles_pos.shape[0]:])

        pinned_positions = pinned_positions + offsets

        pinned_b = igl2bart(pinned_positions)
        return pinned_b

    def compute_pinned_b_w_given_pinning(self, joint_pos):
        pinned_points = self.pinned_points_pos

        pinned_positions = np.zeros((joint_pos.shape[0] + pinned_points.shape[0], 3))
        pinned_positions[:joint_pos.shape[0], :] = joint_pos
        pinned_positions[joint_pos.shape[0]:, :] = pinned_points

        # We have to account for the initial offsets when we pin certain verts. Thus, we have to skin them to the
        # relevant joints and pinned_points used in the pinned_positions.
        if self.obj_data.handles_rot.shape[0] != pinned_points.shape[0]:
            local_rots = self.obj_data.get_parent_rots(self.obj_data.handles_rot)
            local_rots = np.repeat([[1, 0, 0, 0, 1, 0, 0, 0, 1]], pinned_points.shape[0], axis=0)
            offsets = self.obj_data.pinned_vert_offsets.copy()
            offsets[:self.obj_data.handles_pos.shape[0]] = np.einsum('...ij,...j',
                                                    self.obj_data.handles_rot.reshape(-1, 3, 3).transpose((0, 2, 1)),
                                                    offsets[:self.obj_data.handles_pos.shape[0]])
            offsets[self.obj_data.handles_pos.shape[0]:] = np.einsum('...ij,...j',
                                                                local_rots.reshape(-1, 3, 3).transpose((0, 2, 1)),
                                                                offsets[self.obj_data.handles_pos.shape[0]:])
        else:
            main_rots = np.matmul(np.transpose(self.obj_data.skeleton.inv_rest_skel[:, :3, :3], axes=(0, 1, 2)),
                                               self.obj_data.skeleton.get_rotations())
            offsets = self.obj_data.pinned_vert_offsets.copy()
            offsets[:self.obj_data.handles_pos.shape[0]] = np.einsum('...ij,...j',
                                                                     main_rots.transpose((0, 2, 1)),
                                                                     offsets[:self.obj_data.handles_pos.shape[0]])
            offsets[self.obj_data.handles_pos.shape[0]:] = np.einsum('...ij,...j',
                                                    self.obj_data.handles_rot.reshape(-1, 3, 3).transpose((0, 2, 1)),
                                                    offsets[self.obj_data.handles_pos.shape[0]:])

        pinned_positions = pinned_positions + offsets

        pinned_b = igl2bart(pinned_positions)
        return pinned_b
    def compute_J(self, joint_rots):
        R = self.get_tet_rots(joint_rots)

        if self.debug:
            start = time()

        R_mat_py, R_mat_blocks = block_r3d(R, blocks=True)  # python function
        if self.debug:
            end = time()
            print(f"block_r3d took {end - start} seconds.")

        if self.debug:
            start = time()
        J = compute_J_SVD(R_mat_py, self.obj_data.B, r_mat_blocks=R_mat_blocks)  # cupy/numpy function
        if self.debug:
            end = time()
            print(f"J computed in {end - start} seconds.")
        return J

    def get_tet_rots(self, in_joint_rots):
        num_tets = self.obj_data.tets.shape[0]
        tet_rots = np.zeros((num_tets, 9))
        # assigning rotations to each tet according to fAssign info
        for i in range(num_tets):
            tet_rots[i, :] = in_joint_rots[self.obj_data.tet_assign[i], :]
        return tet_rots

    def direct_solve(self, A, b):
        if self.debug:
            print("Using the direct solver...")
            start_solve = time()
        sol = scipy.sparse.linalg.spsolve(A, b)
        if self.debug:
            end_solve = time()
            print(f"The solve took {end_solve-start_solve} seconds.")
        return sol

    # Note this may be broken atm
    def multi_grid_solve(self, A, b, tol=1e-5, itr_num=1):
        UTAU = []
        for i in range(len(self.obj_data.Ut)):
            if i == 0:
                UTAU.append(self.obj_data.Ut[i].T.dot(A).dot(self.obj_data.Ut[i]) + self.obj_data.NN)
            else:
                UTAU.append(self.obj_data.Ut[i].T.dot(UTAU[i - 1]).dot(self.obj_data.Ut[i]))

        norm_val = float('inf')
        sol = np.zeros(b.shape)
        l = len(self.obj_data.Ut) - 1
        U, L = A_L_sum_U(A)  # python
        if self.debug:
            print("Using the MG solver...")
            start_solve = time()
        while norm_val > tol:
            sol_old = sol
            sol = v_cycle(A, U, L, b, UTAU, self.obj_data.Ut, l, itr_num, sol_old, debug=False)
            norm_val = np.linalg.norm(b - A.dot(sol))
            if self.debug:
                print("error: ", norm_val)
        if self.debug:
            end_solve = time()
            print(f"The solve took {end_solve-start_solve} seconds.")
        return sol
