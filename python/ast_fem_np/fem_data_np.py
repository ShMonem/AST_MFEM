# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

import meshio
import scipy
import os
import numpy as np
from python.common.skeleton import Skeleton
import python.ast_fem_np.config as config
from time import time
from python.common.def_grad3d import def_grad3d
from python.common.energies import linear_tet3dmesh_arap_ds, linear_tet3dmesh_arap_ds2, linear_tet3dmesh_corot_ds, linear_tet3dmesh_corot_ds2
from python.ast_fem_np.build_u_np import build_u
from python.ast_fem_np.closest_index_np import closest_index, point_to_line_distance
from python.ast_fem_np.tet_assignment_np import tet_assignment
from python.ast_fem_np.pinvert_np import pinvert


class FEMData:
    def __init__(self, obj_name, load_skel=False, use_eulers=True, visibility=True):
        # UI properties
        self.visibility = visibility
        # Data specific properties
        self.handles_pos = None
        self.handles_rot = None
        self.eulers = None
        self.debug = config.DEBUG
        self.energy_type = 'corot'
        self.ps_vol = None
        # Deformation gradient
        self.B = None
        # Energy gradient
        self.grad = None
        # Energy hessian
        self.hess = None
        # Initial pinned positions
        self.init_pinned_pos = None
        # Initial pinned vertex ids
        self.init_pin_verts = None
        # Offsets for the pinned verts
        self.pinned_vert_offsets = None
        # Path properties
        self.obj_name = obj_name
        self.obj_root_path = os.path.abspath(f"../../data/{self.obj_name}")
        if not os.path.isdir(self.obj_root_path):
            raise FileNotFoundError(f"The folder {self.obj_root_path} does not exist!")
        # We need to load the mesh now and the tets, since they are different for different formats,
        # and we use them creating material properties.
        try:
            mesh = meshio.read(os.path.join(self.obj_root_path, f'{self.obj_name}.msh'))
            self.tets = mesh.cells[0].data.astype("int32")
        except:
            mesh = meshio.read(os.path.join(self.obj_root_path, f'{self.obj_name}.mesh'))
            self.tets = mesh.cells[1].data.astype("int32")
        self.verts = mesh.points
        # Material properties
        # la is only for the corot energy
        self.la = None
        # s is only for the arap energy
        self.s = None
        # both energies use mu and k_bc
        self.mu = None
        self.k_bc = None  # stiffness
        if self.energy_type == 'arap':
            self.s = np.zeros((6 * self.tets.shape[0], 1))
            self.k_bc = 100000
            self.mu = np.ones([self.tets.shape[0], 1]) * 1000
            if os.path.isfile(os.path.join(self.obj_root_path, f'{self.obj_name}_rigid_tets.npy')):
                rigid_tets = np.load(os.path.join(self.obj_root_path, f'{self.obj_name}_rigid_tets.npy'))
                self.mu[rigid_tets] = 100000
        elif self.energy_type == 'corot':
            self.k_bc = 1000000000  # stiffness
            self.la = np.ones([self.tets.shape[0], 1]) * 2.1724e+07
            self.mu = np.ones([self.tets.shape[0], 1]) * 2.4138e+06
            if os.path.isfile(os.path.join(self.obj_root_path, f'{self.obj_name}_rigid_tets.npy')):
                rigid_tets = np.load(os.path.join(self.obj_root_path, f'{self.obj_name}_rigid_tets.npy'))
                # self.mu[rigid_tets] = 1e10
                # self.la[rigid_tets] = 1e11
        # Skeleton stuff
        self.hier = None
        if use_eulers:
            self.eulers = scipy.io.loadmat(os.path.join(self.obj_root_path, f'{self.obj_name}_eulers.mat'))['eulers']
            self.eulers[0] = np.array([0, 0, 0])
        # Build the internal skeleton
        if load_skel:
            self.overwrite = True
            self.load_skeleton()
        else:
            self.overwrite = False
            self.load_skeleton()
        # Run pre-computation of initial structures and associations
        self.do_precompute()
        # Testing custom tet assignment - we overwrite the results from do_precompute here.
        # NOTE: If you do not want to use this, after commenting out, make sure to uncomment a line in set_bones!
        # TODO: This needs to be a rotation clustering selection option
        # self.tet_assign = np.load(r'C:\Users\DimitryKachkovski\git\personal\AST_MFEM\data\human\human_tet_maya_weights.npy')
        # self.tet_assign = np.load(os.path.join(self.obj_root_path, f'{self.obj_name}_tet_maya_weights.npy'))

    def load_skeleton(self):
        self.skeleton = Skeleton()
        self.skeleton.load_skeleton(os.path.join(self.obj_root_path, f'{self.obj_name}_skel_ws_tms.npy'),
                                    os.path.join(self.obj_root_path, f'{self.obj_name}_skel_hier.json'),
                                    os.path.join(self.obj_root_path, f'{self.obj_name}_skel_names.npy'),
                                    transpose=True)
        # This sets the transforms twice, but it sets the other parameters, too
        self.handles_pos = self.skeleton.rest_skel[:, -1, :3]
        self.hier = self.skeleton.hier[:, 1]
        self.set_bones(self.skeleton.skel_tms)

    def set_bones(self, bones_tms):
        self.skeleton.set_bones(bones_tms)
        if self.overwrite:
            if self.eulers is not None:
                self.handles_rot, self.handles_pos = self.forward_kinematics(self.skeleton.get_positions(),
                                                                             self.eulers)
            else:
                self.handles_pos = self.skeleton.get_positions()
                self.handles_rot = np.matmul(np.transpose(self.skeleton.inv_rest_skel[:, :3, :3], axes=(0, 1, 2)),
                                             self.skeleton.get_rotations()).reshape((-1, 9))
                # Uncomment the next line if we do not use the maya assigned weights.
                self.handles_rot = self.get_parent_rots(self.handles_rot)

    def forward_kinematics(self, in_positions, in_eulers):
        n = in_positions.shape[0]
        new_handles = np.zeros_like(in_positions)
        newR = np.zeros((n - 1, 3, 3))
        newR[:, :] = np.eye(3)
        T = np.zeros((n, 3, 3))
        T[:, :] = np.eye(3)
        for i in range(n):
            loc_rot_mtx = self.compute_rotation(in_eulers[i, 0], in_eulers[i, 1], in_eulers[i, 2])  # (3, 3)
            if self.hier[i] == 0:  # what does this case mean? # this is the root node
                T[i] = loc_rot_mtx  # F order
                new_handles[i] = in_positions[i]
            else:
                newR[i - 1] = T[int(self.hier[i] - 1)]
                Tp = T[int(self.hier[i] - 1)].T
                vecl = in_positions[i] - in_positions[int(self.hier[i] - 1)]
                vec = new_handles[int(self.hier[i] - 1)] + Tp @ vecl
                new_handles[i, :] = vec
                temp = (Tp @ loc_rot_mtx).astype(float)
                T[i] = temp.T
        return newR.reshape((-1, 9)), new_handles

    @staticmethod
    def compute_rotation(in_euler_angles):
        '''
        Simple conversion from euler angles to a matrix. Don't ask me about the ordering...
        '''
        alpha, beta, gamma = in_euler_angles
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

    def get_parent_rots(self, in_rotations):
        '''
        This method finds the parent rotations for each joint. No math is done, but when using midpoints,
        we need to use parent rotations, so this sorting is necessary.
        '''
        n = in_rotations.shape[0]
        out_rots = np.zeros((n - 1, 9))
        for i in range(n):
            if self.hier[i] == 0:  # 0 means this is a root node
                pass
            else:
                out_rots[i - 1] = in_rotations[int(self.hier[i] - 1)]
        return out_rots

    def do_precompute(self):
        '''
        This does a one-time precomputation step to find static variables used in the MFem formulation. Specifically,
        we compute:
        - B: the deformation gradient
        - grad: the energy gradient
        - hess: the energy hessian
        - tet_assign: the association of tets to midpoints (our default rotation clustering)
        - init_pinned_pos: the initial pinned positions - joints + midpoints. We only use this locally to pin vertices
                           with the solver only using this for its shape dimensions. So we keep it.
        - init_pin_verts: the idis of vertices that get pinned to the handles and midpoints. These IDs are crucial in
                          forming the pinned_mat matrix for our main equation.
        '''

        if self.debug:
            print("Starting one time setup process...")
            start_setup = time()

        if self.debug:
            start = time()
        self.B = def_grad3d(self.verts, self.tets)
        if self.debug:
            end = time()
            print(f"Getting grad took {end - start} seconds.")
        if self.debug:
            start = time()
        if self.energy_type == 'arap':
            self.grad = linear_tet3dmesh_arap_ds(self.verts, self.tets, self.s, self.mu)
            self.hess = linear_tet3dmesh_arap_ds2(self.verts, self.tets, self.s, self.mu)
        elif self.energy_type == 'corot':
            # We need to crank the stiffness up a lot for co-rotational energy
            self.grad = linear_tet3dmesh_corot_ds(self.mu, self.la, self.tets)
            self.hess = linear_tet3dmesh_corot_ds2(self.mu, self.la, self.tets)
        else:
            raise NotImplementedError(f"The {self.energy_type} energy is not supported.")
        if self.debug:
            end = time()
            print(f"Getting energy took {end - start} seconds.")

        midpoints_np = np.zeros((self.handles_pos.shape[0] - 1, 3))
        for i in range(self.handles_pos.shape[0]):
            if self.hier[i] == 0:
                continue
            else:
                midpoints_np[i - 1, :] = \
                    (self.handles_pos[i, :] + self.handles_pos[int(self.hier[i]) - 1, :]) / 2

        if self.debug:
            start = time()

        # self.tet_assign = tet_assignment(self.verts, self.tets, midpoints_np)
        barycenters = self.verts[self.tets].sum(1) / 4.0
        self.tet_assign = point_to_line_distance(barycenters, self.handles_pos, self.hier)

        if self.debug:
            end = time()
            print(f"Computing tet assignments took {end - start} seconds.")

        num_handles = self.handles_pos.shape[0]
        num_midpoints = midpoints_np.shape[0]
        self.init_pinned_pos = np.zeros((num_handles + num_midpoints, 3))
        self.init_pinned_pos[:num_handles, :] = self.handles_pos
        self.init_pinned_pos[num_handles:, :] = midpoints_np

        if self.debug:
            start = time()

        self.init_pin_verts = pinvert(self.verts, self.init_pinned_pos)
        self.pinned_vert_offsets = self.verts[self.init_pin_verts] - self.init_pinned_pos

        if self.debug:
            end = time()
            print(f"Computing the pinned verts took {end - start} seconds.")

        if self.debug:
            print(f"Finished the one time setup.\nThe setup took {end - start_setup} seconds.")

    # Note this may be broken atm
    def init_multi_grid(self):
        reduction_size = 50
        self.reduction_size = reduction_size
        # Number of levels for multi grid
        self.b_levels = np.array([[self.reduction_size]]).astype(int)
        # The reduced subset of points for multi grid
        self.P = scipy.io.loadmat(os.path.join(self.obj_root_path, f'{self.obj_name}_P.mat'))['P']
        # Defining variables used by the solver
        self.Ut = None
        self.NN = None
        self.mg_initialized = False
        # Number of levels for multi grid
        self.b_levels = np.array([[self.reduction_size]]).astype(int)
        # The reduced subset of points for multi grid
        self.P = scipy.io.loadmat(os.path.join(self.obj_root_path, f'{self.obj_name}_P.mat'))['P']
        l = self.b_levels.shape[0]
        # Assigning verts to the reduced elements
        self.ver_assign = closest_index(self.verts, self.P)
        self.Ut, self.NN = build_u(self.ver_assign, self.b_levels, l, self.P, self.verts)
        self.mg_initialized = True
