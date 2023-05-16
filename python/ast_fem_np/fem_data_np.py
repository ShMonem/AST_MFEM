import meshio
import scipy
import os
import numpy as np
from fill_euler_np import fill_euler


class FEMData:
    def __int__(self, obj_root_path, hier=None, handles=None, reduction_size=50):
        mesh_path = os.path.join(obj_root_path, 'tet_mesh.msh')
        joints_path = os.path.join(obj_root_path, 'handles.mat')
        hier_path = os.path.join(obj_root_path, 'hierarchy.mat')
        p_path = os.path.join(obj_root_path, 'P.mat')
        pi_path = os.path.join(obj_root_path, 'PI.mat')
        mesh = meshio.read(mesh_path)
        self.verts = mesh.points
        self.tets = mesh.cells[0].data.astype("int32")
        self.handles_dict = scipy.io.loadmat(joints_path)
        self.hier_dict = scipy.io.loadmat(hier_path)
        self.handles_pos = self.handles_dict['position']  ## to access the mat from the dict use: handles['position'] (numHandels, 3)
        self.hier = self.hier_dict['hierarchy'][:, 1]
        self.b_levels = np.array([[reduction_size]]).astype(int)
        self.P = scipy.io.loadmat(p_path)['P']  ## to access the mat from the dict use: Pt['P']
        self.PI = scipy.io.loadmat(pi_path)['PI']
        self.eulers = fill_euler(self.handles_pos)