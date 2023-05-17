import meshio
import scipy
import os
import numpy as np
from


class FEMData:
    def __init__(self, obj_name, load_skel_anim=False, reduction_size=50):
        obj_root_path = os.path.abspath(f"../../data/{obj_name}")
        if not os.path.isdir(obj_root_path):
            raise FileNotFoundError(f"The folder {obj_root_path} does not exist!")
        # harcoding to try to load a .msh file, if not, then a .mesh file
        try:
            mesh = meshio.read(os.path.join(obj_root_path, f'{obj_name}.msh'))
            self.tets = mesh.cells[0].data.astype("int32")
        except:
            mesh = meshio.read(os.path.join(obj_root_path, f'{obj_name}.mesh'))
            self.tets = mesh.cells[1].data.astype("int32")
        self.verts = mesh.points
        self.handles_dict = scipy.io.loadmat(os.path.join(obj_root_path, f'{obj_name}_handles.mat'))
        self.hier_dict = scipy.io.loadmat(os.path.join(obj_root_path, f'{obj_name}_hierarchy.mat'))
        self.handles_pos = self.handles_dict['position']  ## to access the mat from the dict use: handles['position'] (numHandels, 3)
        self.hier = self.hier_dict['hierarchy'][:, 1]
        self.b_levels = np.array([[reduction_size]]).astype(int)
        self.P = scipy.io.loadmat(os.path.join(obj_root_path, f'{obj_name}_P.mat'))['P']  ## to access the mat from the dict use: Pt['P']
        # self.PI = scipy.io.loadmat(os.path.joint(obj_root_path, f'{obj_name}_PI.mat'))['PI']  # this seems unused
        self.eulers = scipy.io.loadmat(os.path.join(obj_root_path, f'{obj_name}_eulers.mat'))['eulers']
        if load_skel_anim:
            self.load_skel_anim(os.path.join(obj_root_path, f'{obj_name}_skel_anim.npy'))

    def load_skel_anim(self, skel_anim_path):
        self.skeleton = Skeleton()


if __name__ == "__main__":
    obj_name = 'human'
    FEMData(obj_name)
