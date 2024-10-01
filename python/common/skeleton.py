# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

import numpy as np
from python.common.bone import Bone
import json


class Skeleton:

    def __init__(self):
        self.bones = []
        # skel tms are referenced by each bone, not copied. Be careful how you mess with it!
        self.skel_tms = None
        self.skel_names = None
        self.bone_handles = list()
        self.visibility = False
        self.hier = None
        self.par_child_tree = None

    def add(self, bone):
        self.bones.append(bone)

    def load_skeleton(self, file_transform, file_hierarchy, file_namelist, transpose=False):
        # read rest shape transformation from file: #bones by 4 by 4
        self.skel_tms = np.load(file_transform)
        self.set_rest_skel(self.skel_tms)
        with open(file_hierarchy, 'r') as f:
            skeleton_hierarchy = json.load(f)
        namelist = np.load(file_namelist)
        self.skel_names = namelist
        self.par_child_tree = self.build_child_parent_tree(skeleton_hierarchy)
        self.hier = self.build_child_parent_index(skeleton_hierarchy, namelist)
        skel = list()
        # load all bones, store lengths to each child, we will draw the cones to each child
        for idx, bone_name in enumerate(self.par_child_tree):
            bone_tm = self.skel_tms[idx].T if transpose else self.skel_tms[idx]
            lens = list()
            children_idxs = list()
            parent = self.par_child_tree[bone_name]['parent']
            for child in self.par_child_tree[bone_name]['children']:
                child_idx = np.where(namelist == child)[0][0]
                children_idxs.append(child_idx)
                length = np.linalg.norm(self.skel_tms[child_idx, 3, 0:3] - self.skel_tms[idx, 3, 0:3])
                lens.append(length)
            parent_idx = -1 if parent == 'root' else np.where(namelist == parent)[0][0]
            b = Bone(lens, bone_tm, self.skel_tms, idx, parent_idx, children_idxs)
            skel.append(b)
        self.bones = skel

    @staticmethod
    def build_child_parent_tree(hier):
        # We will be outputting a dict in the form of {"jnt_name":{"parent":"jnt_name", "children":[]}}
        out_dict = dict()
        # first pass for parents
        for elem in hier:
            jnt_dict = {"parent": hier[elem], "children": []}
            out_dict[elem] = jnt_dict
        # now we add children
        for elem in out_dict:
            par = out_dict[elem]['parent']
            if par != 'root':
                out_dict[par]['children'].append(elem)
        return out_dict

    @staticmethod
    def build_child_parent_index(hier, name_list):
        out_array = list()
        for elem in hier:
            elem_id = np.where(name_list == elem)[0][0] + 1
            par_id = 0 if hier[elem] == 'root' else np.where(name_list == hier[elem])[0][0] + 1
            out_array.append([elem_id, par_id])
        return np.array(out_array)

    def set_bones(self, bones_tms):
        # we use [:] to update the base tms which are referenced by bones.
        self.skel_tms[:] = bones_tms

    def get_positions(self):
        return self.skel_tms[:, -1, :3]

    def get_rotations(self):
        # return np.transpose(self.skel_tms[:, :3, :3], axes=(0, 2, 1))
        return self.skel_tms[:, :3, :3]

    def set_rest_skel(self, skel_tms):
        self.inv_rest_skel = np.linalg.inv(skel_tms)
        self.rest_skel = skel_tms.copy()

