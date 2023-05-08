import numpy as np
from bone import Bone
import json


class Skeleton:

    def __init__(self):
        self.skeleton = []

    def add(self, bone):
        self.skeleton.append(bone)

    def load_skeleton(self, file_transform, file_hierarchy, file_namelist):
        # read rest shape transformation from file: #bones by 4 by 4
        init_t = np.load(file_transform)
        with open(file_hierarchy, 'r') as f:
            skeleton_hierarchy = json.load(f)
        namelist = np.load(file_namelist)
        par_child_tree = self.build_child_parent_tree(skeleton_hierarchy)
        skel = list()
        # load all bones, store lengths to each child, we will draw the cones to each child
        for idx, bone_name in enumerate(par_child_tree):
            bone_tm = init_t[idx].T
            lens = list()
            children_idxs = list()
            parent = par_child_tree[bone_name]['parent']
            for child in par_child_tree[bone_name]['children']:
                child_idx = np.where(namelist == child)[0][0]
                children_idxs.append(child_idx)
                length = np.linalg.norm(init_t[child_idx, 3, 0:3] - init_t[idx, 3, 0:3])
                lens.append(length)
            parent_idx = -1 if parent == 'root' else np.where(namelist == parent)[0][0]
            b = Bone(lens, bone_tm, idx, parent_idx, children_idxs)
            skel.append(b)
        self.skeleton = skel

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

    def set_bones(self, bones_t, transpose=False):
        for i in range(len(self.skeleton)):
            self.skeleton[i].tm = bones_t[i].T if transpose else bones_t[i]
