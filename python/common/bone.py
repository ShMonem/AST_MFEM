# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

import numpy as np


def COMPUTE_CIRCLE():
    v1 = np.array([[np.cos(np.pi / 5.0 * i), 0, np.sin(np.pi / 5.0 * i)] for i in range(10)])
    v2 = np.array([[np.cos(np.pi / 5.0 * i), np.sin(np.pi / 5.0 * i), 0] for i in range(10)])
    v3 = np.array([[0, np.cos(np.pi / 5.0 * i), np.sin(np.pi / 5.0 * i)] for i in range(10)])
    e1 = np.array([[i, (i + 1) % 10] for i in range(10)])
    e2 = np.array([[i + 10, (i + 1) % 10 + 10] for i in range(10)])
    e3 = np.array([[i + 20, (i + 1) % 10 + 20] for i in range(10)])

    v = np.concatenate([v1, v2, v3])
    e = np.concatenate([e1, e2, e3])
    return v, e


def COMPUTE_ARROW():
    v = np.array([[np.cos(np.pi / 2.0 * i), 0, np.sin(np.pi / 2.0 * i)] for i in range(4)])
    v = np.concatenate([v, np.array([[0,1.0,0]])])
    e1 = np.array([[i, (i + 1) % 4] for i in range(4)])
    e2 = np.array([[0, 4], [1, 4], [2, 4], [3, 4]])
    e = np.concatenate((e1, e2))
    return v, e


class Bone:
    def __init__(self, lengths, bone_tm, all_tms, idx, parent_idx, children_idxs, radius=1):
        self.idx = idx
        self.lens = lengths
        self.tm = bone_tm
        self.all_tms = all_tms
        self.parent = parent_idx
        self.children = children_idxs
        self.radius = radius
        self.Euler = np.array([[0.0], [0.0], [0.0]])
        self.verts, self.edges = self.init_geom()

    def init_geom(self):
        # We init the geom once, then store the base shape. The radius and the length are taken into account.
        v, e = COMPUTE_CIRCLE()
        v *= self.radius

        # add arrows only for single child parents
        if len(self.children) > 0:
            for child_idx in self.children:
                v_arrow, e_arrow = COMPUTE_ARROW()
                e_arrow = e_arrow + v.shape[0]
                v_arrow[:, [0, 2]] = v_arrow[:, [0, 2]] * self.radius
                v = np.concatenate((v, v_arrow), axis=0)
                e = np.concatenate((e, e_arrow), axis=0)
        return v, e

    def get_verts(self):
        # homogenized neutral shape
        v_base = np.c_[self.verts, np.ones(self.verts.shape[0])]
        v = (self.tm @ v_base.T).T
        if len(self.children) > 0:
            for i, child_idx in enumerate(self.children):
                tip_vert = self.all_tms[child_idx].T[0:3, -1]
                # Figure out the id of each tip vertex
                vert_id = 29 + i * 5 + 5
                v[vert_id][0:3] = tip_vert
        return v[:, 0:3]

    def get_edges(self):
        return self.edges
