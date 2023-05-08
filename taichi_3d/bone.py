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
    def __init__(self, lengths, bone_tm, idx, parent_idx, children_idxs, radius=1, last_idx=0):
        self.idx = idx
        self.lens = lengths
        self.tm = bone_tm
        self.parent = parent_idx
        self.children = children_idxs
        self.radius = radius
        self.Euler = np.array([[0.0], [0.0], [0.0]])
        self.last_idx = last_idx + 1 if last_idx > 0 else 0
        self.verts, self.edges = self.init_geom()

    def init_geom(self):
        # We init the geom once, then store the base shape. The radius and the length are taken into account.
        v, e = COMPUTE_CIRCLE()
        v *= self.radius
        e = e + self.last_idx

        # add arrows only for single child parents
        if len(self.children) > 0:
            len_tip = self.lens[0]
            v_arrow, e_arrow = COMPUTE_ARROW()
            e_arrow = e_arrow + 30 + self.last_idx
            v_arrow = np.c_[v_arrow, np.ones(v_arrow.shape[0])]
            v_arrow[:, [0, 2]] = v_arrow[:, [0, 2]] * self.radius
            v_arrow[-1, 1] = len_tip
            v = np.concatenate((v[:, 0:3], v_arrow[:, 0:3]), axis=0)
            e = np.concatenate((e, e_arrow), axis=0)
        return v, e

    def get_verts(self):
        # homogenized neutral shape
        v_base = np.c_[self.verts, np.ones(self.verts.shape[0])]
        v = (self.tm @ v_base.T).T
        return v[:, 0:3]

    def get_edges(self):
        return self.edges
