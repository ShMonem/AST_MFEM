import numpy as np
from os import listdir, remove, path
from copy import deepcopy
import json
from math import sqrt
import meshio

folder = "data/beam"

filename = "beam_skel_names"
if filename in listdir(folder):
    remove(path.join(folder, filename))
beam_skel_names = []
n_jnts = 1
for i in range(n_jnts):
    beam_skel_names.append('right_bone_' + str(i + 1))
for i in range(n_jnts):
    beam_skel_names.append('left_bone_' + str(i + 1))
np.save("data/beam/beam_skel_names", beam_skel_names)

filename = "beam_skel_hier.json"
beam_skel_hier = {}
for i in range(n_jnts):
    beam_skel_hier['right_bone_' + str(i + 1)] = 'root'
    beam_skel_hier['left_bone_' + str(i + 1)] = 'right_bone_' + str(i + 1)
with open(path.join(folder, filename), "w") as hier_file:
    json.dump(beam_skel_hier, hier_file)

filename = "beam_skel_ws_tms"
if filename in listdir(folder):
    remove(path.join(folder, filename))
filename = "beam_skel_anim"
if filename in listdir(folder):
    remove(path.join(folder, filename))
beam_skel_ws_tms =[]
if n_jnts == 1:
    beam_skel_ws_tms = [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.5, 0.5, -3, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.0, 1.0]]]
else:
    n_jnt_sqrt = int(sqrt(n_jnts))
    for i in range(n_jnt_sqrt):
        for j in range(n_jnt_sqrt):
            beam_skel_ws_tms.append([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [i / (n_jnt_sqrt - 1), j / (n_jnt_sqrt - 1), -3, 1.0]])
    for i in range(n_jnt_sqrt):
        for j in range(n_jnt_sqrt):
            beam_skel_ws_tms.append([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [i / (n_jnt_sqrt - 1), j / (n_jnt_sqrt - 1), 0, 1.0]])
np.save("data/beam/beam_skel_ws_tms", beam_skel_ws_tms)

beam_skel_anim = []
n_frames = 100
beam_skel_ws_tms_pose = deepcopy(beam_skel_ws_tms)
stretch = 5
for i in range(n_frames):
    t = (n_frames / 2 - abs(n_frames / 2 - i)) / (n_frames / 2)
    for k in range(n_jnts):
        beam_skel_ws_tms_pose[k][3][2] = beam_skel_ws_tms[k][3][2] - (stretch - 1) * 3 * t / 2
    for k in range(n_jnts, 2 * n_jnts):
        beam_skel_ws_tms_pose[k][3][2] = beam_skel_ws_tms[k][3][2] + (stretch - 1) * 3 * t / 2
        # beam_skel_ws_tms_pose[0][3][2] = beam_skel_ws_tms[0][3][2] * (1 - t)
    beam_skel_anim.append(deepcopy(beam_skel_ws_tms_pose))
for i in range(n_frames):
    print(beam_skel_anim[i][-1][3][2])
np.save("data/beam/beam_skel_anim", beam_skel_anim)

mesh = meshio.read("data/beam/beam.msh")
v = mesh.points
max_z = float("-Inf")
min_z = float("Inf")
for i in range(len(v)):
    max_z = max(max_z, v[i][2])
    min_z = min(min_z, v[i][2])
pinned_inds = []
for i in range(len(v)):
    if abs(max_z - v[i][2]) < 0.001 or abs(min_z - v[i][2]) < 0.001:
        pinned_inds.append(i)
np.save("data/beam/beam_pinned_verts", pinned_inds)
content = np.load("data/beam/beam_pinned_verts.npy")
print(content)