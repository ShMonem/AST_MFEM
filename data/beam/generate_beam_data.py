# import numpy as np
# from os import listdir, remove, path
# from copy import deepcopy
# import json
# import meshio
#
# # Folder containing beam data
# folder = "../../data/beam/"
#
# # Set the number of joints to 8 (for both left and right sides)
# n_jnts = 6
# middle_bones = 4  # Keeping 4 middle bones for smooth deformation
#
# # Load the beam mesh and extract vertices
# mesh = meshio.read(folder + "beam.msh")
# v = mesh.points
#
# # Find the leftmost and rightmost z-values in the beam (corners will be at min/max z-values)
# max_z = float("-Inf")
# min_z = float("Inf")
# for i in range(len(v)):
#     max_z = max(max_z, v[i][2])
#     min_z = min(min_z, v[i][2])
#
# # Find vertices on the leftmost (min_z) and rightmost (max_z) sides of the beam
# left_side_verts = [i for i in range(len(v)) if abs(v[i][2] - min_z) < 0.001]
# right_side_verts = [i for i in range(len(v)) if abs(v[i][2] - max_z) < 0.001]
#
# # Identify corner vertices (top-left, bottom-left, top-right, bottom-right)
# top_left_vert = min(left_side_verts, key=lambda idx: v[idx][1])
# bottom_left_vert = max(left_side_verts, key=lambda idx: v[idx][1])
# top_right_vert = min(right_side_verts, key=lambda idx: v[idx][1])
# bottom_right_vert = max(right_side_verts, key=lambda idx: v[idx][1])
#
# # Bone names definition
# filename = "beam_skel_names"
# if filename in listdir(folder):
#     remove(path.join(folder, filename))
# beam_skel_names = []
#
# # Add right and left bone names for 8 joints
# for i in range(n_jnts):
#     beam_skel_names.append('right_bone_' + str(i + 1))
# for i in range(n_jnts):
#     beam_skel_names.append('left_bone_' + str(i + 1))
#
# # Add middle bone names
# for i in range(middle_bones):
#     beam_skel_names.append('middle_bone_' + str(i + 1))
#
# # Add corner bones
# beam_skel_names.append('top_left_bone')
# beam_skel_names.append('bottom_left_bone')
# beam_skel_names.append('top_right_bone')
# beam_skel_names.append('bottom_right_bone')
#
# # Save bone names
# np.save(folder + "/beam_skel_names", beam_skel_names)
#
# # Define bone hierarchy in JSON
# filename = "beam_skel_hier.json"
# beam_skel_hier = {}
#
# # Set up hierarchy for 8 right and 8 left bones
# for i in range(n_jnts):
#     beam_skel_hier['right_bone_' + str(i + 1)] = 'root'
#     beam_skel_hier['left_bone_' + str(i + 1)] = 'right_bone_' + str(i + 1)
#
# # Add middle bones hierarchy, connecting them to the right and left bones
# for i in range(middle_bones):
#     if i == 0:
#         beam_skel_hier['middle_bone_' + str(i + 1)] = 'right_bone_1'
#     else:
#         beam_skel_hier['middle_bone_' + str(i + 1)] = 'middle_bone_' + str(i)
#
# # Add corner bones hierarchy
# beam_skel_hier['top_left_bone'] = 'left_bone_1'
# beam_skel_hier['bottom_left_bone'] = 'left_bone_1'
# beam_skel_hier['top_right_bone'] = 'right_bone_1'
# beam_skel_hier['bottom_right_bone'] = 'right_bone_1'
#
# # Save the hierarchy to a file
# with open(path.join(folder, filename), "w") as hier_file:
#     json.dump(beam_skel_hier, hier_file)
#
# # Create workspace transformation matrices for the skeleton
# filename = "beam_skel_ws_tms"
# if filename in listdir(folder):
#     remove(path.join(folder, filename))
# filename = "beam_skel_anim"
# if filename in listdir(folder):
#     remove(path.join(folder, filename))
#
# beam_skel_ws_tms = []
#
# # Create transformation matrices for right, left, and middle bones
# # Define transformations for right and left bones across 8 joints
# for i in range(n_jnts):
#     beam_skel_ws_tms.append(
#         [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [i / (n_jnts - 1), 0.5, -3, 1.0]])
#
# for i in range(n_jnts):
#     beam_skel_ws_tms.append(
#         [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [i / (n_jnts - 1), 0.5, 0.0, 1.0]])
#
# # Add transformation matrices for middle bones
# for i in range(middle_bones):
#     beam_skel_ws_tms.append(
#         [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [i / (middle_bones - 1), 0.5, -1.5, 1.0]])
#
# # Add transformation matrices for corner bones
# beam_skel_ws_tms.append([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
#                          [v[top_left_vert][0], v[top_left_vert][1], v[top_left_vert][2], 1.0]])
# beam_skel_ws_tms.append([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
#                          [v[bottom_left_vert][0], v[bottom_left_vert][1], v[bottom_left_vert][2], 1.0]])
# beam_skel_ws_tms.append([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
#                          [v[top_right_vert][0], v[top_right_vert][1], v[top_right_vert][2], 1.0]])
# beam_skel_ws_tms.append([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
#                          [v[bottom_right_vert][0], v[bottom_right_vert][1], v[bottom_right_vert][2], 1.0]])
#
# # Save the workspace transformations
# np.save(folder + "beam_skel_ws_tms", beam_skel_ws_tms)
#
# # Animation setup
# beam_skel_anim = []
# n_frames = 100
# beam_skel_ws_tms_pose = deepcopy(beam_skel_ws_tms)
# stretch = 5
#
# # Animate the transformation matrices for all bones (right, left, and middle)
# for i in range(n_frames):
#     t = (n_frames / 2 - abs(n_frames / 2 - i)) / (n_frames / 2)
#
#     # Animate right bones
#     for k in range(n_jnts):
#         beam_skel_ws_tms_pose[k][3][2] = beam_skel_ws_tms[k][3][2] - (stretch - 1) * 3 * t / 2
#
#     # Animate left bones
#     for k in range(n_jnts, 2 * n_jnts):
#         beam_skel_ws_tms_pose[k][3][2] = beam_skel_ws_tms[k][3][2] + (stretch - 1) * 3 * t / 2
#
#     # Animate middle bones smoothly
#     for k in range(2 * n_jnts, 2 * n_jnts + middle_bones):
#         beam_skel_ws_tms_pose[k][3][2] = (beam_skel_ws_tms[k][3][2] + t * stretch * 0.2)
#
#     # Animate corner bones (optional, modify according to need)
#     # beam_skel_ws_tms_pose for corners can be manipulated similarly
#
#     beam_skel_anim.append(deepcopy(beam_skel_ws_tms_pose))
#
# # Save the animation data
# np.save(folder + "beam_skel_anim", beam_skel_anim)
#
# # Load mesh data and find pinned vertices based on min/max Z
# mesh = meshio.read(folder + "beam.msh")
# v = mesh.points
# max_z = float("-Inf")
# min_z = float("Inf")
# for i in range(len(v)):
#     max_z = max(max_z, v[i][2])
#     min_z = min(min_z, v[i][2])
#
# # Find the vertices that are pinned
# pinned_inds = []
# for i in range(len(v)):
#     if abs(max_z - v[i][2]) < 0.001 or abs(min_z - v[i][2]) < 0.001:
#         pinned_inds.append(i)
#
# # Save pinned vertices
# np.save(folder + "/beam_pinned_verts", pinned_inds)
# content = np.load(folder + "beam_pinned_verts.npy")
# print(content)


import numpy as np
from os import listdir, remove, path
from copy import deepcopy
import json
from math import sqrt
import meshio

folder = "../../data/beam"

filename = "beam_skel_names"
if filename in listdir(folder):
    remove(path.join(folder, filename))
beam_skel_names = []
n_jnts = 1
for i in range(n_jnts):
    beam_skel_names.append('right_bone_' + str(i + 1))
for i in range(n_jnts):
    beam_skel_names.append('left_bone_' + str(i + 1))
np.save("../../data/beam/beam_skel_names", beam_skel_names)

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
np.save("../../data/beam/beam_skel_ws_tms", beam_skel_ws_tms)

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
np.save("../../data/beam/beam_skel_anim", beam_skel_anim)

mesh = meshio.read("../../data/beam/beam.msh")
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
np.save("../../data/beam/beam_pinned_verts", pinned_inds)
content = np.load("../../data/beam/beam_pinned_verts.npy")
print(content)