import taichi as ti
import igl
import scipy as sp
import numpy as np
import meshplot
from meshplot import plot, subplot, interact
meshplot.offline()

import os
root_folder = os.getcwd()
import meshio 

# io for reading .mat files
import scipy.io as sio
## ----------------------------------------------------------
from closest_index import *
from build_U import *
from tetAssignment import *
from fill_euler import *
from forward_kinematics import *
## ----------------------------------------------------------
## Load a mesh in OFF format
_, Ft = igl.read_triangle_mesh(("../data/output_mfem1.msh"))
mesh = meshio.read("../data/output_mfem1.msh")
Vt = mesh.points
Tt = mesh.cells[0].data
## Print the vertices and faces matrices 
print("Vertices: ", len(Vt), Vt.shape)
print("Faces: ", len(Ft), Ft.shape)
print("Tets: ", len(Tt), Tt.shape)

# Taichi: adjusting parameters data types for parallelizatio
ti.init(arch=ti.cpu)
V = ti.Vector.field(3, dtype=ti.i32, shape=Vt.shape[0])
F = ti.Vector.field(3, dtype=ti.i32, shape=Ft.shape[0])
T = ti.Vector.field(3, dtype=ti.i32, shape=Tt.shape[0])

V.from_numpy(Vt)
F.from_numpy(Ft)


## an array contains the number of handles for each multi-grid level
b = np.array([[50]])
# number of multi-grid levels for the Gauss Seidel solver
l = b.shape[0]

## sampling vertices  on the surface for the first level, and save it in a .mat to safe sometime
# either used the function smpling3d(T, V, b[0])
# TODO: P, PI = sampling 3d()
# or upload previously stored samples
Pt = sio.loadmat("../data/P.mat") ## to access the mat from the dict use: Pt['P']
PIt = sio.loadmat("../data/PI.mat") ## to access the entries from the dict use: PIt['PI']

# initialize the required taichi fields to pass, and store in 'weights', 
# also a dummy vectorfield used by the function "closest_index" for intermediate commputations
weight = ti.field(ti.i32, shape=Vt.shape[0])
D = ti.Vector.field(Pt['P'].shape[0], dtype=ti.f32, shape=Vt.shape[0])
P = ti.Vector.field(3, dtype=ti.f32, shape=Pt['P'].shape[0])
P.from_numpy(Pt['P'])

# compute the weights per vertex and store them in "weight"
closest_index(V, P, weight, D)
#print(weight) # TODO: not giving same results as matlab so far



"""
Build the reduced subspace basis mat U, where U is the prolongation matrix introduce in the 
"Galerkin Multi-grid Method" paper by Xian et. al. (same notation used)
# U here is not a matrix, but rather a list that stores matrices
# U stores the corresponding prolongation matrix at each MG level
# in this example, we only have 1 level, so, U only have one matrix
# for level= i, the corresponding skinning basism atrix in U is "Ui"
# Ui has size [V, T], where T is (12 * number of handles) -> the total number of degrees of all affine transformation
# NN is a matrix which is used to do the regularization as in the paper.
"""
#U, NN = build_U(weight.to_numpy(), b, l, P, Vt)  # TODO: due to the issue with weights, this function is giving errors

# or upload previously stored samples
handlest = sio.loadmat('../data/handles.mat') 
hiert = sio.loadmat("../data/hierarchy.mat") 
handles = handlest['position'] ## to access the mat from the dict use: handles['position'] (numHandels, 3)
hier = hiert['hierarchy'][:, 1] ## to access the entries from the dict use: hiert['hierarchy']  (numHandles, 2)

"""
As we are working on a complex model, we need to pin more than joints vertices (?)
We also assign each tet's rotation with its closest midpoint rather than the joints
"""
# compute midpoint of each bone
midpointst = np.zeros((handles.shape[0]-1, 3))
for i in range(handles.shape[0]):
    if hier[i] == 0:
        continue
    else:
        midpointst[i - 1, :] = (handles[i, :] + handles[int(hier[i])-1, :]) / 2
"""
Compute tet blongings
fAssign is computed in a same way as weight above, size (T, 1),
and contains the closes index of midpoint for each tet
"""
# initialize taichi template to be filled by tetAssignment
midpoints = ti.Vector.field(3, dtype=ti.f32, shape=midpointst.shape[0])
midpoints.from_numpy(midpointst)
fAssign = ti.field(dtype=ti.i32, shape=Tt.shape[0])
tetAssignment(V, T, midpoints, fAssign)   # we use midpoint for rotation
# print(fAssign) ## TODO: double check results with matlab

eulers = fill_euler(handles)
#print(eulers) ## TODO: double check results with matlab

"""
Apply forward kinematics to compute each rotation and translation after deformation
newR: [n, 9], where n is the number of joints, is the new rotation matrix after kinematics
each entry is a vectorized rotation matrix
new_handles: [n, 3] is the new joints position after kinematics
"""
# we need to initialize the output of forward kinematics, so that the ti.kernel updates
# them instead of returning them
n_newR = handles.shape[0] - 1
n_newR = handles.shape[0] - 1
Tdummy = np.zeros((n_newR + 1, 9))
newR = np.zeros((n_newR, 9))
Tdummy[:, [0, 4, 8]] = 1
newR[:, [0, 4, 8]] = 1
new_handles = np.zeros_like(handles)

#forward_kinematics(handles, hier, eulers, Tdummy, new_handles, newR)
#print(newR)
## plot mesh
#k = igl.gaussian_curvature(Vt, Ft)

"""
# comment for now so that no more .html generated
p = plot(Vt, Ft, k)
p.to_html()
#p.save("myp")
"""

