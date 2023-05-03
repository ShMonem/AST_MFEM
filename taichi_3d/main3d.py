import taichi as ti
import igl
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from scipy.sparse.linalg import svds, lsqr, spsolve
from scipy.sparse import coo_matrix, csc_matrix, isspmatrix
from scipy.linalg import pinv

import meshplot
from meshplot import plot, subplot, interact
import scipy.sparse as sps
meshplot.offline()

import os
root_folder = os.getcwd()
import meshio 

# io for reading .mat files
import scipy.io as sio

#import bartels python bindings
import sys
sys.path.append("../../Bartels/python/build")
import bartelspy as bt
## ----------------------------------------------------------
from closest_index import *
from build_U import *
from tetAssignment import *
from fill_euler import *
from forward_kinematics import *
from block_R3d import *
from pinvert import *
from igl2bart import *
from def_grad3D import *
from linear_tet3dmesh_arap_ds import *
from linear_tet3dmesh_arap_ds2 import *
from compute_J import *
#from V_Cycle import *
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
V = ti.Vector.field(3, dtype=ti.f32, shape=Vt.shape[0])
F = ti.Vector.field(3, dtype=ti.i32, shape=Ft.shape[0])
T = ti.Vector.field(3, dtype=ti.i32, shape=Tt.shape[0])

V.from_numpy(Vt)
F.from_numpy(Ft)


## an array contains the number of handles for each multi-grid level
b_levels = np.array([[50]]).astype(int)
# number of multi-grid levels for the Gauss Seidel solver
l = b_levels.shape[0]

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
closest_index(V, P, weight, D) # checked


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
U, NN = build_U(weight.to_numpy(), b_levels, l, P, Vt)  # TODO: due to the issue with weights, this function is giving errors

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

# H contains handles with midpoints, the total skeleton joints that we used to attach with tetmesh
# H: [n + (n - 1), 3], where n is the number of handles, n-1 is the number of midpoints
num_handles = handles.shape[0]
num_midpoints = midpointst.shape[0]
Ht = np.zeros((num_handles + num_midpoints, 3))
Ht[:num_handles, :] = handles
Ht[num_handles:, :] = midpoints.to_numpy()


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
forward_kinematics(handles, hier, eulers, Tdummy, new_handles, newR) # TODO: we are replacing this step 
#print(newR)

# assigning rotations to each tet according to fAssign info
# this is slow, could be improved
rows_T = Tt.shape[0]
fAssign = fAssign.to_numpy()
R = np.zeros((rows_T, 9))
for i in range(rows_T):
    R[i, :] = newR[fAssign[i], :]

# R_mat: [9T, 6T]
R_mat_ti, R_mat_py = block_R3d(R)

#print(type(R_mat_py))
#plt.spy(R_mat_py, precision=0.5, markersize=5)
#plt.show()

# vAssign: [V, 1] contains the closest handles' index for each vertex
# computation is same as fAssign

# initiate dummy mats used by taichi
vAssign = ti.field(ti.i32, shape=(Vt.shape[0],))
Dt = ti.field(ti.f32, shape=(Vt.shape[0], Ht.shape[0]))

# for some very wird reason, at this point "V" was not known to the compiler 
# any more and I needed to re define it!. TODO (fix it!)
V = ti.field(ti.f32, shape=(Vt.shape[0], 3))
H = ti.field(ti.f32, shape=(Ht.shape[0], 3))

V.from_numpy(Vt)
H.from_numpy(Ht)

pinvert(V, H, vAssign, Dt)
#print(vAssign)
I = np.eye(3)

# Here we re-order vertex positions and stack them in one long vector
# q = [x0, y0, z0, x1, y1, z1,..., xn, yn, zn] (Bartel's way of doing things! :-D)
q =igl2bart(Vt)


# compute pinned_mat
# pinned_mat: [3H, 3V], where H is the total number of handles (joint + midpoint)
pinned_mat = sps.lil_matrix((3 * H.shape[0], 3 * V.shape[0]))
vAssign = vAssign.to_numpy()
for i in range(H.shape[0]):
    pinned_mat[3*i:3*(i+1), 3*vAssign[i]-3:3*vAssign[i]] = I

# Because we apply forward kinematics, so, we need to recompute pinned b
# But now we have already known the new joints position, so, we only need to compute the midpoint again
midpoints = np.zeros((new_handles.shape[0] - 1, 3))
for i in range(new_handles.shape[0]):
    if hier[i] == 0:
        continue
    else:
        midpoints[i-1, :] = (new_handles[i, :] + new_handles[hier[i], :]) / 2

new_new_handles = np.zeros((new_handles.shape[0] + midpoints.shape[0], 3))
new_new_handles[:new_handles.shape[0], :] = new_handles
new_new_handles[new_handles.shape[0]:, :] = midpoints

# Assuming you have already defined the igl2bart() function
pinned_b = igl2bart(new_new_handles)

# Compute Deformation Gradient B: [9T, 3V]
B = def_grad3D(Vt, Tt)


mu = 100  # material properties
k_bc = 1000 # stiffness 

s = np.zeros((6*Tt.shape[0], 1))

# compute gradient and hessian for As-Rigid-As-Possible Model
# grad: [3T, 3T]
# hess: [3T, 3T]
grad = linear_tet3dmesh_arap_ds(Vt,Tt, s, mu)
hess = linear_tet3dmesh_arap_ds2(Vt,Tt, s, mu)

nq = q.shape[0]
ns = s.shape[0]
nlambda = 9*Tt.shape[0]

## System matrices
J = compute_J(R_mat_py, B)
#plt.spy(J, precision=0.5, markersize=5)
#plt.show()

A = k_bc * pinned_mat.T @ pinned_mat + J.T @ hess @ J
b = k_bc * pinned_mat.T @ pinned_b - J.T @ grad
#print(isspmatrix(A)) ##True
#plt.spy(A, precision=0.5, markersize=0.1)
#plt.show()

## precompute system reduced-matrices at each MG level
# regularization is done only for the first level
UTAU = [] # creat a list
for i in range(len(U)):
    if i == 0:
        UTAU.append(U[i].T @ A @ U[i] + NN)  # UTAU[i]
    else:
        UTAU.append(U[i].T @ UTAU[i-1] @ U[i]) # UTAU[i]

## the Multi-grid solver
normVal = float('inf')
itr = 0
tol = 1e-5
sol = np.zeros(b.shape)


## TODO: check time 
while normVal > tol:
    sol_old = sol
    sol = V_Cycle(A, b, UTAU, U, 3, sol_old, 1)
    itr = itr + 1


"""
# comment for now so that no more .html generated
p = plot(Vt, Ft, k)
p.to_html()
#p.save("myp")
"""

