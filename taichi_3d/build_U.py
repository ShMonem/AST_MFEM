import numpy as np
import taichi as ti
from scipy.sparse import kron, eye, find, csr_matrix
from scipy.linalg import eigh
import matplotlib.pyplot as plt
## ----------------------------------------------------------
from closest_index import *

## ----------------------------------------------------------
def build_U(weight, b, l, P, V):
    # 2D or 3D
    dims = V.shape[1]

    # build up homogeneous coordinate
    if dims == 2:
        homoV = np.ones((V.shape[0], 3))
        homoV[:, :2] = V
    elif dims == 3:
        homoV = np.ones((V.shape[0], 4))
        homoV[:, :3] = V

    # U is a list
    U = []
    n = V.shape[0]

    # sparse identity matrix
    I = eye(dims)

    # this is a Kronecker product
    tmp = kron(homoV, I).T.toarray()
    # t is the degree of freedom of affine transformation at each dim
    if dims == 2:
        t = 6
    elif dims == 3:
        t = 12

    # building sparse matrix of U by computing index of each entry first
    i = np.tile(np.arange(dims * n), (t, 1))
    j = (weight * t)   # in matlab: j = (weight * t- (t-1))
    j = np.tile(j, (dims * t, 1)).ravel("F")
    
    offset = np.tile(np.arange(t), (dims * n, 1)).ravel("F")

    data = tmp.ravel("F").astype("float")
    rows = i.ravel("F")
    cols = (j + offset).ravel("F")
    #print("cols min:", np.min(cols))
    U1 = csr_matrix((data, (rows, cols)), shape=(dims * n, t * b[0, 0]))
    U.append(U1)

    #plt.spy(U1, precision=0.5, markersize=5)
    #plt.show()

    # Regularize
    NN = csr_matrix((t * b[0, 0], t * b[0, 0]))
    for i in range(b[0, 0]):
        # find the indecies in weight that equal i
        # that is the verts indecies have "i" as the closesd handels
        k = np.argwhere(weight == i).ravel()
        Ub = homoV[k, :]
        Z = Ub.T @ Ub
        D, Vz = eigh(Z)
        normal = np.zeros((dims + 1, 1))
        # check if there are any 0 eigenvalues
        # Find the index of the first eigenvalue that is less than the threshold
        for j in range(dims + 1):
            if D[j] < 1e-10:  # If there's at least one such eigenvalue
                normal = Vz[:, j].reshape(-1, 1)
                reg = normal @ normal.T
                regMat = kron(reg, I)
                NN[i*t:(i+1)*t,i*t:(i+1)*t] = regMat

    

    # if we have more than 1 level in MG
    if l != 1:
        P_j = P
        for j in range(1, l):
            P_j = P_j[:b[j, 0], :]
            w = closest_index(P[:b[j - 1, 0], :], P_j)
            U_j = csr_matrix((t * b[j - 1], t * b[j]))
            for k in range(b[j - 1, 0]):
                tmp = csr_matrix(np.eye(t))
                U_j[t * k - 5:t * k, (w[k - 1, 0] - 1) * t:w[k - 1, 0]  ] = tmp  ## TODO: fix this is wrong
    return U, NN            
"""
# Numerical test
np.random.seed(42)

n_vertices = 10
n_handles = 3
weight = np.random.randint(1, n_handles + 1, size=(n_vertices, 1))
b = np.array([[n_handles]])
l = 1
P = np.random.rand(n_handles, 3)
V = np.random.rand(n_vertices, 3)

U, NN = build_U(weight, b, l, P, V)
print("U:", U)
print("NN:", NN)
"""