import numpy as np
import taichi as ti
from scipy.sparse import kron, eye, find, csr_matrix
from scipy.linalg import eigh

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
    tmp = kron(homoV, I).T
    # t is the degree of freedom of affine transformation at each dim
    if dims == 2:
        t = 6
    elif dims == 3:
        t = 12

    # building sparse matrix of U by computing index of each entry first
    print("Unique weights:", np.unique(weight))
    i = np.tile(np.arange(1, dims * n + 1), (t, 1))
    j = (weight * t - (t - 1))
    j = np.tile(j, (dims * t, 1))
    print("j min:", np.min(j))
    print("j max:", np.max(j))
    offset = np.tile(np.arange(0, t), (dims * n, 1)).reshape(-1, 1)
    data = tmp.toarray().flatten('F')
    rows = (np.ravel(i) - 1)
    cols = (np.ravel(j) - 1) + np.ravel(offset)
    #print("cols min:", np.min(cols))
    U1 = csr_matrix((data, (rows, cols)), shape=(dims * n, t * b[0, 0]))

    

    # Regularize
    NN = csr_matrix((t * b[0, 0], t * b[0, 0]))
    for i in range(1, b[0, 0] + 1):
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
                NN[t * i - t:t * i, t * i - t:t * i] = regMat


    U.append(U1)

    # if we have more than 1 level in MG
    if l != 1:
        P_j = P
        for j in range(2, l + 1):
            P_j = P_j[:b[j - 1, 0], :]
            w = closest_index(P[:b[j - 2, 0], :], P_j)
            U_j = csr_matrix((t * b[j - 2], t * b[j - 1]))
            for k in range(1, b[j - 2] + 1):
                tmp = csr_matrix(np.eye(t))
                U_j[t * k - 5:t * k, (w[k - 1, 0] - 1) * t:w[k - 1, 0]  ]
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