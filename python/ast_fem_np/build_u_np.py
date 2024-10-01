# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

import numpy as np
from scipy.sparse import kron, eye, csr_matrix
from scipy.linalg import eigh


def build_u(weight, b, levels, P, V):
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
    j = (weight * t).T   # in matlab: j = (weight * t- (t-1))
    j = np.tile(j, (dims * t, 1)).ravel("F")

    offset = np.tile(np.arange(t), dims * n).ravel("F")

    data = tmp.ravel("F").astype("float")
    rows = i.ravel("F")
    cols = j.ravel("F") + offset.ravel("F")
    U1 = csr_matrix((data, (rows, cols)), shape=(dims * n, t * b[0, 0]))
    U.append(U1)

    # Regularize
    NN = csr_matrix((t * b[0, 0], t * b[0, 0]))
    for i in range(b[0, 0]):
        # find the indecies in weight that equal i
        # that is the verts indecies have "i" as the closesd handels
        k = np.nonzero(weight == i)[0]
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
    if levels != 0:
        P_j = P
        for j in range(1, levels):
            print("IN THIS LOOP!")
            P_j = P_j[:b[0, j], :]
            raise NotImplementedError
    return U, NN
