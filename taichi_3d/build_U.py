import numpy as np
import taichi as ti
from scipy.sparse import kron, eye, find, csr_matrix, isspmatrix
from scipy.linalg import eigh
import matplotlib.pyplot as plt
## ----------------------------------------------------------
from closest_index import *
from GS import *
## ----------------------------------------------------------
#ti.init()

def U_toTichi(Ut, A, Ub, UTAU,projA_solvers, NN):
    ## Ut: list of numpy ndarrays

    for l in range(len(Ut)):

        print(type(Ut[l]))
        #print(isspmatrix(Ut[l]))
        rows, cols = Ut[l].nonzero()
        #U_l = ti.linalg.SparseMatrix(n=Ut[l].shape[0], m=Ut[l].shape[1], dtype=ti.f32)
        #triplets_u_l = ti.Vector.ndarray(n=3, dtype=ti.f32, shape= rows.shape[0])
        #fill_tripletsSparse(triplets_u_l, rows, cols, Ut[l].data)
        #U_l.build_from_ndarray(triplets_u_l)
        U_l = ti.field(dtype=ti.f32, shape=Ut[l].shape)
        fill_U_Sparse(U_l, rows, cols, Ut[l].data)
        Ub.append(U_l)  # basis list
        #Ub_shapes.append(Ut[l].shape)
        print("U_l filled")

        if l == 0:
            UTAUt_l = csr_matrix(Ut[l].T @ A @ Ut[l] + NN)   # UTAU[i]
        else:
            UTAUt_l = csr_matrix(Ut[l].T @ UTAU[l-1] @ Ut[l]) # UTAU[i]

        UTAU_l = ti.linalg.SparseMatrix(n=UTAUt_l.shape[0], m=UTAUt_l.shape[1], dtype=ti.f32)
        triplets_utau_l = ti.Vector.ndarray(n=3, dtype=ti.f32, shape= UTAUt_l.nnz)

        rows_, cols_ = UTAUt_l.nonzero()
        fill_tripletsSparse(triplets_utau_l, rows_, cols_, UTAUt_l.data,)
        UTAU_l.build_from_ndarray(triplets_utau_l)

        UTAU.append(UTAU_l) # projected basis list
        print("UTAT_l filled")
        #UTAU_shapes.append(UTAUt_l.shape)

        solver_l = ti.linalg.SparseSolver()
        solver_l.compute(UTAU_l)
        projA_solvers.append(solver_l)

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
    j = (weight * t).T   # in matlab: j = (weight * t- (t-1))
    j = np.tile(j, (dims * t, 1)).ravel("F")

    offset = np.tile(np.arange(t), dims * n).ravel("F")

    data = tmp.ravel("F").astype("float")
    rows = i.ravel("F")
    cols = j.ravel("F") + offset.ravel("F")
    #print("cols min:", np.min(cols))
    U1 = csr_matrix((data, (rows, cols)), shape=(dims * n, t * b[0, 0]))
    U.append(U1)

    #plt.spy(U1, precision=0.5, markersize=0.1)
    #plt.show()

    # Regularize  ## TODO: ti.kernel
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



    # if we have more than 1 level in MG ## TODO: ti.kernel
    if l != 0:
        P_j = P
        for j in range(1, l):
            P_j = P_j[:b[0, j], :]

            ti_P_j = ti.Vector.field(3, dtype=ti.f32, shape=b[0, j])
            ti_P_last = ti.Vector.field(3, dtype=ti.f32, shape=b[0, j-1])
            ti_P_j.from_numpy(P_j)
            P_last = P[:b[0, j-1], :]
            ti_P_last.from_numpy(P_last)
            weight = ti.field(ti.f32, shape=b[0, j-1])
            D = ti.Vector.field(P_j.shape[0], dtype=ti.f32, shape=P_last.shape[0])

            closest_index(ti_P_last, ti_P_j, weight, D)
            U_j = csr_matrix((t * b[0, j-1], t * b[0, j]))
            w = weight.to_numpy().astype("int")
            for k in range(b[j - 1, 0]):
                tmp = csr_matrix(np.eye(t))
                U_j[t * k:t * (k+1), w[k] * t:w[k]*t+t] = tmp
            U.append(U_j)
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

if __name__ == '__main__':
    ti.init()
    np.random.seed(42)

    n_vertices = 30
    n_handles = 10
    weight = np.random.randint(0, n_handles, size=(n_vertices, 1))
    b = np.array([[n_handles, 3]])
    l = 2
    P = np.random.rand(n_handles, 3)
    V = np.random.rand(n_vertices, 3)

    U, NN = build_U(weight, b, l, P, V)
    import matplotlib.pyplot as plt
    plt.matshow(U[0].toarray())
    plt.show()