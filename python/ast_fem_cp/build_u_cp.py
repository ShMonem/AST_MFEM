import cupy as cp
from cupyx.scipy.sparse import kron, eye, csr_matrix
from cupy.linalg import eigh


def build_u_cp(weight, b, levels, P, V):
    dims = V.shape[1]

    # build up homogeneous coordinate
    if dims == 2:
        homoV = cp.ones((V.shape[0], 3))
        homoV[:, :2] = V
    elif dims == 3:
        homoV = cp.ones((V.shape[0], 4))
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
    i = cp.tile(cp.arange(dims * n), (t, 1))
    j = (weight * t).T   # in matlab: j = (weight * t- (t-1))
    j = cp.tile(j, (dims * t, 1)).ravel("F")
    offset = cp.tile(cp.arange(t), dims * n).ravel("F")
    data = tmp.ravel("F").astype("float")
    rows = i.ravel("F")
    cols = j.ravel("F") + offset.ravel("F")
    U1 = csr_matrix((data, (rows, cols)), shape=(dims * n, t * b[0, 0]))
    U.append(U1)

    # Regularize  ## TODO: ti.kernel
    NN = csr_matrix((t * b[0, 0], t * b[0, 0]))
    for i in range(int(b[0, 0])):
        # find the indecies in weight that equal i
        # that is the verts indecies have "i" as the closesd handels
        k = cp.nonzero(weight == i)[0]
        Ub = homoV[k, :]
        Z = Ub.T @ Ub
        D, Vz = eigh(Z)
        # check if there are any 0 eigenvalues
        # Find the index of the first eigenvalue that is less than the threshold
        for j in range(dims + 1):
            if D[j] < 1e-10:  # If there's at least one such eigenvalue
                normal = Vz[:, j].reshape(-1, 1)
                reg = normal @ normal.T
                regMat = kron(reg, I)
                NN[i*t:(i+1)*t,i*t:(i+1)*t] = regMat

    # if we have more than 1 level in MG ## TODO: ti.kernel
    if levels != 0:
        P_j = P
        for j in range(1, levels):
            pass  # Multilevel implementation here

    return U, NN
