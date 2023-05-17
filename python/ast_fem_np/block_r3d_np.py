import numpy as np
from scipy.sparse import block_diag, csr_matrix

def block_r3d(R, blocks=False):
    n = R.shape[0]

    zz = np.zeros((n,))

    tmp = np.vstack((
        R[:, 0], zz, zz,
        R[:, 3],
        zz,
        R[:, 6],
        R[:, 1],
        zz,
        zz,
        R[:, 4],
        zz,
        R[:, 7],
        R[:, 2],
        zz,
        zz,
        R[:, 5],
        zz,
        R[:, 8],
        zz,
        R[:, 3],
        zz,
        R[:, 0],
        R[:, 6],
        zz,
        zz,
        R[:, 4],
        zz,
        R[:, 1],
        R[:, 7],
        zz,
        zz,
        R[:, 5],
        zz,
        R[:, 2],
        R[:, 8],
        zz,
        zz,
        zz,
        R[:, 6],
        zz,
        R[:, 3],
        R[:, 0],
        zz,
        zz,
        R[:, 7],
        zz,
        R[:, 4],
        R[:, 1],
        zz,
        zz,
        R[:, 8],
        zz,
        R[:, 5],
        R[:, 2]
    ))

   
    tmp = tmp.ravel("F").astype("float")
    tmp = tmp.reshape(R.shape[0], 9, 6)
    block_mtx = block_diag(tmp)
    blockR_numpy = csr_matrix(block_mtx)

    if blocks:
        return blockR_numpy, tmp

    return blockR_numpy
