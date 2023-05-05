
import numpy as np
from scipy.sparse.linalg import svds, lsqr, spsolve
from scipy.sparse import lil_matrix

import taichi as ti

#ti.init(arch=ti.cpu)   # Initialize Taichi

#@ti.kernel  ## TODO
def compute_J_as_blocks(R_mat: ti.types.ndarray(), B: ti.types.ndarray(), blockSize=(9,6)) -> ti.types.ndarray():
    bs0 = blockSize[0]
    bs1 = blockSize[1]
    n = R_mat.shape[0] // bs0
    #print("tet size confirm: ", n)
    J = lil_matrix((R_mat.shape[1], B.shape[1]))
    for k in range(n):
        for r in range(B.shape[1]):
            b = np.squeeze(np.asarray(B[bs0*k: bs0*(k+1),r].todense()))
            J[bs1*k: bs1*(k+1), r], istop, itn, r1norm = lsqr(R_mat[ bs0*k: bs0*(k+1), bs1*k: bs1*(k+1)], b, iter_lim =4)[:4]   ## lsqr cannot be called inside a ti scope

        #print(k, "termination reason: ", istop, "itr num: ", itn, "error: ", r1norm)
    return J

def compute_J(R_mat: ti.types.ndarray(), B: ti.types.ndarray(), blockSize=(9,6)) -> ti.types.ndarray():
    bs0 = blockSize[0]
    bs1 = blockSize[1]
    n = R_mat.shape[0] // bs0
    #print("tet size confirm: ", n)
    J = lil_matrix((R_mat.shape[1], B.shape[1]))
    
    for r in range(B.shape[1]):
        b = np.squeeze(np.asarray(B[:,r].todense()))
        J[:, r], istop, itn, r1norm = lsqr(R_mat, b, iter_lim =4)[:4]   ## lsqr cannot be called inside a ti scope

        #print(r, "termination reason: ", istop, "itr num: ", itn, "error: ", r1norm)
        #print(r)
    return J