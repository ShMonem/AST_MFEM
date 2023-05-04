
import numpy as np
from scipy.sparse.linalg import svds, lsqr, spsolve
from scipy.sparse import lil_matrix

import taichi as ti

ti.init(arch=ti.cpu)   # Initialize Taichi

#@ti.kernel  ## TODO
def compute_J(R_mat: ti.types.ndarray(), B: ti.types.ndarray()) -> ti.types.ndarray():
    J = lil_matrix((R_mat.shape[1], B.shape[1]))
    for k in range(B.shape[1]):
        b = x = np.squeeze(np.asarray(B[:,k].todense()))
        J[:, k], istop, itn, r1norm = lsqr(R_mat, b, iter_lim =4)[:4]   ## lsqr cannot be called inside a ti scope

        #print( "sol: ", J[:, k], "termination reason: ", istop, "itr num: ", itn, "error: ", r1norm)
    return J
