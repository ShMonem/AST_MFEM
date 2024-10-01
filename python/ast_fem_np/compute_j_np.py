# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0
from time import time
import numpy as np
from scipy.sparse.linalg import lsqr
from scipy.sparse import block_diag


def compute_J_SVD(r_mat, b, r_mat_blocks=None, block_size=(9, 6)):
    m, n = block_size
    num_blocks = int(r_mat.shape[0] / block_size[0])
    if r_mat_blocks is not None:
        r_blocks_np = r_mat_blocks
    else:
        r_blocks_np = get_blocks(r_mat, num_blocks, m, n)
        r_blocks_np = r_blocks_np.reshape((-1, m, n))
    output_np = np.linalg.svd(r_blocks_np)
    sigma_np = np.zeros((r_blocks_np.shape[0], n, m))
    # TODO: I think we can improve this by forming square matrices from the singular values, then adding 0 rows/cols.
    for i in range(int(r_mat.shape[0]/m)):
        for j in range(6):
            if output_np[1][i][j] > 0.0 or output_np[1][i][j] < 0.0:
                sigma_np[i][j, j] = 1.0 / output_np[1][i][j]
    out_inv_np = np.matmul(np.matmul(output_np[2].transpose((0, 2, 1)), sigma_np),
                           output_np[0].transpose((0, 2, 1)))
    block_out_inv_np = block_diag(out_inv_np)
    output = np.dot(block_out_inv_np, b)
    return output


def get_blocks(sparse_mtx, n_blocks, size_i, size_j):
    out_mtx = np.zeros([n_blocks, size_i, size_j])
    for i in range(n_blocks):
        out_mtx[i] = sparse_mtx[i*size_i:(i+1)*size_i, i*size_j:(i+1)*size_j].todense()
    return out_mtx

