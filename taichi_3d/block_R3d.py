import numpy as np
import scipy.sparse as sps
import taichi as ti
from taichi.linalg import SparseMatrixBuilder
from scipy.sparse import lil_matrix, csr_matrix

def block_R3d(R):
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
    i = np.tile(np.arange(0, 9 * n), (6,1)).ravel("F").astype(int)
    el_offset = np.tile(np.arange(0, 6 * n, 6), (54, 1)).ravel("F").astype(int)
    j = np.tile(np.arange(0, 6).T, 9 * n).ravel("F").astype(int)
    #print(i.shape, j.shape, el_offset.shape)
    num_triplets = len(i)

    ## numpy rotation matrix
    # in python it is more efficient to fill-in a lil_matix
    blockR_numpy = lil_matrix((9 * n, 6 * n))
    for k in range(num_triplets):
        blockR_numpy[i[k], el_offset[k] + j[k]] = tmp[k]
    # then we transform to csr/coo/csc for linalg convenience!
    blockR_numpy = csr_matrix(blockR_numpy)

    ti.init(arch=ti.cpu)   # Initialize Taichi
    ## taichi rotation matrix
    """
    blockR = ti.MatrixNdarray( 9 * n, 6 * n, ti.f64, shape=())
    blockR.from_numpy(blockR_numpy.todense()[:, :])

    dime = n
    blockR_taichi = ti.linalg.SparseMatrix(n= 9*dime, m=6*dime, dtype= ti.f64)
    blockR_taichi.build_from_ndarray(blockR)
    """

    builder = SparseMatrixBuilder(9 * n, 6 * n, max_num_triplets=num_triplets, dtype=ti.f32)
    @ti.kernel
    def fill_taichi_matrix(tmp: ti.types.ndarray(), i: ti.types.ndarray(), j: ti.types.ndarray(), el_offset: ti.types.ndarray(), builder: ti.sparse_matrix_builder(), num_triplets:ti.i32):
        for k in range(num_triplets):
            row = ti.cast(i[k], ti.i32)
            col =  ti.cast(el_offset[k], ti.i32) + ti.cast(j[k], ti.i32) 
            value = ti.cast(tmp[k], ti.f32)
            builder[row, col] += value
            

    fill_taichi_matrix(tmp, i, j, el_offset, builder, num_triplets)

    blockR_taichi = builder.build(dtype=ti.f32, _format = 'CSR')  # wrong values!!!
    return blockR_taichi , blockR_numpy

"""
# Example usage
R = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [9, 8, 7, 6, 5, 4, 3, 2, 1]
])

try:
    blockR_taichi, blockR_numpy = block_R3d(R)
    print("Taichi Sparse Matrix (blockR):")
    print(blockR_taichi)
    print(blockR_numpy)
except Exception as e:
    print(f"An error occurred: {e}")
"""
