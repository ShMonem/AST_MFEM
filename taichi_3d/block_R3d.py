import numpy as np
import scipy.sparse as sps
import taichi as ti
from taichi.linalg import SparseMatrixBuilder


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

    tmp = tmp.ravel()

    i = np.repeat(np.arange(1, 9 * n + 1), 6)
    el_offset = np.repeat(np.arange(0, 6 * n, 6), 54)
    j = np.tile(np.arange(1, 7), 9 * n)

    num_triplets = len(i)

    ti.init(arch=ti.cpu, default_fp=ti.f32, kernel_profiler=True, default_ip=ti.i32)   # Initialize Taichi

    builder = SparseMatrixBuilder(9 * n, 6 * n, max_num_triplets=num_triplets * 2)

    @ti.kernel
    def fill_taichi_matrix(tmp: ti.ext_arr(), i: ti.ext_arr(), j: ti.ext_arr(), builder: ti.sparse_matrix_builder()):
        for k in range(num_triplets):
            row = ti.cast(i[k] - 1, ti.i32)
            col = ti.cast(j[k] - 1, ti.i32)
            value = tmp[k]
            builder[row, col] += value

    fill_taichi_matrix(tmp, i, j, builder)

    blockR_taichi = builder.build()

    return blockR_taichi

"""
# Example usage
R = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [9, 8, 7, 6, 5, 4, 3, 2, 1]
])

try:
    blockR_taichi = block_R3d(R)
    print("Taichi Sparse Matrix (blockR):")
    print(blockR_taichi)
except Exception as e:
    print(f"An error occurred: {e}")
"""