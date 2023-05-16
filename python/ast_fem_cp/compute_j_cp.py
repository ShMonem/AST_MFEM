from time import time
import cupy as cp
from cupyx.scipy.sparse.linalg import lsqr
from cupyx.scipy.sparse import spdiags


def compute_J_SVD(r_mat, b, block_size=(9, 6), use_cupy=False):
    m, n = block_size
    num_blocks = int(r_mat.shape[0] / block_size[0])
    r_blocks_np = get_blocks(r_mat, num_blocks, m, n)
    r_blocks_np = r_blocks_np.reshape((-1, m, n))
    block_out_inv_np = lil_matrix((r_mat.shape[1], r_mat.shape[0]))
    if use_cupy:
        import cupy as cp
        start_tr = time()
        r_blocks_cu = cp.asarray(r_blocks_np)
        end_tr = time()

        start = time()
        sigma_cu = cp.zeros((r_blocks_cu.shape[0], 6, 9))
        svd_start = time()
        output_cu = cp.linalg.svd(r_blocks_cu)
        svd_end = time()
        for i in range(2998):
            for j in range(6):
                if output_cu[1][i][j] > 0.0 or output_cu[1][i][j] < 0.0:
                    sigma_cu[i][j, j] = 1.0/output_cu[1][i][j]

        out_inv_cu = cp.matmul(cp.matmul(output_cu[2].transpose((0, 2, 1)), sigma_cu),
                               output_cu[0].transpose((0, 2, 1)))
        end = time()
        print("Transfer of r_blocks to GPU took {0} seconds".format(end_tr - start_tr))
        print("SVD compute time is {0} seconds".format(svd_end - svd_start))
        out_inv_np = cp.asnumpy(out_inv_cu)
    else:
        start = time()
        svd_start = time()
        output_np = np.linalg.svd(r_blocks_np)
        svd_end = time()
        sigma_np = np.zeros((r_blocks_np.shape[0], n, m))
        start_sig = time()
        for i in range(int(r_mat.shape[0]/m)):
            for j in range(6):
                if output_np[1][i][j] > 0.0 or output_np[1][i][j] < 0.0:
                    sigma_np[i][j, j] = 1.0 / output_np[1][i][j]
        end_sig = time()
        start_mul = time()
        out_inv_np = np.matmul(np.matmul(output_np[2].transpose((0, 2, 1)), sigma_np),
                               output_np[0].transpose((0, 2, 1)))
        end_mul = time()
        end = time()
        print("SVD compute time is {0} seconds".format(svd_end - svd_start))
        print("Filling inverse sigma took {0} seconds".format(end_sig - start_sig))
        print("Multiplying for inverse took {0} seconds".format(end_mul - start_mul))
    start_set_sparse = time()
    for bl in range(r_blocks_np.shape[0]):
        block_out_inv_np[bl * n : (bl+1) * n, bl * m : (bl+1) * m] = out_inv_np[bl]
    end_set_sparse = time()
    print("Setting the sparse output took {0} seconds".format(end_set_sparse - start_set_sparse))
    start_dot = time()
    output = np.dot(block_out_inv_np, b)
    end_dot = time()
    print("Output multiply took {0} seconds".format(end_dot-start_dot))
    print("SVD and inverse compute time was {0} seconds".format(end-start))
    return output


def get_blocks(sparse_mtx, n_blocks, size_i, size_j):
    out_mtx = np.zeros([n_blocks, size_i, size_j])
    for i in range(n_blocks):
        out_mtx[i] = sparse_mtx[i*size_i:(i+1)*size_i, i*size_j:(i+1)*size_j].todense()
    return out_mtx

