import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, isspmatrix, lil_matrix
from scipy.linalg import pinv

def left_inverse_block_diagonal(B, block_size):

    n = B.shape[1] // block_size[1]
    left_inv_blocks = []
    
    for i in range(n):
        B_block = B[i * block_size[0]:(i + 1) * block_size[0], i * block_size[1]:(i + 1) * block_size[1]]
        left_inv_block = np.linalg.pinv(B_block.todense())
        left_inv_blocks.append(left_inv_block)
        
    #left_inverse_B = np.block([[left_inv_blocks[i] if j == i else np.zeros(block_size[::-1]) for j in range(n)] for i in range(n)])

    #assert np.allclose(np.dot(left_inverse_B, B), np.eye(6*n), atol=1e-8), "The computed left inverse is not correct"

    return left_inv_blocks

def block_matrix_multiply(left_inv_blocks, Mat, block_size):
    n = Mat.shape[0] // block_size[1]
    r = Mat.shape[1]
    J = lil_matrix((n * block_size[1], r))
    print(len(left_inv_blocks))
    for i in range(n):
        left_inv_block = left_inv_blocks[i]
        Mat_block = Mat[i * block_size[0]:(i + 1) * block_size[0], :].todense()
        J_block = np.dot(left_inv_block,Mat_block)
        #print(J_block)
        J[i * block_size[1]:(i + 1) * block_size[1], :] = J_block
        
    return J

def blockMatSolve(B, Mat, blockSiyze=(9, 6)):
    left_inverse_B = left_inverse_block_diagonal(B, blockSiyze)
    J = block_matrix_multiply(left_inverse_B, Mat, blockSiyze)

    assert np.allclose(np.dot(left_inverse_B, Mat), J, atol=1e-8), "The computed J matrix is not correct"
    return J


"""
# Create B matrix with the specified structure
def create_sample_B(n, block_size=(9, 6)):
    B = np.zeros((n * block_size[0], n * block_size[1]))
    for i in range(n):
        B_block = np.random.rand(block_size[0], block_size[1])
        B[i * block_size[0]:(i + 1) * block_size[0], i * block_size[1]:(i + 1) * block_size[1]] = B_block
    return B


def create_sample_Mat(n, r, block_size=(9, 6)):
    Mat = np.random.rand(n * block_size[0], r)
    return Mat


n = 2000  # You can change the value of n as needed
B = create_sample_B(n)

#print("The left inverse of B is:")
#print(B, left_inverse_B)
r = 5
Mat = create_sample_Mat(n, r)

blockSiyze=(9, 6)

J = blockMatSolve(B, Mat, blockSiyze)

print("The J matrix is:")
print(J)
"""
