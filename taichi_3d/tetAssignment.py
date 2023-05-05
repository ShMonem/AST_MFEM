import taichi as ti
import numpy as np

# initialized for CPU computation.
#ti.init(arch=ti.cpu)

@ti.func
def dist(a, b):
    """calculates the Euclidean distance between two input vectors."""
    return ti.sqrt((a - b).dot(a - b))

@ti.kernel
def barycenter(V: ti.template(), T: ti.template(), B: ti.template()):
    """ Takes in three input fields V, T, and B with the ti.template() decorator, 
        which means the actual data types and dimensions of these fields will be determined later in the code.
        This kernel calculates the barycenter (or center of mass) of each tetrahedron in T and accumulates the results in B.
    """
    for i in range(T.shape[0]):
        for j in range(4):
            B[T[i][j]] += V[i] / 4.0

@ti.kernel
def tetAssignment(V: ti.template(), T: ti.template(), J: ti.template(), tetAssign: ti.template()):
    """ It takes in four input fields V, T, J, and tetAssign with the ti.template() decorator. 
        This kernel calculates the distance between the barycenter of each tetrahedron and each point in J
        and assigns each tetrahedron to the closest point in J."""
    for i in range(T.shape[0]):
        barycenter = ti.Vector([0.0, 0.0, 0.0])
        for j in range(4):
            barycenter += V[T[i][j]] / 4.0
        max_dist = -1.0
        for j in range(J.shape[0]):
            d = dist(J[j], barycenter)
            if d > max_dist:
                max_dist = d
                tetAssign[i] = j

def main():
    """ This is an example of usage: 
    """
    # define/ load your mesh, prepare the input data in NumPy arrays
    Vnp = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Tnp = np.array([[0, 1, 2, 3]], dtype=np.int32)
    barycenters = np.zeros((Vnp.shape[0], 3), dtype=np.float32)
    tetAssign = np.zeros(Tnp.shape[0], dtype=np.int32)
    
    # create the corresponding Taichi fields,
    V = ti.Vector.field(3, dtype=ti.f32, shape=Vnp.shape[0])
    T = ti.Vector.field(4, dtype=ti.i32, shape=Tnp.shape[0])
    B = ti.Vector.field(3, dtype=ti.f32, shape=barycenters.shape[0])
    tetAssign = ti.field(dtype=ti.i32, shape=tetAssign.shape[0])

    # copy the data from NumPy to Taichi fields
    V.from_numpy(Vnp)
    T.from_numpy(Tnp)
    B.from_numpy(barycenters)
    
    # run the two kernels barycenter and tetAssignment in order using the ndarrays
    barycenter(V, T, B)
    tetAssignment(V, T, B, tetAssign)
    # finally print out the resulting tetAssign field as a NumPy array.
    print(tetAssign.to_numpy())

"""
# uncomment to see the result = [0] , because the mesh given above containes only one tet
if __name__ == '__main__':
    main()
"""

