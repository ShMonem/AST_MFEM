import taichi as ti
import random
import scipy
import numpy as np
import scipy.sparse as sps
import polyscope as ps
from time import time
import meshio
from def_grad3D import def_grad3D
from linear_tet3dmesh_arap_ds2 import linear_tet3dmesh_arap_ds2
from linear_tet3dmesh_arap_ds import linear_tet3dmesh_arap_ds
from build_U import build_U
from closest_index import closest_index_np
from tetAssignment import tetAssignment_py
from forward_kinematics import forward_kinematics_np
from block_R3d import block_R3d
from pinvert import pinvert_np
from igl2bart import igl2bart
from compute_J import compute_J_SVD, compute_J
from V_Cycle import v_cycle_py
from GS import *
from fill_euler import fill_euler
import config


# @ti.func
def randomized_graph_coloring(A):
    if config.USE_TAICHI:
        N = A.shape[0]  # number of nodes

        # find the neighbors of each node
        # use an adjacency matrix to represent their occupation
        # taichi does not support dynamic array which is extremely inconvenient here
        # taichi does have dynamic SNode, but the operation it supports is quite limited
        neighbor_builder = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * N)
        fill_neighbors(A, neighbor_builder)
        neighbors = neighbor_builder.build()

        # the color of each node
        cNodes = ti.field(ti.i32, shape=N)
        # the candidate color when the palette runs out
        cNexts = ti.field(ti.i32, shape=N)
        # possible of nodes
        palettes = np.array([set() for i in range(N)])
        nSets = {}

        # @ti.kernel
        # def sum_row(mat: ti.template()):
        #     degrees = ti.field(ti.i32, shape=N)
        #     for i in range(mat.shape[0]):
        #         for j in range(mat.shape[1]):
        #             degrees[i] += mat[i, j]
        #
        #     return ti.max(degrees)
    else:
        N = A.shape[0]

        # build up node neighbor
        neighbors = np.array([set() for i in range(N)])
        for i in range(N):
            for j in range(N):
                if (np.abs(A[i, j]) > 0) & (i != j):
                    neighbors[i].add(j)

        # this is to compute the max degrees of each node
        degrees = 0
        for i in range(N):
            if len(neighbors[i]) > degrees:
                degrees = int(len(neighbors[i]))

        max_color = int(degrees / config.SHRINKAGE_FACTOR)
        if max_color <= 0:
            max_color = 1
        # max_color = 2

        # randomly choose color for each node from color palette
        cNext = np.zeros(N)
        palletes = [set() for i in range(N)]
        for i in range(N):
            for j in range(max_color):
                palletes[i].add(j)
            cNext[i] = max_color

        # initialize the node sets
        nSets = set()
        for i in range(N):
            nSets.add(i)

        no_progress_streak = 0

        cNode = np.zeros(N)
        while len(nSets):
            for n in nSets:
                idx = int(random.randint(0, 100) % len(palletes[n]))
                for id, item in enumerate(palletes[n]):
                    if id == idx:
                        cNode[n] = item
                        break

            temp_container = set()

            for n in nSets:
                color = cNode[n]
                satisfying = True

                for neighbor in neighbors[n]:
                    if cNode[neighbor] == color:
                        satisfying = False
                        break

                if satisfying:
                    for neighbor in neighbors[n]:
                        palletes[neighbor].discard(color)
                else:
                    temp_container.add(n)

                if len(palletes[n]) == 0:
                    palletes[n].add(cNode[n] + 1)

            if len(nSets) == len(temp_container):
                no_progress_streak += 1

                if no_progress_streak > config.NO_PROGRESS_STREAK_THRESHOLD:
                    idx = random.randint(0, 100) % len(nSets)
                    for id, item in enumerate(nSets):
                        if id == idx:
                            palletes[item].add(int(cNext[item]))
                            cNext[item] += 1
                            break
                    no_progress_streak = 0

            nSets = temp_container

        nColor = 0
        for i in range(N):
            if cNext[i] > nColor:
                nColor = cNext[i]

        partitions = []
        for i in range(int(nColor)):
            part = []
            for j in range(N):
                if cNode[j] == i:
                    part.append(j)

            partitions.append(part)

        return partitions


@ti.kernel
def fill_neighbors(A: ti.template(), adj: ti.types.sparse_matrix_builder()):
    for i, j in A:
        if (A[i, j] != 0) & (i != j):
            adj[i, j] += 1


# @ti.func
def gauss_seidel_solve(A, b, max_iter, partitions):
    x = ti.field(ti.f32, shape=A.shape[0])
    # x = np.zeros(A.shape[1])
    for i in range(max_iter):
        for part in partitions:
            # ti_part = ti.field(ti.f32, shape=len(part))
            # ti_part.from_numpy(np.array(part))
            partition_solve(A, b, x, np.array(part))

    return x


# this part can be solved in parallel
@ti.kernel
def partition_solve(A: ti.template(), b: ti.template(), x: ti.template(), partitions: ti.types.ndarray()):
    for i in partitions:
        temp = 0.0
        for j in range(A.shape[1]):
            if j != partitions[i]:
                temp += A[partitions[i], j] * x[j]
        x[partitions[i]] = (1.0 / A[partitions[i], partitions[i]]) * (b[partitions[i]] - temp)


def partition_solve_py(A, b, x, partition):
    for i in partition:
        # temp = 0
        temp = A[i, :].dot(x) - A[i, i] * x[i]
        # for j in range(A.shape[1]):
        #     if i != j:
        #         temp += A[i, j] * x[j]
        x[i] = (1.0 / A[i, i]) * (b[i] - temp)


if __name__ == '__main__':
    # ti.init(arch=ti.cpu)
    # a = np.array([[3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
    #               [0, 15, 7, 0, 0, 0, 8, 0, 0, 0],
    #               [0, 7, 28, 9, 9, 0, 0, 0, 3, 0],
    #               [0, 0, 9, 18, 0, 0, 0, 0, 9, 0],
    #               [0, 0, 9, 0, 20, 0, 0, 8, 0, 3],
    #               [0, 0, 0, 0, 0, 8, 0, 0, 0, 8],
    #               [0, 8, 0, 0, 0, 0, 14, 0, 6, 0],
    #               [3, 0, 0, 0, 8, 0, 0, 11, 0, 0],
    #               [0, 0, 3, 9, 0, 0, 6, 0, 18, 0],
    #               [0, 0, 0, 0, 3, 8, 0, 0, 0, 11]])
    # A = ti.field(ti.f32, shape=(a.shape[0], a.shape[1]))
    # A.from_numpy(a)
    # B = np.array([4, 6, 2, 7, 3, 8, 23, 2, 5, 34])
    # b = ti.field(ti.f32, shape=B.shape[0])
    # b.from_numpy(B)
    #
    # partitions = randomized_graph_coloring(a)
    # print(partitions)
    # start = time()
    # x = gauss_seidel_solve(A, b, 3, partitions)
    # # x = gauss_seidel_solve(a, B, 200, partitions)
    # end = time()
    # print("parallel: ", end - start)
    # print(x)

    # from GS import *
    #
    # U, L = A_L_sum_U_np(a)
    # start = time()
    # x = gauss_seidel_np(U, L, B, 200, np.zeros(10))
    # end = time()
    # print("triangular: ", end-start)
    # print(x)

    print("Reading data...")
    start_data = time()
    ########## BEAM DATA ##############
    # read beam model
    mesh = meshio.read("../data/beam.mesh")
    Vt = mesh.points
    Tt = mesh.cells[1].data.astype("int32")
    # make beam skeleton
    hier = np.array([0, 1, 2])
    handles = np.array([[-4.5, 0.0, 0.0], [0.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
    # level of beam
    b_levels = np.array([[30]]).astype(int)
    # load beam samples
    py_P = scipy.io.loadmat("../data/P_beam.mat")['P']
    py_PI = scipy.io.loadmat("../data/PI_beam.mat")['PI']
    # beam eulers
    eulers = np.array([[0.0, 0.0, 0.0], [-np.pi / 3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # eulers = np.array([[0.0, 0.0, 0.0], [0.0, -np.pi/2, 0.0], [0.0, 0.0, 0.0]])
    ########## BEAM DATA END ##############

    ########## HUMAN DATA ##############
    # # read human model
    # mesh = meshio.read("../data/output_mfem1.msh")
    # Vt = mesh.points
    # Tt = mesh.cells[0].data.astype("int32")
    # # load human skeleton
    # handlest = scipy.io.loadmat('../data/handles.mat')
    # hiert = scipy.io.loadmat("../data/hierarchy.mat")
    # handles = handlest['position']  ## to access the mat from the dict use: handles['position'] (numHandels, 3)
    # hier = hiert['hierarchy'][:, 1]
    # # level of human
    # b_levels = np.array([[50]]).astype(int)
    # # load human samples
    # py_P = scipy.io.loadmat("../data/P.mat")['P']  ## to access the mat from the dict use: Pt['P']
    # py_PI = scipy.io.loadmat("../data/PI.mat")['PI']
    # # load human eulers
    # eulers = fill_euler(handles)
    ########## HUMAN DATA END ##############

    end_data = time()
    print("Done reading data...")
    print("Reading data took {0} seconds.".format(end_data - start_data))
    print("Vertices: ", len(Vt), Vt.shape)
    print("Tets: ", len(Tt), Tt.shape)

    print("Getting grad...")
    B = def_grad3D(Vt, Tt)
    mu = 100  # material properties
    k_bc = 10000  # stiffness
    s = np.zeros((6 * Tt.shape[0], 1))
    print("Getting energy...")
    grad = linear_tet3dmesh_arap_ds(Vt, Tt, s, mu)
    hess = linear_tet3dmesh_arap_ds2(Vt, Tt, s, mu)

    l = b_levels.shape[0]

    print("Computing closest index...")
    weight = closest_index_np(Vt, py_P)
    print("Building the U matrix...")
    Ut, NN = build_U(weight, b_levels, l, py_P, Vt)

    midpoints_np = np.zeros((handles.shape[0] - 1, 3))
    for i in range(handles.shape[0]):
        if hier[i] == 0:
            continue
        else:
            midpoints_np[i - 1, :] = (handles[i, :] + handles[int(hier[i]) - 1, :]) / 2

    print("Computing tet assignments...")
    fAssign = tetAssignment_py(Vt, Tt, midpoints_np)

    # We move the handles here, so let's consider that we start the sim from around this point
    start_all_sim = time()

    print("Running FWD Kinematics...")
    newR, new_handles = forward_kinematics_np(handles, hier, eulers)
    rows_T = Tt.shape[0]
    R = np.zeros((rows_T, 9))
    # assigning rotations to each tet according to fAssign info
    for i in range(rows_T):
        R[i, :] = newR[int(fAssign[i]), :]

    print("Getting the R_Mat...")
    R_mat_py = block_R3d(R)  # python function

    num_handles = handles.shape[0]
    num_midpoints = midpoints_np.shape[0]
    Ht = np.zeros((num_handles + num_midpoints, 3))
    Ht[:num_handles, :] = handles
    Ht[num_handles:, :] = midpoints_np

    print("Computing the pinned verts...")
    vAssign = pinvert_np(Vt, Ht)
    I = np.eye(3)
    q = igl2bart(Vt)

    pinned_mat = sps.lil_matrix((3 * Ht.shape[0], 3 * Vt.shape[0]))
    for i in range(Ht.shape[0]):
        pinned_mat[3 * i:3 * (i + 1), 3 * int(vAssign[i]):3 * int(vAssign[i]) + 3] = I

    midpoints = np.zeros((new_handles.shape[0] - 1, 3))
    for i in range(new_handles.shape[0]):
        if hier[i] == 0:
            continue
        else:
            midpoints[i - 1, :] = (new_handles[i, :] + new_handles[hier[i] - 1, :]) / 2

    new_new_handles = np.zeros((new_handles.shape[0] + midpoints.shape[0], 3))
    new_new_handles[:new_handles.shape[0], :] = new_handles
    new_new_handles[new_handles.shape[0]:, :] = midpoints

    pinned_b = igl2bart(new_new_handles)

    nq = q.shape[0]
    ns = s.shape[0]
    nlambda = 9 * Tt.shape[0]
    start_j = time()
    if config.USE_SVD:
        J = compute_J_SVD(R_mat_py, B)  # cupy/numpy function
    else:
        J = compute_J(R_mat_py, B)
    end_j = time()
    print("J computed in {0} seconds".format(end_j - start_j))
    A = k_bc * pinned_mat.T @ pinned_mat + J.T @ hess @ J
    b = k_bc * pinned_mat.T @ pinned_b - J.T @ grad
    b = np.squeeze(b)

    partitions = randomized_graph_coloring(A)
    print(len(partitions))
    ti.init(arch=ti.cpu)
    Aa = ti.field(ti.f32, shape=(A.shape[0], A.shape[1]))
    Aa.from_numpy(A.toarray())
    bb = ti.field(ti.f32, shape=b.shape[0])
    bb.from_numpy(b)
    start = time()
    x = gauss_seidel_solve(Aa, bb, 3, partitions)
    end = time()
    print("3 iters of GS: ", end-start)

    U, L = A_L_sum_U_np(A)
    start = time()
    x = gauss_seidel_np(U, L, b, 3, np.zeros(U.shape[0]))
    end = time()
    print("triangular: ", end - start)

