import taichi as ti
import numpy as np
import config
import random


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
    # x = ti.field(ti.f32, shape=A.shape[0])
    x = np.zeros(A.shape[1])
    for i in range(max_iter):
        for part in partitions:
            # ti_part = ti.field(ti.f32, shape=len(part))
            # ti_part.from_numpy(np.array(part))
            partition_solve_py(A, b, x, part)

    return x


@ti.kernel
def partition_solve(A: ti.template(), b: ti.template(), x: ti.template(), partitions: ti.template()):
    for i in partitions:
        temp = 0.0
        for j in range(A.shape[1]):
            if j != i:
                temp += A[i, j] * x[j]
        x[i] = (1.0 / A[i, i]) * (b[i] - temp)


def partition_solve_py(A, b, x, partition):
    for i in partition:
        temp = 0
        for j in range(A.shape[1]):
            if i!=j:
                temp += A[i, j] * x[j]
        x[i] = (1.0 / A[i, i]) * (b[i] - temp)


if __name__ == '__main__':
    ti.init(arch=ti.cpu)
    a = np.array([[3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                  [0, 15, 7, 0, 0, 0, 8, 0, 0, 0],
                  [0, 7, 28, 9, 9, 0, 0, 0, 3, 0],
                  [0, 0, 9, 18, 0, 0, 0, 0, 9, 0],
                  [0, 0, 9, 0, 20, 0, 0, 8, 0, 3],
                  [0, 0, 0, 0, 0, 8, 0, 0, 0, 8],
                  [0, 8, 0, 0, 0, 0, 14, 0, 6, 0],
                  [3, 0, 0, 0, 8, 0, 0, 11, 0, 0],
                  [0, 0, 3, 9, 0, 0, 6, 0, 18, 0],
                  [0, 0, 0, 0, 3, 8, 0, 0, 0, 11]])
    # A = ti.field(ti.f32, shape=(a.shape[0], a.shape[1]))
    # A.from_numpy(a)
    B = np.array([4, 6, 2, 7, 3, 8, 23, 2, 5, 34])
    # b = ti.field(ti.f32, shape=B.shape[0])
    # b.from_numpy(B)

    partitions = randomized_graph_coloring(a)
    print(partitions)
    x = gauss_seidel_solve(a, B, 1000, partitions)
    print(x)
    # print(neighbors)

