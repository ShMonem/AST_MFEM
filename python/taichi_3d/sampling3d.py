import numpy as np
import networkx as nx

def sampling3d(T, V, n):
    startp = np.concatenate((T[:, 0], T[:, 0], T[:, 0], T[:, 1], T[:, 1], T[:, 2]))
    endp = np.concatenate((T[:, 1], T[:, 2], T[:, 3], T[:, 2], T[:, 3], T[:, 3]))

    # Combine startp and endp into a 2D array
    edge = np.column_stack((startp, endp))

    # Get unique rows and their indices
    C, ia = np.unique(edge, return_index=True, axis=0)

    # Use the unique indices to update startp and endp
    startp = C[:, 0].astype(int)
    endp = C[:, 1].astype(int)

    # Compute the Euclidean distance between the start and end points
    weight = np.linalg.norm(V[startp] - V[endp], axis=1)

    # Create a weighted graph
    G = nx.Graph()
    for i in range(len(startp)):
        G.add_edge(startp[i], endp[i], weight=weight[i])

    # Create a sequence from 0 to the number of rows in V
    S = np.arange(V.shape[0])

    # Create a zero array with the same number of rows as V
    dist = np.zeros(V.shape[0])

    # Create an empty list
    D = []

    # Create a zero array with n rows
    PI = np.zeros(n, dtype=int)

    for i in range(n):
        if i == 0:
            start = 0
        else:
            start = PI[i - 1]
        for j in range(V.shape[0]):
            path = nx.shortest_path(G, source=start, target=S[j], weight='weight')
            d = sum(G[path[k - 1]][path[k]]['weight'] for k in range(1, len(path)))
            dist[j] = d
        D.append(dist.copy())
        D_min = np.min(np.array(D), axis=0)
        if i != 0:
            D_min[PI[:i]] = 0
        d = np.max(D_min)
        idx = np.argmax(D_min)
        PI[i] = idx

    P = V[PI]

    return P, PI


"""
T = np.array([[0, 1, 2, 3],
              [1, 2, 3, 0],
              [2, 3, 0, 1]])

V = np.array([[0, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

n = 3

P, PI = sampling3d(T, V, n)
print("P: ", P)
print("PI: ", PI)
"""