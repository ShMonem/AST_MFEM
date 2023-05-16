
def igl2bart(V):

    q = V.T.reshape(-1, 1, order= 'F')
    return q