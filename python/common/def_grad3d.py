import sys
sys.path.append("../barterlsBin")
import bartelspy as bt

def def_grad3d(V, T):
    B = bt.linear_tetmesh_B(V,T)
    return B

