# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

import sys
sys.path.append("../barterlsBin")
import bartelspy as bt

def def_grad3d(V, T):
    B = bt.linear_tetmesh_B(V,T)
    return B

