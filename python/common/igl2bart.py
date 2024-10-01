# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

def igl2bart(V):

    q = V.T.reshape(-1, 1, order= 'F')
    return q