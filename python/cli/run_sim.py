# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0

import argparse
import sys
import os
import numpy as np
sys.path.append('../../../AST_MFEM')
from python.ast_fem_np.fem_data_np import FEMData
from python.ast_fem_np.mfem_solver_np import MFEMSolver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--obj_name', required=True)
    parser.add_argument('-a', '--anim_path', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('-p', '--prefix', default='')
    args = parser.parse_args()
    skel_anim = np.load(args.anim_path)
    obj_data = FEMData(args.obj_name, load_skel=True, use_eulers=False)
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        raise NotADirectoryError
    fem_solver = MFEMSolver(obj_data)
    out_data = np.zeros((skel_anim.shape[0], obj_data.verts.shape[0], 3))
    for frame in range(skel_anim.shape[0]):
        print(f"processing frame {frame}...")
        fem_solver.obj_data.set_bones(skel_anim[frame])
        out_data[frame] = fem_solver.solve().reshape((-1, 3))
    in_file_name = os.path.basename(args.anim_path)
    out_file_name = f'sim_{args.obj_name}_' + in_file_name if args.prefix == ''\
        else '_'.join([args.prefix, in_file_name])
    output_path = os.path.join(output_dir, out_file_name)
    np.save(output_path, out_data)
    print(f"Simulation finished.\nOutput saved to {output_path}")
