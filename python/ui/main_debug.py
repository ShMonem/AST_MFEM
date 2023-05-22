import polyscope as ps
import numpy as np
import polyscope.imgui as psim
from python.common.skeleton import Skeleton
from python.ast_fem_np.fem_data_np import FEMData
from python.ast_fem_np.mfem_solver_np import MFEMSolver
from python.common.bone import COMPUTE_CIRCLE

frame = 0
skel_anim = np.load('../../data/human/static_anim.npy')
new_tms = np.load('../../data/human/new_tms.npy')
obj_data = FEMData('human', load_skel=True, use_eulers=False)
fem_solver = MFEMSolver(obj_data)
bone_handles = []
ps_vol = None
names = np.load('../../data/human/human_skel_names.npy')
skel_2 = Skeleton()


def get_vs(tms_id):
    skel_anim_pos = skel_anim[19][tms_id][-1, :3]
    new_tms_pos = new_tms[tms_id][-1, :3]

    skel_tm = np.eye(4)
    skel_tm[:3, :3] = skel_anim[19][tms_id][:3, :3]
    skel_tm[-1, :3] = skel_anim_pos

    new_tm = np.eye(4)
    new_tm[:3, :3] = new_tms[tms_id][:3, :3]
    new_tm = new_tms[tms_id]
    v_x, e = obj_data.skeleton.bones[0].init_geom()
    v_y, _ = obj_data.skeleton.bones[0].init_geom()
    v_z, _ = obj_data.skeleton.bones[0].init_geom()
    v_x[-1] = v_x[-1] + np.array([10, 0, 0])
    v_y[-1] = v_y[-1] + np.array([0, 10, 0])
    v_z[-1] = v_z[-1] + np.array([0, 0, 10])

    v_x_base = np.c_[v_x, np.ones(v_x.shape[0])]
    v_y_base = np.c_[v_y, np.ones(v_y.shape[0])]
    v_z_base = np.c_[v_z, np.ones(v_z.shape[0])]
    v_x = (skel_tm.T @ v_x_base.T).T
    v_y = (skel_tm.T @ v_y_base.T).T
    v_z = (skel_tm.T @ v_z_base.T).T

    v2_x, _ = obj_data.skeleton.bones[0].init_geom()
    v2_y, _ = obj_data.skeleton.bones[0].init_geom()
    v2_z, _ = obj_data.skeleton.bones[0].init_geom()
    v2_x[-1] = v2_x[-1] + np.array([10, 0, 0])
    v2_y[-1] = v2_y[-1] + np.array([0, 10, 0])
    v2_z[-1] = v2_z[-1] + np.array([0, 0, 10])

    v2_x_base = np.c_[v2_x, np.ones(v2_x.shape[0])]
    v2_y_base = np.c_[v2_y, np.ones(v2_y.shape[0])]
    v2_z_base = np.c_[v2_z, np.ones(v2_z.shape[0])]
    v2_x = (new_tm.T @ v2_x_base.T).T
    v2_y = (new_tm.T @ v2_y_base.T).T
    v2_z = (new_tm.T @ v2_z_base.T).T
    return [v_x, v_y, v_z], [v2_x, v2_y, v2_z], e

def callback_frame():
    global frame, fem_solver, skel_anim
    changed, frame = psim.SliderInt("Anim Frame", frame, v_min=0, v_max=69)
    if changed:
        if frame:
            tms = skel_anim[frame-1]
            fem_solver.obj_data.set_bones(tms)
            if fem_solver.obj_data.skeleton.visibility:
                for i, bone in enumerate(fem_solver.obj_data.skeleton.bones):
                    fem_solver.obj_data.skeleton.bone_handles[i].update_node_positions(bone.get_verts())
            if fem_solver.obj_data.visibility:
                sol = fem_solver.solve()
                fem_solver.obj_data.ps_vol.update_vertex_positions(sol.reshape(-1, 3))
        else:
            tms = np.load("../../data/human/human_skel_ws_tms.npy")
            fem_solver.obj_data.set_bones(tms)
            if fem_solver.obj_data.skeleton.visibility:
                for i, bone in enumerate(fem_solver.obj_data.skeleton.bones):
                    fem_solver.obj_data.skeleton.bone_handles[i].update_node_positions(bone.get_verts())
            if fem_solver.obj_data.visibility:
                sol = fem_solver.solve()
                fem_solver.obj_data.ps_vol.update_vertex_positions(sol.reshape(-1, 3))


def callback_joint_sel():
    global frame, obj_data, skel_anim, bone_handles, ps_vol, names
    changed, frame = psim.SliderInt("Anim Frame", frame, v_min=0, v_max=64)

    if changed:
        if frame:
            # print(f"Current joint displayed - {names[frame]}")
            # [v_x, v_y, v_z], [v2_x, v2_y, v2_z], e = get_vs(frame)
            # bone_handles[0].update_node_positions(v_x[:, :3])
            # bone_handles[1].update_node_positions(v_y[:, :3])
            # bone_handles[2].update_node_positions(v_z[:, :3])
            # bone_handles[3].update_node_positions(v2_x[:, :3])
            # bone_handles[4].update_node_positions(v2_y[:, :3])
            # bone_handles[5].update_node_positions(v2_z[:, :3])
            pass


def main():
    global fem_solver, bone_handles, ps_vol, new_tms

    fem_solver.obj_data.set_bones(skel_anim[0])
    fem_solver.obj_data.skeleton.visibility = False
    fem_solver.obj_data.visibility = True

    sol = fem_solver.solve()
    ps.init()
    for i, bone in enumerate(obj_data.skeleton.bones):
        tmp = ps.register_curve_network("w_joint" + str(i), bone.get_verts(), bone.get_edges(), radius=0.002)
        obj_data.skeleton.bone_handles.append(tmp)
        tmp.set_enabled(obj_data.skeleton.visibility)
    ################## ROTATION VISUALIZING ##################
    # tms_id = 0
    # [v_x, v_y, v_z], [v2_x, v2_y, v2_z], e = get_vs(tms_id)
    #
    # bone_handles.append(ps.register_curve_network("skel_anim_x", v_x[:, :3], e, radius=0.002))
    # bone_handles.append(ps.register_curve_network("skel_anim_y", v_y[:, :3], e, radius=0.002))
    # bone_handles.append(ps.register_curve_network("skel_anim_z", v_z[:, :3], e, radius=0.002))
    # bone_handles.append(ps.register_curve_network("new_tms_x", v2_x[:, :3], e, radius=0.002))
    # bone_handles.append(ps.register_curve_network("new_tms_y", v2_y[:, :3], e, radius=0.002))
    # bone_handles.append(ps.register_curve_network("new_tms_z", v2_z[:, :3], e, radius=0.002))
    ##########################################################

    v_base_x, e_2 = COMPUTE_CIRCLE()
    v_base_x = v_base_x + np.array([20, 0, 0])
    v_base_y, e_2 = COMPUTE_CIRCLE()
    v_base_y = v_base_y + np.array([0, 20, 0])
    v_base_z, e_2 = COMPUTE_CIRCLE()
    v_base_z = v_base_z + np.array([0, 0, 20])

    ps.register_curve_network("basex", v_base_x, e_2, radius=0.002)
    ps.register_curve_network("basey", v_base_y, e_2, radius=0.002)
    ps.register_curve_network("basez", v_base_z, e_2, radius=0.002)

    # visualization
    ps.set_user_callback(callback_frame)
    ps.set_program_name("mfem")
    ps.set_ground_plane_mode("none")
    fem_solver.obj_data.ps_vol = ps.register_volume_mesh("test volume mesh2", sol.reshape(-1, 3),
                                                         tets=fem_solver.obj_data.tets)
    fem_solver.obj_data.ps_vol.set_enabled(fem_solver.obj_data.visibility)
    ps.show()


if __name__ == "__main__":
    main()
