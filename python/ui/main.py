import polyscope as ps
import numpy as np
import polyscope.imgui as psim
from python.ast_fem_np.fem_data_np import FEMData
from python.ast_fem_np.mfem_solver_np import MFEMSolver
from python.common.bone import COMPUTE_CIRCLE

frame = 0
skel_anim = np.load('../../data/human/human_skel_anim.npy')
# skel_anim = np.load('../../data/human/human_arm_rotate.npy')
obj_data = FEMData('human', load_skel=True, use_eulers=False)
fem_solver = MFEMSolver(obj_data)


def update_elements():
    global frame, fem_solver, skel_anim
    if frame >= 0 and frame < skel_anim.shape[0]:
        tms = skel_anim[frame]
        fem_solver.obj_data.set_bones(tms)
        if fem_solver.obj_data.skeleton.visibility:
            for i, bone in enumerate(fem_solver.obj_data.skeleton.bones):
                fem_solver.obj_data.skeleton.bone_handles[i].update_node_positions(bone.get_verts())
        if fem_solver.obj_data.visibility:
            sol = fem_solver.solve()
            fem_solver.obj_data.ps_vol.update_vertex_positions(sol.reshape(-1, 3))


def bwd_anim():
    global frame, fem_solver, skel_anim
    frame -= 1
    frame = max(0, frame)
    frame = min(skel_anim.shape[0], frame)
    update_elements()
def fwd_anim():
    global frame, fem_solver, skel_anim
    frame += 1
    frame = max(0, frame)
    frame = min(skel_anim.shape[0], frame)
    update_elements()

def callback_frame():
    global frame, fem_solver, skel_anim
    changed, frame = psim.SliderInt("Anim Frame", frame, v_min=0, v_max=skel_anim.shape[0]-1)
    if (psim.Button("<")):
        # This code is executed when the button is pressed
        bwd_anim()
    psim.SameLine()
    if (psim.Button(">")):
        # This code is executed when the button is pressed
        fwd_anim()
    if changed:
        if frame >= 0 and frame < skel_anim.shape[0]:
            update_elements()
        else:
            tms = np.load("../../data/human/human_skel_ws_tms.npy")
            fem_solver.obj_data.set_bones(tms)
            if fem_solver.obj_data.skeleton.visibility:
                for i, bone in enumerate(fem_solver.obj_data.skeleton.bones):
                    fem_solver.obj_data.skeleton.bone_handles[i].update_node_positions(bone.get_verts())
            if fem_solver.obj_data.visibility:
                sol = fem_solver.solve()
                fem_solver.obj_data.ps_vol.update_vertex_positions(sol.reshape(-1, 3))


def main():
    global fem_solver

    fem_solver.obj_data.set_bones(skel_anim[0])
    fem_solver.obj_data.skeleton.visibility = False
    fem_solver.obj_data.visibility = True

    sol = fem_solver.solve()
    ps.init()
    for i, bone in enumerate(obj_data.skeleton.bones):
        tmp = ps.register_curve_network("w_joint" + str(i), bone.get_verts(), bone.get_edges(), radius=0.002)
        obj_data.skeleton.bone_handles.append(tmp)
        tmp.set_enabled(obj_data.skeleton.visibility)

    ######### Axis markers #########
    v_base_x, e_2 = COMPUTE_CIRCLE()
    v_base_x = v_base_x + np.array([20, 0, 0])
    v_base_y, e_2 = COMPUTE_CIRCLE()
    v_base_y = v_base_y + np.array([0, 20, 0])
    v_base_z, e_2 = COMPUTE_CIRCLE()
    v_base_z = v_base_z + np.array([0, 0, 20])

    ps.register_curve_network("basex", v_base_x, e_2, radius=0.002)
    ps.register_curve_network("basey", v_base_y, e_2, radius=0.002)
    ps.register_curve_network("basez", v_base_z, e_2, radius=0.002)
    ################################

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
