import polyscope as ps
import numpy as np
import polyscope.imgui as psim
from skeleton import Skeleton
import python.ast_fem_np.vanilla_test_np as vanilla_test_np

frame = 0
skel_anim = np.load("../../data/human/human_skel_anim.npy")
sk = Skeleton()
bone_handles = []


def callback():
    global frame, sk, skel_anim, bone_handles
    changed, frame = psim.SliderInt("Anim Frame", frame, v_min=0, v_max=69)

    if changed:
        if frame > 0:
            tms = skel_anim[frame-1]
            # Seems I exported the bloody joints transposed...
            sk.set_bones(tms, transpose=True)
            for i, bone in enumerate(sk.skeleton):
                bone_handles[i].update_node_positions(bone.get_verts())
        else:
            tms = np.load("../../data/human/skel_ws_tms.npy")
            sk.set_bones(tms, transpose=True)
            for i, bone in enumerate(sk.skeleton):
                bone_handles[i].update_node_positions(bone.get_verts())

def main():
    global sk, bone_handles
    sol, tets = vanilla_test_np.main('human')

    sk.load_skeleton("../../data/human/skel_ws_tms.npy",
                     "../../data/human/skel_hier.json",
                     "../../data/human/skel_names.npy", transpose=True)
    ps.init()
    for i, bone in enumerate(sk.skeleton):
        tmp = ps.register_curve_network("joint" + str(i), bone.get_verts(), bone.get_edges(), radius=0.002)
        bone_handles.append(tmp)

    # visualization
    ps.set_user_callback(callback)
    ps.set_program_name("mfem")
    ps.set_ground_plane_mode("shadow_only")
    ps_vol = ps.register_volume_mesh("test volume mesh", sol.reshape(-1, 3), tets=tets)
    ps.show()


if __name__ == "__main__":
    main()
