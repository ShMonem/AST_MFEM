import polyscope as ps
import numpy as np
import polyscope.imgui as psim
from skeleton import Skeleton

frame = 0
skel_anim = np.load("../data/skel_anim.npy")
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
            tms = np.load("../data/skel_ws_tms.npy")
            sk.set_bones(tms, transpose=True)
            for i, bone in enumerate(sk.skeleton):
                bone_handles[i].update_node_positions(bone.get_verts())

def main():
    global sk, bone_handles
    ps.set_program_name("mfem")
    ps.set_ground_plane_mode("none")
    ps.init()

    sk.load_skeleton("../data/skel_ws_tms.npy", "../data/skel_hier.json", "../data/skel_names.npy", transpose=True)

    for i, bone in enumerate(sk.skeleton):
        tmp = ps.register_curve_network("joint" + str(i), bone.get_verts(), bone.get_edges(), radius=0.002)
        bone_handles.append(tmp)

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
