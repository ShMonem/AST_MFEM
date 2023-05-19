import numpy as np
import polyscope as ps
import igl
import tetgen

ps.init()

# Read & register the mesh
v , f = igl.read_triangle_mesh("../../data/mfem_human_model_surface.obj")
#ps_mesh = ps.register_surface_mesh("my mesh", v, f)

tgen = tetgen.TetGen(v, f)
nodes, elems = tgen.tetrahedralize('-q0.1', nobisect=True, steinerleft=500000, minratio=1.1)
ps_vol = ps.register_volume_mesh("test volume mesh", nodes, tets=elems)
# Add a slice plane
ps_plane = ps.add_scene_slice_plane()
ps_plane.set_draw_plane(True) # render the semi-transparent gridded plane
ps_plane.set_draw_widget(True)

# Animate the plane sliding along the scene
for t in np.linspace(0., 2*np.pi, 12):
    pos = np.cos(t) * .8 + .6
    ps_plane.set_pose((0., 0., pos), (0., 0., -1.))

    # Take a screenshot at each frame
    #ps.screenshot(transparent_bg=False)
    ps.show()