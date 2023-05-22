import numpy as np
from scipy.spatial import KDTree
import meshio


def obj_to_tet_mapping(tet_mesh_path, obj_path):
    tet_mesh = meshio.read(tet_mesh_path)
    tet_verts = tet_mesh.points
    obj_mesh = meshio.read(obj_path)
    obj_verts = obj_mesh.points
    tree = KDTree(tet_verts)
    result = tree.query(obj_verts)
    obj_to_tet_map = result[1]
    return obj_to_tet_map


def tet_assign_from_skin(tet_mesh_path, obj_path, skel_names_path, infs_path, skin_weights_path):
    skel_names = np.load(skel_names_path)
    infs = np.load(infs_path)
    inf_name_remap = np.array([np.where(skel_names == name)[0][0] for name in infs])
    skin_weights = np.load(skin_weights_path)
    skin_weights_remap = np.array([[elem[0], inf_name_remap[elem[1]]] for elem in skin_weights])
    tet_mesh = meshio.read(tet_mesh_path)
    if '.msh' in tet_mesh_path:
        tets = tet_mesh.cells[0].data.astype("int32")
    elif '.mesh' in tet_mesh_path:
        tets = tet_mesh.cells[1].data.astype("int32")
    tet_verts = tet_mesh.points
    obj_mesh = meshio.read(obj_path)
    obj_verts = obj_mesh.points
    barycenters = tet_verts[tets].sum(1) / 4.0
    tree = KDTree(obj_verts)
    result = tree.query(barycenters)
    tet_weights = skin_weights_remap[result[1], 1]
    return tet_weights


if __name__ == "__main__":
    obj_path = r'C:\Users\DimitryKachkovski\git\personal\AST_MFEM\data\turtle\turtle.obj'
    tet_mesh_path = r'C:\Users\DimitryKachkovski\git\personal\AST_MFEM\data\turtle\turtle.msh'
    skel_names_path = r'C:\Users\DimitryKachkovski\git\personal\AST_MFEM\data\turtle\turtle_skel_names.npy'
    infs_path = r'C:\Users\DimitryKachkovski\git\personal\AST_MFEM\data\turtle\turtle_skin_infs.npy'
    skin_weights_path = r'C:\Users\DimitryKachkovski\git\personal\AST_MFEM\data\turtle\turtle_skin_weights.npy'

    tet_weights = tet_assign_from_skin(tet_mesh_path, obj_path, skel_names_path, infs_path, skin_weights_path)
    np.save(r'C:\Users\DimitryKachkovski\git\personal\AST_MFEM\data\turtle\turtle_tet_maya_weights.npy', tet_weights)
