# This file is part of the AST_MFEM project (https://github.com/ShMonem/AST_MFEM).
# Copyright AST_MFEM to all developers and contributors. All rights reserved.
# License: Apache-2.0
import numpy as np


def compute_tetrahedron_volume(v1, v2, v3, v4):
    """
    Computes the volume of a tetrahedron given its four vertices (v1, v2, v3, v4).
    Uses the determinant formula for the volume of a tetrahedron.
    """
    v1, v2, v3, v4 = np.array(v1), np.array(v2), np.array(v3), np.array(v4)
    return np.abs(np.dot(np.cross(v2 - v1, v3 - v1), v4 - v1)) / 6.0


def compute_mesh_volume(vertices, faces):
    """
    Computes the volume of a closed triangular mesh by decomposing it into tetrahedra.
    vertices: A list or array of mesh vertices (shape: Nx3).
    faces: A list or array of triangular faces, where each face contains 3 vertex indices (shape: Mx3).
    Returns the volume of the mesh.
    """
    total_volume = 0.0
    origin = np.array([0.0, 0.0, 0.0])  # Use the origin as the reference point for tetrahedron decomposition

    for face in faces:
        # Get the vertices of the face
        v1 = vertices[face[0]]
        v2 = vertices[face[1]]
        v3 = vertices[face[2]]

        # Compute the volume of the tetrahedron formed by the origin and the face
        tetra_volume = compute_tetrahedron_volume(origin, v1, v2, v3)
        total_volume += tetra_volume

    return total_volume



