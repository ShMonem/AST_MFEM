from python.ast_fem_cp.closest_index_cp import closest_index_cp, closest_index_np


def tet_assignment_cp(V, T, J):
    """ It takes in four input fields V, T, J, and tetAssign with the ti.template() decorator.
        This kernel calculates the distance between the barycenter of each tetrahedron and each point in J
        and assigns each tetrahedron to the closest point in J."""
    barycenters = V[T].sum(1) / 4.0
    out_inds = closest_index_cp(barycenters, J)
    return out_inds


def tet_assignment_np(V, T, J):
    """ It takes in four input fields V, T, J, and tetAssign with the ti.template() decorator.
        This kernel calculates the distance between the barycenter of each tetrahedron and each point in J
        and assigns each tetrahedron to the closest point in J."""
    barycenters = V[T].sum(1)/4.0
    out_inds = closest_index_np(barycenters, J)
    return out_inds
