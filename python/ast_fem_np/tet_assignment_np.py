from python.ast_fem_np.closest_index_np import closest_index


def tet_assignment(V, T, J):
    """ It takes in four input fields V, T, J, and tetAssign with the ti.template() decorator.
        This kernel calculates the distance between the barycenter of each tetrahedron and each point in J
        and assigns each tetrahedron to the closest point in J."""
    barycenters = V[T].sum(1)/4.0
    out_inds = closest_index(barycenters, J)
    return out_inds
