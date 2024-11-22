import numpy as np

def poke_character(V, poke_mag, poke_at_vert, poke_raduis, poke_width, poke_type, scaled_poke = 1.0):

    poke_mag = poke_mag # magnitude of poking effect
    character_init_pos = np.array([0.0, 0.0, 0.0])
    poke_at_vert = poke_at_vert
    poke_raduis = poke_raduis  #0.4
    verts = V.copy()
    distance_to_poked_vert = np.linalg.norm(verts - verts[poke_at_vert], axis=1)
    poke_effect = np.zeros_like(distance_to_poked_vert)
    poke_width = poke_width

    if poke_type == "plane":
        poke_effect[distance_to_poked_vert <= poke_raduis] = poke_mag
    elif poke_type == "cos effect":
        scaled_poke = poke_mag * np.cos(distance_to_poked_vert * poke_width)
        poke_effect[distance_to_poked_vert <= poke_raduis] = scaled_poke[distance_to_poked_vert <= poke_raduis]
    else:
        print("poke type undefined: ", poke_type)

    return poke_effect