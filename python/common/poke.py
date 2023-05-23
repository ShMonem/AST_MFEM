import numpy as np
import numpy.linalg as npla

def poke_init (vert, poke_raduis, ball_center = np.array([0.0, 0.0, 0.0]) ): # poking direction : z

    # draw a ball poke_raduis away from vert
    direction = ball_center
    ball_center = vert + ball_center  #* poke_raduis
    return ball_center, direction

def get_poked_inds(V, poke_at_vert, poke_raduis):
    distance_to_poked_vert = np.linalg.norm(V - V[poke_at_vert], axis=1).reshape(1, -1)
    indices = np.where(np.any(distance_to_poked_vert <= poke_raduis, axis=0))
    return np.squeeze(np.asarray(indices))

def update_ball_center(frame, mag, ball_center, direction):
    return ball_center + mag * frame * direction

def poke_shift(pos, ball_center, poke_raduis):
    poke_effect = np.zeros_like(pos)
    for k in range(pos.shape[0]):
        # vector to indices
        v = np.array([pos[k, 0] - ball_center[0], pos[k, 1] - ball_center[1], pos[k, 2] - ball_center[2]])
        v = v + np.array([0, 0, 1])
        if (v**2).sum() <= poke_raduis**2:
            # find the closest point on the ball surface
            poke_effect[k, :] = ball_center + poke_raduis * (v / npla.norm(v))
    return poke_effect




