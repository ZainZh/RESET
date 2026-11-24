'''The ONE source<->physical TCP frame primitive (new7 LIFT canonical-TCP contract).

Canonical = SOURCE xTrainer controller TCP axes: approach +z, closing +y (position = fingertip centre).
Physical  = LIFT TCP axes (URDF tool frame / MJCF <arm>_tcp site): approach +x, closing +y, same origin.

M_CANON_TO_PHYS holds the physical axes (x_p, y_p, z_p) expressed in canonical axes as columns:
    x_p = +z_c (approach), y_p = +y_c (closing), z_p = -x_c.
    R_physical  = R_canonical @ M          R_canonical = R_physical @ M.T
Positions are unchanged. A fixed TCP rotation at the same origin leaves the WORLD-frame spatial angular Jacobian
unchanged (w_world does not depend on the tool-frame labelling), so jacobian() needs no row relabelling.
Every conversion in common/ calls these helpers; no other matrix for this mapping may be defined.
'''
import numpy as np

M_CANON_TO_PHYS = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
M_CANON_TO_PHYS.setflags(write=False)
M_SOURCE_TO_TARGET = M_CANON_TO_PHYS  # legacy name (priors.target_priors, old tests): same object, not a second definition
assert np.allclose(M_CANON_TO_PHYS.T @ M_CANON_TO_PHYS, np.eye(3)) and abs(np.linalg.det(M_CANON_TO_PHYS) - 1) < 1e-12


def canonical_to_physical_R(R):
    return np.asarray(R, float) @ M_CANON_TO_PHYS


def physical_to_canonical_R(R):
    return np.asarray(R, float) @ M_CANON_TO_PHYS.T


def pose_canonical_to_physical(T):
    T = np.array(T, float); T[:3, :3] = canonical_to_physical_R(T[:3, :3]); return T


def pose_physical_to_canonical(T):
    T = np.array(T, float); T[:3, :3] = physical_to_canonical_R(T[:3, :3]); return T
