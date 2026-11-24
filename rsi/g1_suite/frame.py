# Per-task rigid frame between the SOURCE xTrainer task world and the G1 model-root frame.
# The source world (table, objects, gravity, scorer axes) is never transformed. T_g1_from_source maps the task's TABLETOP centre
# (reconstruction.json table.center, already the top surface: do NOT add the 18 mm half thickness) to the declared nominal mount,
# and the task's own XML gravity to -z. Its inverse T_source_from_g1 places the G1 robot (and G1-mounted cameras) into the source world.
# The nominal mount is explicit and identical for every task and seed; no reach-optimised per-seed poses.
import json
import re
from pathlib import Path

import numpy as np

NOMINAL_MOUNT = np.array([0.40, 0.0, 0.78])   # tabletop centre in the G1 model-root frame, fixed for all tasks and seeds
G1_PELVIS_POS = np.array([0.0, 0.0, 0.79])    # original pelvis local pose in g1_scene.xml
G1_PELVIS_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def rot_between(a, b):
    a = np.asarray(a, float); a = a / np.linalg.norm(a)
    b = np.asarray(b, float); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b); s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antipodal: the cross product vanishes, so rotate by pi about any axis perpendicular to a
        axis = np.cross(a, [1., 0., 0.])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0., 1., 0.])
        axis = axis / np.linalg.norm(axis)
        return 2. * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def quat_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def gravity_from_scene(scene_xml):
    txt = Path(scene_xml).read_text()
    m = re.search(r'<option[^>]*\bgravity="([^"]+)"', txt)
    if not m:
        raise ValueError('no option gravity in ' + str(scene_xml))
    return np.array([float(x) for x in m.group(1).split()])


class TaskFrame:
    # p_g1 = R @ (p_src - top) + mount ; p_src = R.T @ (p_g1 - mount) + top. R maps source gravity to -z.
    def __init__(self, top_center, gravity, mount=NOMINAL_MOUNT):
        self.top = np.asarray(top_center, float).copy()
        self.gravity = np.asarray(gravity, float).copy()
        self.mount = np.asarray(mount, float).copy()
        self.R = rot_between(-self.gravity, [0., 0., 1.])
        assert np.allclose(self.R @ self.R.T, np.eye(3), atol=1e-12) and abs(np.linalg.det(self.R) - 1.0) < 1e-12
        g = self.R @ self.gravity
        assert np.allclose(g / np.linalg.norm(g), [0., 0., -1.], atol=1e-9)

    @classmethod
    def from_world(cls, world_dir, mount=NOMINAL_MOUNT):
        world_dir = Path(world_dir)
        rec = json.loads((world_dir / 'reconstruction.json').read_text())
        return cls(rec['table']['center'], gravity_from_scene(world_dir / 'scene.xml'), mount)

    # source -> G1
    def to_g1(self, p):
        return self.R @ (np.asarray(p, float) - self.top) + self.mount

    def dir_to_g1(self, v):
        return self.R @ np.asarray(v, float)

    def pose_to_g1(self, R, p):
        return self.R @ np.asarray(R, float), self.to_g1(p)

    # G1 -> source (T_source_from_g1)
    def from_g1(self, p):
        return self.R.T @ (np.asarray(p, float) - self.mount) + self.top

    def dir_from_g1(self, v):
        return self.R.T @ np.asarray(v, float)

    def pose_from_g1(self, R, p):
        return self.R.T @ np.asarray(R, float), self.from_g1(p)

    def pelvis_in_source(self, pelvis_pos=G1_PELVIS_POS, pelvis_quat=G1_PELVIS_QUAT):
        # Rigid placement composition: source_from_g1 o (original pelvis local pose in the G1 model-root frame).
        # Returns (pos, quat wxyz, rotation matrix) of the pelvis body in the SOURCE world.
        Rp, pp = self.pose_from_g1(quat_to_mat(pelvis_quat), pelvis_pos)
        return pp, mat_to_quat(Rp), Rp

    def describe(self):
        return {'top_center_source': self.top.tolist(), 'gravity_source': self.gravity.tolist(), 'mount_g1': self.mount.tolist(),
                'R_g1_from_source': self.R.tolist(), 'tilt_deg': float(np.degrees(np.arccos(np.clip(-self.gravity[2] / np.linalg.norm(self.gravity), -1, 1)))),
                'rotation_angle_deg': float(np.degrees(np.arccos(np.clip((np.trace(self.R) - 1) / 2, -1, 1))))}
