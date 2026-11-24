"""Robot-only self-collision check for motion planning (lift_box b5).

Inputs are joint vectors only: the 12 arm encoders and, optionally, finger positions. State comes from a static
robot-only model (robot_only.xml: the calibrated scene with object_* bodies removed, no free joints, nothing of the
task world). No live simulator data/qpos/contacts are read; the helper never touches io._w or a MotorWorld.
"""
RESET_ROOT = __import__('os').environ.get('RESET_ROOT') or next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'eval.py').is_file()).as_posix()
import numpy as np
import mujoco

MODEL = RESET_ROOT + '/rsi/xtrainer_lift_box/robot_only_model/robot_only.xml'


class RobotCollision:
    def __init__(self, path=MODEL):
        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)
        self.adr = {a: np.array([self.m.joint(f'{a}_j{j}').qposadr[0] for j in range(1, 7)]) for a in ('left', 'right')}
        self.fadr = {a: np.array([self.m.joint(f'{a}_gripper{j}').qposadr[0] for j in (1, 2)]) for a in ('left', 'right')}
        self.names = [self.m.body(int(self.m.geom_bodyid[g])).name for g in range(self.m.ngeom)]
        self.other = None

    def set_encoders(self, enc):
        """enc: (2, 6) joint encoders [left, right] at planning time (other arm held there)."""
        self.other = np.asarray(enc, float).reshape(2, 6).copy()

    def self_clear(self, arm, q, fingers=(0., 0.)):
        """True if `arm` at joint vector q (6,) has no same-arm self-contact (finger-finger excluded)."""
        m, d = self.m, self.d
        d.qpos[:] = 0
        for a, k in (('left', 0), ('right', 1)):
            d.qpos[self.adr[a]] = np.asarray(q, float) if a == arm else self.other[k]
        d.qpos[self.fadr[arm]] = np.asarray(fingers, float)
        mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
        pre = arm + '_'
        for c in d.contact[:d.ncon]:
            a, b = self.names[c.geom1], self.names[c.geom2]
            if a.startswith(pre) and b.startswith(pre) and c.dist < 0 and not ('finger' in a and 'finger' in b):
                return False
        return True
