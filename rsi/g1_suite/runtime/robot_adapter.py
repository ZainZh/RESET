# G1 runtime robot adapter (r2). Pure code: no MuJoCo import, no task objects, no scorer, no live model/data.
# Public poses stay in the unchanged SOURCE frame. The hand module alone applies the fixed rigid transform via base_T = T_source_from_g1
# (world-from-original-G1-model-root); the original pelvis local pose (0,0,0.79) stays inside the hand module and is composed exactly once.
# opening: 1 = open, 0 = closed (source convention, passed through unchanged). Encoders: the 26 driven joints (7 arm + 6 hand per side).
# Passive distal values are model estimates computed by the hand module; optional measured values may be passed as passive=.
# r2: ONE explicit task grasp profile (kind,size) is used for FK, IK and finger targets; the source TCP convention is declared, not an undeclared identity.
import numpy as np

SIDES = ('left', 'right')
ARM = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
HAND = ('thumb_metacarpal', 'thumb_proximal', 'index_proximal', 'middle_proximal', 'ring_proximal', 'pinky_proximal')


def drive_names(side):
    return [f'{side}_{n}_joint' for n in ARM + HAND]


ENCODER_NAMES = tuple(n for s in SIDES for n in drive_names(s))
assert len(ENCODER_NAMES) == 26

# Declared TCP convention. Source canonical TCP frame (xTrainer controllers): z = approach, y = lateral (pinch/pad axis), x = y cross z.
# The G1 hand frame (HAND_CONTRACT v0.3 tcp_R) has the same axes: z = approach, y = thumb-to-index pad axis, x = lateral x approach.
# The nominal rotation from the source TCP convention to the G1 pinch frame is therefore the identity BY DECLARATION (not by omission).
# Whether the demo-prior approach/pinch directions are physically reachable by the G1 hand is a separate question answered only by IK/physics.
CONVENTION_NAME = 'source_tcp_z_approach_y_lateral__g1_pinch_z_approach_y_padaxis'
CANONICAL_CONVENTION_R = np.eye(3)

# (object_kind, size_m) per task for the hand module; None = no supported G1 grasp profile declared. Nothing is fabricated for the others.
GRASP_PROFILE = {
    'collect_apple': ('sphere', 0.068),
    'collect_cubes': ('cube', 0.032),
    'build_tower': None,
    'stack_cups': None,
    'put_mug_on_box': None,
    'open_drawer_and_place_object': None,
    'sweep_objects': None,
    'lift_box': None,
}


class AdapterError(RuntimeError):
    pass


class UnsupportedGraspProfile(AdapterError):
    pass


class IncompatibleBaseline(AdapterError):
    pass


def profile_for(task):
    prof = GRASP_PROFILE.get(task, 'unknown_task')
    if isinstance(prof, str):
        raise UnsupportedGraspProfile('unknown_task:' + str(task))
    if prof is None:
        raise UnsupportedGraspProfile('no_g1_grasp_profile:' + str(task))
    return prof


def check_so3(R, tag, atol=1e-6):
    try:
        R = np.array(R, float)
    except (TypeError, ValueError):
        raise AdapterError('bad_rotation:' + tag)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        raise AdapterError('bad_rotation_shape_or_nonfinite:' + tag)
    if not np.allclose(R @ R.T, np.eye(3), atol=atol) or abs(np.linalg.det(R) - 1.0) > atol:
        raise AdapterError('not_SO3:' + tag)
    return R


def check_rigid_T(T, tag):
    try:
        T = np.array(T, float)
    except (TypeError, ValueError):
        raise AdapterError('bad_T:' + tag)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise AdapterError('bad_T_shape_or_nonfinite:' + tag)
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9):
        raise AdapterError('bad_T_last_row:' + tag)
    check_so3(T[:3, :3], tag + '.R')
    return T


def T_source_from_g1(frame):
    # p_src = R.T (p_g1 - mount) + top  ->  rotation R.T, translation top - R.T mount. frame is a TaskFrame (read-only use).
    T = np.eye(4)
    T[:3, :3] = frame.R.T
    T[:3, 3] = frame.top - frame.R.T @ frame.mount
    return T


def check_encoders(named):
    if not isinstance(named, dict):
        raise AdapterError('encoders_not_dict')
    missing = [n for n in ENCODER_NAMES if n not in named]
    if missing:
        raise AdapterError('missing_encoder:' + missing[0])
    extra = [n for n in named if n not in ENCODER_NAMES]
    if extra:
        raise AdapterError('unknown_encoder:' + str(extra[0]))
    out = {}
    for n in ENCODER_NAMES:
        v = named[n]
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(v):
            raise AdapterError('bad_encoder_value:' + n)
        out[n] = float(v)
    return out


class RobotAdapter:
    # hand: injectable object with fk / solve_ik / finger_targets matching HAND_CONTRACT v0.3 (HandArmModel or a corrected drop-in).
    # task selects the single grasp profile used consistently by tcp (FK), plan (IK) and hand_targets; unsupported tasks raise on use.
    def __init__(self, hand, T_src_from_g1, task, convention_R=None):
        self.T = check_rigid_T(T_src_from_g1, 'T_source_from_g1')
        self.convention_R = check_so3(CANONICAL_CONVENTION_R if convention_R is None else convention_R, 'convention_R')
        self.convention = CONVENTION_NAME if convention_R is None else 'custom'
        self.hand, self.task = hand, task

    def profile(self):
        return profile_for(self.task)

    def tcp(self, named, side, passive=None):
        kind = self.profile()[0]
        e = check_encoders(named)
        r = self.hand.fk(e, side, self.T, kind=kind, passive=passive)
        if not r.valid:
            raise AdapterError('fk_invalid:' + str(r.reason))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = r.tcp_R @ self.convention_R.T, r.tcp_pos
        return T

    def plan(self, named, side, position, rotation):
        # returns the 7 named arm targets or raises; reachability here is kinematic only.
        from types import SimpleNamespace
        kind = self.profile()[0]
        e = check_encoders(named)
        R = check_so3(rotation, 'target_rotation') @ self.convention_R
        tgt = SimpleNamespace(pos=np.array(position, float), approach=R[:, 2], lateral=R[:, 1])
        r = self.hand.solve_ik(e, side, tgt, self.T, kind=kind)
        if not r.valid:
            raise AdapterError('ik_invalid:' + str(r.reason))
        return dict(r.q_arm)

    def hand_targets(self, task_unused, side, opening, squeeze=0.0):
        kind, size = self.profile()
        r = self.hand.finger_targets(side, opening, kind, size, squeeze=squeeze)
        if not r.valid:
            raise UnsupportedGraspProfile('finger_targets_invalid:' + str(r.reason))
        # hand module ctrl keys are actuator names a_{side}_*_joint; runtime targets are keyed by the drive-encoder joint names
        out = {(k[2:] if k.startswith('a_') else k): float(v) for k, v in r.ctrl.items()}
        if set(out) != {f'{side}_{h}_joint' for h in HAND}:
            raise UnsupportedGraspProfile('finger_ctrl_names_mismatch:' + str(sorted(out)))
        return out
