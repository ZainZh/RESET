# Robot-only G1 hand/arm primitive, HAND_CONTRACT v0.3. Isolated model: the original G1 robot only (every non-pelvis world body and every world geom removed), no task world, no object state, no contact feedback.
# Conventions: opening 1 = open, 0 = closed (source rsi.env.IO convention; converted exactly once in opening_to_closure). base_T = world-from-original-G1-model-root; the original pelvis local pose (0,0,0.79) stays inside the model and is composed exactly once.
# Hand input: the 6 driven hand encoders per side (required for measured FK). The 10 passive distals are model estimates from the original equalities (distal = gain x proximal) unless measured values are passed.
# TCP: calibrated pinch frame (TCP_CALIBRATION.json, signed mj_geomDistance on the real collision geoms), valid only near the calibrated finger shape of that object kind; fk reports hand_matches_calibration. Geometry only, not a grasp claim.
RESET_ROOT = __import__('os').environ.get('RESET_ROOT') or next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'eval.py').is_file()).as_posix()
import json, os
from dataclasses import dataclass, field
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = RESET_ROOT + '/robots/g1/g1_scene.xml'
ARM = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
HAND_DRIVES = ('thumb_metacarpal', 'thumb_proximal', 'index_proximal', 'middle_proximal', 'ring_proximal', 'pinky_proximal')
DISTAL_GAIN = {'thumb': 1.0, 'index': 1.155, 'middle': 1.155, 'ring': 1.155, 'pinky': 1.155}  # original equality polycoef
ROBOT_ROOT_BODY = 'pelvis'  # the only world-level body kept; every other world-level body, whatever its name, is removed
ROBOT_COUNTS = {'njnt': 36, 'nu': 26, 'neq': 10}  # original G1 robot after world removal: 36 joints, 26 actuators, 10 mimic couplings
PELVIS_LOCAL_Z = 0.79
CAL = json.load(open(os.path.join(HERE, 'TCP_CALIBRATION.json')))
KIND_KEY = {'cube': 'cube_32mm', 'sphere': 'apple_68mm'}
SIZE_RANGE = {'cube': (0.025, 0.045), 'sphere': (0.055, 0.085)}
THUMB_FRAC = {'thumb_metacarpal': 0.7, 'thumb_proximal': 0.2}  # verified opposition posture (fractions of the upper joint limit), held for every opening
F_MAX = 0.5  # finger closure fraction (of the coupling-consistent proximal cap) at opening 0
PASSIVE_CONSISTENCY_TOL = 0.05  # measured passive vs estimate
CAL_MATCH_TOL = 0.1  # rad, per drive, for hand_matches_calibration


class HandError(ValueError):
    pass


def _num(v, tag):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise HandError('bad_value:' + str(tag))
    v = float(v)
    if not np.isfinite(v):
        raise HandError('nonfinite:' + str(tag))
    return v


def opening_to_closure(opening):
    # the ONE place where the source opening convention (1 open, 0 closed) becomes a closure fraction (0 open, 1 closed)
    o = _num(opening, 'opening')
    if not 0.0 <= o <= 1.0:
        raise HandError('bad_opening')
    return 1.0 - o


def check_T(T):
    if T is None:
        return np.eye(4)
    try:
        T = np.array(T, float)
    except (TypeError, ValueError):
        raise HandError('bad_base_T')
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise HandError('bad_base_T')
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
        raise HandError('base_T_not_orthonormal')
    if np.linalg.det(R) < 0:
        raise HandError('base_T_reflection')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9):
        raise HandError('base_T_bad_last_row')
    return T


@dataclass
class TCPTarget:
    pos: np.ndarray
    approach: np.ndarray
    lateral: np.ndarray


@dataclass
class FKResult:
    valid: bool
    reason: str = ''
    tcp_pos: np.ndarray = None
    tcp_R: np.ndarray = None
    wrist_pos: np.ndarray = None
    wrist_R: np.ndarray = None
    q_arm: np.ndarray = None
    q_hand: dict = field(default_factory=dict)
    passive: dict = field(default_factory=dict)
    passive_source: dict = field(default_factory=dict)
    passive_clamped: list = field(default_factory=list)
    hand_source: str = ''
    hand_matches_calibration: bool = False
    tcp_frame_label: str = 'VIRTUAL_PLANNING_GRASP_FRAME'  # fixed calibrated frame in base_link, a planning frame; NOT an actual contact point; measured hand encoders do not move it, only the arm does
    tcp_is_contact_point: bool = False


@dataclass
class SegmentFK:
    # encoder-dependent real collision-geom frames of one hand (see HandArmModel.segment_fk). Kinematic geometry only.
    valid: bool
    reason: str = ''
    segments: dict = field(default_factory=dict)  # body name -> digit, geom_id, geom_type, frame, body_pos/R, geom_pos/R (world), surface_point
    q_arm: np.ndarray = None
    q_hand: dict = field(default_factory=dict)
    passive: dict = field(default_factory=dict)
    passive_source: dict = field(default_factory=dict)
    hand_source: str = ''
    contact_inference: str = 'none: kinematic geometry only, no object, contact, force or surface point is inferred'


@dataclass
class IKResult:
    valid: bool
    reason: str = ''
    q_arm: dict = field(default_factory=dict)
    pos_residual: float = float('inf')
    angle_residual: float = float('inf')
    approach_dot: float = -1.0
    iters: int = 0
    joint_margin_min: float = float('nan')


@dataclass
class FingerTargets:
    valid: bool
    reason: str = ''
    ctrl: dict = field(default_factory=dict)
    coupled_distal: dict = field(default_factory=dict)  # model estimates, not actuators
    closure_f: float = float('nan')
    source: str = ''
    calibrated_pad_gap_mm: float = float('nan')  # geometry at the calibrated closure only
    pad_contact_expected: bool = False  # never claimed from geometry


def so3_log_angle(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def rot_vec_error(R_cur, R_tar):
    # rotation vector of R_tar @ R_cur.T (world frame), true SO(3) error
    E = R_tar @ R_cur.T
    ang = so3_log_angle(E)
    if ang < 1e-9:
        return np.zeros(3), 0.0
    if np.pi - ang < 1e-6:
        w, v = np.linalg.eigh((E + E.T) / 2.0)
        return v[:, np.argmax(w)] * ang, ang
    ax = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) / (2.0 * np.sin(ang))
    return ax * ang, ang


def frame_from_axes(approach, lateral):
    try:
        a = np.array(approach, float)
        l = np.array(lateral, float)
    except (TypeError, ValueError):
        raise HandError('bad_axes')
    if a.shape != (3,) or l.shape != (3,) or not (np.all(np.isfinite(a)) and np.all(np.isfinite(l))):
        raise HandError('bad_axes')
    na, nl = np.linalg.norm(a), np.linalg.norm(l)
    if na < 1e-9 or nl < 1e-9:
        raise HandError('zero_axis')
    a = a / na
    l = l / nl
    if abs(a @ l) > 1e-3:
        raise HandError('approach_lateral_not_orthogonal')
    return np.column_stack([np.cross(l, a), l, a])  # columns: x = lateral x approach, y = lateral, z = approach (right handed)


def tcp_in_base(side, kind):
    if side not in ('left', 'right') or kind not in KIND_KEY:
        raise ValueError('bad_side_or_kind')
    e = CAL['sides'][side][KIND_KEY[kind]]
    R = np.column_stack([e['x_in_base'], e['lateral_in_base'], e['approach_in_base']])
    U, _, Vt = np.linalg.svd(R)  # JSON axes are rounded to 4 decimals: project to the nearest rotation
    return np.array(e['tcp_in_base'], float), U @ Vt


def _cal_f(side, kind):
    return float(CAL['sides'][side][KIND_KEY[kind]]['finger_closure_f'])


def _load_spec(xml_path):
    # MuJoCo 3.2.x: MjSpec().from_file(path) is an instance method and bodies leave with detach_body. MuJoCo 3.3.x: MjSpec.from_file(path) is a class factory and bodies/geoms leave with spec.delete. Any other API shape fails explicitly.
    cls = getattr(mujoco, 'MjSpec', None)
    if cls is None:
        raise HandError('unsupported_mujoco_version:' + str(mujoco.__version__))
    if hasattr(cls, 'detach_body') and hasattr(cls, 'find_body') and not hasattr(cls, 'delete'):
        spec = cls()
        spec.from_file(xml_path)
        return spec, 'instance'
    if hasattr(cls, 'delete') and not hasattr(cls, 'detach_body'):
        return cls.from_file(xml_path), 'factory'
    raise HandError('unsupported_mujoco_version:' + str(mujoco.__version__))


def _strip_to_robot(spec, style):
    # keep only the original pelvis subtree: remove every other world-level body and every world-level geom (floor included), whatever it is named
    w = spec.worldbody
    for _ in range(100000):
        victim, b = None, w.first_body()
        while b is not None:
            if b.name != ROBOT_ROOT_BODY:
                victim = b
                break
            b = w.next_body(b)
        if victim is None:
            break
        if style == 'instance':
            spec.detach_body(victim)
        else:
            spec.delete(victim)
    else:
        raise HandError('world_body_strip_not_terminating')
    for _ in range(100000):
        g = w.first_geom()
        if g is None:
            break
        if style == 'instance':
            g.delete()
        else:
            spec.delete(g)
    else:
        raise HandError('world_geom_strip_not_terminating')


class HandArmModel:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        if model.neq != 10:
            raise HandError('mimic_equalities_missing')
        self.rng, self.qadr, self.dadr = {}, {}, {}
        names = [f'{s}_{n}_joint' for s in ('left', 'right') for n in ARM + HAND_DRIVES]
        names += [f'{s}_{f}_distal_joint' for s in ('left', 'right') for f in DISTAL_GAIN]
        for n in names:
            j = model.joint(n)
            self.rng[n] = (float(model.jnt_range[j.id][0]), float(model.jnt_range[j.id][1]))
            self.qadr[n] = int(j.qposadr[0])
            self.dadr[n] = int(j.dofadr[0])
        self.drives = {f'{s}_{n}_joint' for s in ('left', 'right') for n in ARM + HAND_DRIVES}
        self.passive_names = {f'{s}_{f}_distal_joint' for s in ('left', 'right') for f in DISTAL_GAIN}
        self.base = {s: model.body(f'{s}_base_link').id for s in ('left', 'right')}
        mujoco.mj_forward(self.m, self.d)

    @classmethod
    def from_robot_xml(cls, xml_path=DEFAULT_XML):
        spec, style = _load_spec(xml_path)
        _strip_to_robot(spec, style)
        m = spec.compile()  # original robot assets, masks, gravcomp, limits, couplings and the pelvis local pose (0,0,0.79) are untouched
        top = [m.body(i).name for i in range(1, m.nbody) if m.body_parentid[i] == 0]
        if top != [ROBOT_ROOT_BODY]:
            raise HandError('world_body_not_removed:' + ','.join(top))
        if bool((m.geom_bodyid == 0).any()):
            raise HandError('world_geom_not_removed')
        for k, v in ROBOT_COUNTS.items():
            if int(getattr(m, k)) != v:
                raise HandError('robot_counts_changed:' + k)
        if abs(float(m.body(ROBOT_ROOT_BODY).pos[2]) - PELVIS_LOCAL_Z) > 1e-9:
            raise HandError('pelvis_pose_changed')
        return cls(m)

    def arm_names(self, side):
        return [f'{side}_{n}_joint' for n in ARM]

    def _validate(self, named, side):
        if side not in ('left', 'right'):
            raise HandError('bad_side')
        if isinstance(named, (mujoco.MjModel, mujoco.MjData)) or not isinstance(named, dict):
            raise HandError('encoders_not_dict')
        vals = {}
        for n, v in named.items():
            if n not in self.drives:
                raise HandError('unknown_joint:' + str(n))
            v = _num(v, n)
            lo, hi = self.rng[n]
            if v < lo - 1e-9 or v > hi + 1e-9:
                raise HandError('encoder_out_of_limit:' + n)
            vals[n] = min(max(v, lo), hi)
        for n in self.arm_names(side):
            if n not in vals:
                raise HandError('missing_joint:' + n)
        return vals

    def _hand_from_f(self, side, f):
        # the six drive values for finger closure fraction f; non-thumb proximal cap = min(range_hi, distal_hi / 1.155) so the exact coupling stays legal
        q = {}
        for h, fr in THUMB_FRAC.items():
            q[f'{side}_{h}_joint'] = fr * self.rng[f'{side}_{h}_joint'][1]
        for fn in ('index', 'middle', 'ring', 'pinky'):
            pn, dn = f'{side}_{fn}_proximal_joint', f'{side}_{fn}_distal_joint'
            cap = min(self.rng[pn][1], self.rng[dn][1] / DISTAL_GAIN[fn])
            q[pn] = float(min(max(f, 0.0), 1.0) * cap)
        return q

    def _apply(self, vals, side, opening, passive):
        vals = dict(vals)
        if opening is not None:
            vals.update(self._hand_from_f(side, opening_to_closure(opening) * F_MAX))
            src = 'prospective_opening'
        else:
            for h in HAND_DRIVES:
                n = f'{side}_{h}_joint'
                if n not in vals:
                    raise HandError('missing_hand_encoder:' + n)
            src = 'measured'
        est, clamped = {}, []  # clamped stays only for the FKResult API: modeled passives are never clipped, so it is always empty
        for s in ('left', 'right'):
            for f, g in DISTAL_GAIN.items():
                pn = f'{s}_{f}_proximal_joint'
                if pn not in vals:
                    continue
                dn = f'{s}_{f}_distal_joint'
                lo, hi = self.rng[dn]
                e = g * vals[pn]  # exact original coupling, gain x proximal, no clipping
                if e > hi:
                    raise HandError('passive_estimate_over_limit:' + dn)
                if e < lo:
                    raise HandError('passive_estimate_under_limit:' + dn)
                est[dn] = e
        psrc = {k: 'model_estimate' for k in est}
        if passive is not None:
            if not isinstance(passive, dict):
                raise HandError('passive_not_dict')
            for dn, v in passive.items():
                if dn not in self.passive_names:
                    raise HandError('unknown_passive:' + str(dn))
                v = _num(v, dn)
                lo, hi = self.rng[dn]
                if v < lo - 1e-9 or v > hi + 1e-9:
                    raise HandError('passive_out_of_limit:' + dn)
                if dn in est and abs(v - est[dn]) > PASSIVE_CONSISTENCY_TOL:
                    raise HandError('passive_inconsistent:' + dn)
                est[dn] = min(max(v, lo), hi)
                psrc[dn] = 'measured'
        self.d.qpos[:] = 0
        for n, v in {**vals, **est}.items():
            self.d.qpos[self.qadr[n]] = v
        return vals, est, psrc, clamped, src

    def fk(self, named_encoders, side, base_T=None, *, opening=None, kind='cube', passive=None):
        # measured FK (opening=None) needs the 7 arm + 6 hand drive encoders of that side. opening is a prospective model-only hand shape (1 open, 0 closed) and replaces the hand encoders.
        try:
            T = check_T(base_T)
            if kind not in KIND_KEY:
                raise HandError('bad_kind')
            vals = self._validate(named_encoders, side)
            vals, est, psrc, clamped, src = self._apply(vals, side, opening, passive)
            mujoco.mj_kinematics(self.m, self.d)
            b = self.base[side]
            Rb = self.d.xmat[b].reshape(3, 3).copy()
            pb = self.d.xpos[b].copy()
            p_loc, R_loc = tcp_in_base(side, kind)
            R, t = T[:3, :3], T[:3, 3]
            cal = self._hand_from_f(side, _cal_f(side, kind))
            match = all(abs(vals[n] - v) <= CAL_MATCH_TOL for n, v in cal.items())
            return FKResult(True, '', R @ (pb + Rb @ p_loc) + t, R @ Rb @ R_loc, R @ pb + t, R @ Rb,
                            np.array([vals[n] for n in self.arm_names(side)]), {f'{side}_{h}_joint': vals[f'{side}_{h}_joint'] for h in HAND_DRIVES},
                            est, psrc, clamped, src, bool(match))
        except HandError as e:
            return FKResult(False, str(e))

    def segment_fk(self, named_encoders, side, base_T=None, *, passive=None):
        # SEPARATE encoder-dependent FK of the real hand collision geoms. Same inputs as measured fk (7 arm + 6 hand drive encoders of that side, original coupling estimates, optional measured passive override, base_T applied once). The fixed virtual TCP of fk() does not move with the hand encoders; these segment poses do. Returns collision-geom frames only: no object, no contact, no force and no surface point is inferred.
        try:
            T = check_T(base_T)
            vals = self._validate(named_encoders, side)
            vals, est, psrc, clamped, src = self._apply(vals, side, None, passive)
            mujoco.mj_kinematics(self.m, self.d)
            R, t = T[:3, :3], T[:3, 3]
            root = self.base[side]
            seg = {}
            for b in range(self.m.nbody):
                p = b
                while p > 0 and p != root:
                    p = int(self.m.body_parentid[p])
                if p != root:
                    continue
                gs = [g for g in range(self.m.ngeom) if int(self.m.geom_bodyid[g]) == b and int(self.m.geom_contype[g]) != 0]
                bn = self.m.body(b).name
                if len(gs) != 1:
                    raise HandError('unexpected_collision_geom_count:' + bn)
                g = gs[0]
                low = bn.lower()
                digit = next((k for k in ('thumb', 'index', 'middle', 'ring', 'pinky') if k in low), 'palm')
                seg[bn] = {'digit': digit, 'geom_id': g, 'geom_type': mujoco.mjtGeom(int(self.m.geom_type[g])).name, 'frame': 'collision_geom_frame',
                           'body_pos': R @ self.d.xpos[b] + t, 'body_R': R @ self.d.xmat[b].reshape(3, 3),
                           'geom_pos': R @ self.d.geom_xpos[g] + t, 'geom_R': R @ self.d.geom_xmat[g].reshape(3, 3),
                           'surface_point': 'unknown_without_object'}
            return SegmentFK(True, '', seg, np.array([vals[n] for n in self.arm_names(side)]), {f'{side}_{h}_joint': vals[f'{side}_{h}_joint'] for h in HAND_DRIVES}, est, psrc, src)
        except HandError as e:
            return SegmentFK(False, str(e))

    def solve_ik(self, named_encoders, side, target_tcp, base_T=None, *, kind='cube', tol_pos=0.005, tol_ang=0.10, max_iters=200):
        # plans the 7 arm joints for the calibrated pinch frame of that object kind (assumes the hand is commanded to finger_targets for that kind). Only the 7 arm encoders of the side are needed as seed.
        try:
            T = check_T(base_T)
            if kind not in KIND_KEY:
                raise HandError('bad_kind')
            vals = self._validate(named_encoders, side)
            p_loc, R_loc = tcp_in_base(side, kind)
            try:
                pos = np.array(target_tcp.pos, float)
                ap, la = target_tcp.approach, target_tcp.lateral
            except (AttributeError, TypeError, ValueError):
                raise HandError('bad_target')
            if pos.shape != (3,) or not np.all(np.isfinite(pos)):
                raise HandError('bad_target_pos')
            Rt = frame_from_axes(ap, la)
            tol_pos, tol_ang = _num(tol_pos, 'tol_pos'), _num(tol_ang, 'tol_ang')
            if tol_pos <= 0 or tol_ang <= 0:
                raise HandError('bad_tolerance')
            if isinstance(max_iters, (bool, np.bool_)) or not isinstance(max_iters, (int, np.integer)) or not 1 <= max_iters <= 1000:
                raise HandError('bad_max_iters')
        except HandError as e:
            return IKResult(False, str(e))
        names = self.arm_names(side)
        qadr = np.array([self.qadr[n] for n in names])
        dadr = np.array([self.dadr[n] for n in names])
        lo = np.array([self.rng[n][0] for n in names])
        hi = np.array([self.rng[n][1] for n in names])
        b = self.base[side]
        Rm = T[:3, :3].T
        pm = Rm @ (pos - T[:3, 3])  # target in the model-root frame
        Rtm = Rm @ Rt
        sg = 1.0 if side == 'left' else -1.0
        seeds = [np.array([vals[n] for n in names]), np.clip(np.array([-.5, .2 * sg, 0, 1., 0, 0, 0]), lo, hi), np.clip(np.array([-.9, .1 * sg, 0, .6, 0, 0, 0]), lo, hi)]
        self.d.qpos[:] = 0
        best = None
        total = 0
        for sd in seeds:
            q = np.clip(sd, lo + 1e-4, hi - 1e-4)
            for it in range(max_iters + 1):
                total += 1
                self.d.qpos[qadr] = q
                mujoco.mj_kinematics(self.m, self.d)
                mujoco.mj_comPos(self.m, self.d)
                Rb = self.d.xmat[b].reshape(3, 3)
                p = self.d.xpos[b] + Rb @ p_loc
                R = Rb @ R_loc
                ep = pm - p
                er, ang = rot_vec_error(R, Rtm)
                if (np.linalg.norm(ep) < 0.3 * tol_pos and ang < 0.3 * tol_ang) or it == max_iters:
                    break
                jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
                mujoco.mj_jac(self.m, self.d, jp, jr, p, b)
                J = np.vstack([jp[:, dadr], 0.5 * jr[:, dadr]])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), np.r_[ep, 0.5 * er])
                dq *= min(1.0, 0.2 / max(np.abs(dq).max(), 1e-12))
                q = np.clip(q + dq, lo + 1e-4, hi - 1e-4)
            cost = float(np.linalg.norm(ep)) + 0.05 * ang
            if best is None or cost < best[0]:
                best = (cost, q.copy(), ep.copy(), ang, R.copy())
            if np.linalg.norm(ep) <= tol_pos and ang <= tol_ang:
                break
        _, q, ep, ang, R = best  # residuals belong to exactly this q
        adot = float(R[:, 2] @ Rtm[:, 2])  # directed approach: +dot required; a cross-product residual is zero for opposite vectors and is not used
        margin = float(min((q - lo).min(), (hi - q).min()))
        res = IKResult(False, '', dict(zip(names, map(float, q))), float(np.linalg.norm(ep)), float(ang), adot, total, margin)
        if res.pos_residual > tol_pos:
            res.reason = 'position_residual'
        elif res.angle_residual > tol_ang:
            res.reason = 'angle_residual'
        elif adot <= 0.0:
            res.reason = 'approach_reversed'
        elif margin < 1e-5:
            res.reason = 'joint_limit'
        else:
            res.valid = True
        return res

    def finger_targets(self, side, opening, object_kind, object_size, *, squeeze=0.0):
        # opening None -> calibrated closure for the declared object (cube / sphere); otherwise opening in [0,1], 1 open, 0 closed. squeeze adds closure fraction beyond contact so the position servos load the object.
        try:
            if side not in ('left', 'right'):
                raise HandError('bad_side')
            if object_kind not in SIZE_RANGE:
                return FingerTargets(False, 'unsupported_object_kind')
            size = _num(object_size, 'object_size')
            lo_s, hi_s = SIZE_RANGE[object_kind]
            if not lo_s <= size <= hi_s:
                return FingerTargets(False, 'object_size_out_of_range')
            sq = _num(squeeze, 'squeeze')
            if not 0.0 <= sq <= 0.5:
                raise HandError('bad_squeeze')
            if opening is None:
                f, src = _cal_f(side, object_kind), 'calibrated_kind'
            else:
                f, src = opening_to_closure(opening) * F_MAX, 'opening'
            f = min(1.0, f + sq)
            q = self._hand_from_f(side, f)
        except HandError as e:
            return FingerTargets(False, str(e))
        ctrl = {}
        for h in HAND_DRIVES:
            n = f'{side}_{h}_joint'
            lo, hi = self.m.actuator_ctrlrange[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'a_' + n)]
            ctrl['a_' + n] = float(np.clip(q[n], lo, hi))
        dist = {}
        for fn, g in DISTAL_GAIN.items():
            dn = f'{side}_{fn}_distal_joint'
            dist[dn] = float(min(self.rng[dn][1], g * q[f'{side}_{fn}_proximal_joint']))
        gap = float(CAL['sides'][side][KIND_KEY[object_kind]]['pad_gap_mm']) if src == 'calibrated_kind' and sq == 0.0 else float('nan')
        return FingerTargets(True, '', ctrl, dist, float(f), src, gap, False)
