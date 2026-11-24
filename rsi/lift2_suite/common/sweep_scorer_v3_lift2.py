'''LIFT2 role-mapped copy of rsi/xtrainer_sweep/sweep/scorer_v3.py (independently accepted v3). ONLY change: robot body set = whole LIFT2 base subtree,
finger roles = left_link17/18 and right_link27/28 instead of left_/right_finger1/2. Thresholds, Monitor, pan geometry, dwell, drift, injection guard unchanged.
The Monitor's free-body guard covers free objects only; the runner additionally protects every non-robot DOF through TargetWorld.'''
"""Independent sweep_objects scorer v3 (versioned; v2 = scorer.py stays frozen). ONLY change vs v2: cube-pan support is
the NET signed world-z force of all cube-pan contacts (dist <= 2 mm) on the cube, compared with the same 0.01 N threshold
(SUPPORT_MIN_N). v2 required one single contact > 0.01 N, which fails a cube resting on many contacts (director review
SWEEP_SUPPORT_REVIEW.md: 62 contacts, sum 0.296 N, max 0.008 N). All other checks, geometry, exclusions, dwell, drift and limits = v2.

v2 docstring follows.
Independent sweep_objects scorer from simulator truth. Controllers never import this.

Raw goal success requires, at every one of 11 samples over a 0.5 s dwell after the controller returns:
  cube_in_pan   ALL cube collision-mesh vertices inside the dustpan pan region (dustpan frame, conservative
  inner-wall/mouth bounds shrunk by VERTEX_MARGIN), the cube supported by the dustpan (upward
         contact force) and not touching the table;
  pan_held      both left fingers in contact with the dustpan, pan tilt <= TILT_MAX;
  brush_held    both right fingers in contact with the brush;
  no_robot_cube_now  no robot geometry touching the cube;
and over the whole episode (trusted per-step Monitor, never exposed to the policy):
  brush_pushed  >= PUSH_MIN cumulative cube xy travel while the cube touches the brush AND the brush is held
         by both right fingers AND no robot geometry touches the cube;
  no_robot_cube no robot geometry ever touches the cube (excludes direct finger/arm push or pick);
  cube_low      cube centre never higher than LIFT_MAX above the table while outside the pan (no carrying);
  no_injection  exact pre-step state guard: before every physics step (and before scoring) the qpos/qvel of every
         free object must equal the state recorded after the previous legitimate step (or after reset);
         any position, quaternion or velocity write is rejected. Also zero xfrc_applied/qfrc_applied.
  (A velocity-based pose-jump heuristic is kept as a diagnostic only.)
  dwell_time    sample times finite, strictly increasing, each interval >= DWELL_DT - TIME_TOL and total span
  >= DWELL_SPAN - TIME_TOL (TIME_TOL = 1e-6 s << 1 ms physics step), 11 samples;
  dwell_drift   cube drift < 5 mm over the dwell; numerically_valid (no MuJoCo/robot-limit failure).
Final success (runner) additionally requires that the controller returned without error or time-cap overrun
(fail closed: see final_success).

Pan region (dustpan_v2 local frame, metres; asset walls at |x| 0.076-0.080, rounded wall outer radius 0.080 about
(0,-0.065), orange mouth y -0.130..-0.120, floor top z -0.012): floor rectangle |x| <= 0.076, y in
[-0.120, -0.065] joined with the half-disc centre (0, -0.065) radius 0.075, z in [-0.016, 0.060].
Partial = fraction of {cube_in_pan, pan_held, brush_held, brush_pushed} at the final sample.
"""
import math
import numpy as np
import mujoco

VERTEX_MARGIN = .002
Z_LO, Z_HI = -.016, .060
PUSH_MIN = .05
LIFT_MAX = .05
TILT_MAX = 25.
STEP_SLACK = .001
DWELL_SAMPLES, DWELL_DT, DWELL_SPAN, TIME_TOL = 11, .05, .5, 1e-6
LEFT_FINGERS, RIGHT_FINGERS = ('left_link17', 'left_link18'), ('right_link27', 'right_link28')   # LIFT2 role map (was left_/right_finger1/2)


def robot_bodies_of(model, base='base_link'):
    b0 = model.body(base).id; out = set()
    for b in range(1, model.nbody):
        a = b
        while a and a != b0:
            a = int(model.body_parentid[a])
        if a == b0:
            out.add(b)
    return out


SUPPORT_MIN_N = .01   # same threshold as v2, applied to the net cube-pan force


def net_support_force(fz):
    """Net signed world-z force (N) of the cube-pan contacts on the cube. Order/split invariant (math.fsum); non-finite fails closed."""
    fz = [float(x) for x in fz]
    if not all(math.isfinite(x) for x in fz):
        return float('-inf')
    return math.fsum(fz)


def _pan_local(model, data, p):
    pan = data.body('object_dustpan')
    return (np.asarray(p) - pan.xpos) @ pan.xmat.reshape(3, 3)


def in_pan_region(local, margin=VERTEX_MARGIN):
    x, y, z = local
    if z < Z_LO or z > Z_HI:
        return False
    if -.120 + margin <= y <= -.065 and abs(x) <= .076 - margin:
        return True
    return y > -.065 and np.hypot(x, y + .065) <= .075 - margin


def cube_vertices(model, data):
    """World coordinates of the cube collision-mesh vertices (all collision geoms of the cube body)."""
    b = model.body('object_orange_cube').id; out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != b or not model.geom_contype[g]:
            continue
        R = data.geom_xmat[g].reshape(3, 3)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]; a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            out.append(model.mesh_vert[a:a + n] @ R.T + data.geom_xpos[g])
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            s = model.geom_size[g]; c = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * s
            out.append(c @ R.T + data.geom_xpos[g])
        else:
            raise ValueError('unsupported cube collision geometry')
    return np.concatenate(out)


def cube_inside(model, data, margin=VERTEX_MARGIN):
    local = _pan_local(model, data, cube_vertices(model, data))
    return bool(all(in_pan_region(v, margin) for v in local)), local


def cube_outside_pan(model, data):
    """Layout validity: no cube vertex within 30 mm of the pan region (xy)."""
    local = _pan_local(model, data, cube_vertices(model, data))
    return not any(in_pan_region(np.r_[v[:2], 0.], margin=-.03) for v in local)


def _table_height(model, data, p):
    g = model.geom('table').id
    R = data.geom_xmat[g].reshape(3, 3); top = data.geom_xpos[g] + R[:, 2] * model.geom_size[g, 2]
    return float((np.asarray(p) - top) @ R[:, 2])


def _free_slices(model):
    out = []
    for b in range(model.nbody):
        if not model.body(b).name.startswith('object_') or model.body_jntnum[b] != 1:
            continue
        j = int(model.body_jntadr[b])
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            out.append((model.body(b).name, int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])))
    return out


def _sets(model):
    body = lambda n: model.body(n).id
    robot = robot_bodies_of(model)   # LIFT2 role map: whole base_link subtree (arms, fingers, wrist/head cameras, brackets)
    free = [model.body(n).id for n, _, _ in _free_slices(model)]
    return {'cube': body('object_orange_cube'), 'brush': body('object_brush'), 'pan': body('object_dustpan'),
        'robot': robot, 'lf': {body(n) for n in LEFT_FINGERS}, 'rf': {body(n) for n in RIGHT_FINGERS},
        'table': model.geom('table').id, 'free': free}


def _contacts(model, data, S):
    """Booleans/sets from the current contact list (dist <= 2 mm counts as touch)."""
    out = {'robot_cube': False, 'brush_cube': False, 'cube_table': False, 'pan_cube_support': False,
        'lf_pan': set(), 'rf_brush': set(), 'pan_cube_fz': []}
    for i, c in enumerate(data.contact):
        if c.dist > .002:
            continue
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        pair = {b1, b2}
        if S['cube'] in pair:
            other, og = (b2, g2) if b1 == S['cube'] else (b1, g1)
            if other in S['robot']: out['robot_cube'] = True
            if other == S['brush']: out['brush_cube'] = True
            if og == S['table']: out['cube_table'] = True
            if other == S['pan']:
                w = np.zeros(6); mujoco.mj_contactForce(model, data, i, w)
                f = c.frame.reshape(3, 3).T @ w[:3]
                f = f if b2 == S['cube'] else -f   # same sign convention as rsi/scorer.py support edges
                out['pan_cube_fz'].append(float(f[2]))   # v3: signed world-z force on the cube, aggregated below
        if S['pan'] in pair:
            for b in pair & S['lf']: out['lf_pan'].add(b)
        if S['brush'] in pair:
            for b in pair & S['rf']: out['rf_brush'].add(b)
    out['pan_cube_net_fz'] = net_support_force(out['pan_cube_fz'])
    out['pan_cube_support'] = bool(out['pan_cube_net_fz'] > SUPPORT_MIN_N)
    return out


class Monitor:
    """Trusted per-step evidence and exact state guard. Lives in the runner; the policy's io never references it.
    Protocol: create immediately after reset (records the legitimate state); call pre_step() before every physics step
    and before scoring; call update() after every physics step (records the new legitimate state)."""

    def __init__(self, model, data):
        self.S = _sets(model)
        self.slices = _free_slices(model)
        self.prev = data.body('object_orange_cube').xpos.copy()
        self.prev_free = {b: data.xpos[b].copy() for b in self.S['free']}
        self.prev_time = float(data.time)
        self.push = 0.; self.loose_push = 0.; self.robot_cube = False; self.robot_cube_time = None
        self.max_lift_outside = 0.; self.steps = 0; self.pre_checks = 0
        self.injection = None; self.pose_jump_diag = None
        self._record(data)

    def _record(self, data):
        self.expected = {n: (data.qpos[a:a + 7].copy(), data.qvel[v:v + 6].copy()) for n, a, v in self.slices}

    def _flag(self, info):
        if self.injection is None:
            self.injection = info

    def pre_step(self, model, data):
        """Exact guard: free-object qpos/qvel must be bit-identical to the last legitimate post-step state."""
        self.pre_checks += 1
        for n, a, v in self.slices:
            q0, v0 = self.expected[n]
            q, qd = data.qpos[a:a + 7], data.qvel[v:v + 6]
            if not np.array_equal(q[:3], q0[:3]):
                self._flag({'kind': 'object_position_write', 'body': n, 'time': float(data.time), 'delta_m': float(np.linalg.norm(q[:3] - q0[:3]))})
            if not np.array_equal(q[3:], q0[3:]):
                self._flag({'kind': 'object_orientation_write', 'body': n, 'time': float(data.time)})
            if not np.array_equal(qd, v0):
                self._flag({'kind': 'object_velocity_write', 'body': n, 'time': float(data.time)})
        if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
            self._flag({'kind': 'applied_force', 'time': float(data.time)})

    def update(self, model, data):
        self.steps += 1
        dt = max(float(data.time) - self.prev_time, model.opt.timestep)
        if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
            self._flag({'kind': 'applied_force', 'time': float(data.time)})
        for b in self.S['free']:   # diagnostic only (the exact guard is pre_step)
            p = data.xpos[b]; vel = float(np.linalg.norm(data.cvel[b][3:]))
            step = float(np.linalg.norm(p - self.prev_free[b]))
            if self.pose_jump_diag is None and step > vel * dt * 3 + STEP_SLACK:
                self.pose_jump_diag = {'body': model.body(b).name, 'time': float(data.time), 'step_m': step}
            self.prev_free[b] = p.copy()
        self.prev_time = float(data.time)
        C = _contacts(model, data, self.S)
        p = data.body('object_orange_cube').xpos.copy()
        d = float(np.linalg.norm(p[:2] - self.prev[:2]))
        if C['brush_cube'] and not C['robot_cube']:
            if len(C['rf_brush']) == 2: self.push += d
            else: self.loose_push += d
        if C['robot_cube'] and not self.robot_cube:
            self.robot_cube = True; self.robot_cube_time = float(data.time)
        if not in_pan_region(_pan_local(model, data, p), margin=0.):
            self.max_lift_outside = max(self.max_lift_outside, _table_height(model, data, p))
        self.prev = p
        self._record(data)

    def summary(self):
        return {'brush_push_travel_m': self.push, 'loose_brush_push_travel_m': self.loose_push,
            'robot_cube_contact': self.robot_cube, 'robot_cube_first_time': self.robot_cube_time,
            'max_cube_height_outside_pan_m': self.max_lift_outside, 'injection': self.injection,
            'pose_jump_diagnostic': self.pose_jump_diag, 'monitored_steps': self.steps, 'pre_step_checks': self.pre_checks}


def snapshot(model, data):
    S = _sets(model)
    C = _contacts(model, data, S)
    cube = data.body('object_orange_cube').xpos.copy()
    inside, vloc = cube_inside(model, data)
    tilt = float(np.degrees(np.arccos(np.clip(data.body('object_dustpan').xmat.reshape(3, 3)[2, 2], -1, 1))))
    checks = {
        'cube_in_pan': bool(inside and C['pan_cube_support'] and not C['cube_table']),
        'pan_held': bool(len(C['lf_pan']) == 2 and tilt <= TILT_MAX),
        'brush_held': bool(len(C['rf_brush']) == 2),
        'no_robot_cube_now': not C['robot_cube'],
    }
    return {'time': float(data.time), 'checks': checks, 'cube': cube.tolist(),
        'cube_pan_local': _pan_local(model, data, cube).tolist(),
        'cube_vertex_local_min': vloc.min(0).tolist(), 'cube_vertex_local_max': vloc.max(0).tolist(),
        'cube_vertices_inside': inside, 'pan_support': C['pan_cube_support'], 'cube_table': C['cube_table'],
        'pan_cube_net_fz_N': C['pan_cube_net_fz'], 'pan_cube_n_contacts': len(C['pan_cube_fz']),
        'pan_cube_max_fz_N': max(C['pan_cube_fz']) if C['pan_cube_fz'] else None, 'scorer_version': 'v3',
        'pan_tilt_deg': tilt, 'valid': all(checks.values())}


def dwell_time_ok(history):
    try:
        t = [float(h['time']) for h in history]
    except (KeyError, TypeError, ValueError):
        return False
    if len(t) < DWELL_SAMPLES or not all(math.isfinite(x) for x in t):
        return False
    steps = np.diff(t)
    return bool(np.all(steps >= DWELL_DT - TIME_TOL) and t[-1] - t[0] >= DWELL_SPAN - TIME_TOL)


def evaluate(history, monitor, numerical_invalid=False):
    """Raw goal evaluation from trusted history + monitor. Does not know about controller errors (see final_success)."""
    final = history[-1]
    drift = max(float(np.linalg.norm(np.array(final['cube']) - np.array(h['cube']))) for h in history)
    episode = {
        'brush_pushed': monitor['brush_push_travel_m'] >= PUSH_MIN,
        'no_robot_cube': not monitor['robot_cube_contact'],
        'cube_low': monitor['max_cube_height_outside_pan_m'] <= LIFT_MAX,
        'no_injection': monitor.get('injection') is None,
        'numerically_valid': not numerical_invalid,
        'dwell_time': dwell_time_ok(history),
        'dwell_drift': drift < .005,
    }
    raw = all(h['valid'] for h in history) and all(episode.values())
    keys = ('cube_in_pan', 'pan_held', 'brush_held')
    partial = (sum(final['checks'][k] for k in keys) + episode['brush_pushed']) / 4
    failed = sorted({k for h in history for k, v in h['checks'].items() if not v} | {k for k, v in episode.items() if not v})
    return {'raw_goal_success': bool(raw), 'partial': float(partial), 'drift_m': drift, 'failed_checks': failed,
        'episode_checks': episode, 'monitor': monitor, 'final': final}


def final_success(scored, controller_error, numerical_invalid, scoring_error=None):
    """Fail closed: goal reached AND controller finished cleanly (no exception, no time cap) AND valid physics."""
    reasons = []
    if scoring_error: reasons.append('scoring_error')
    if controller_error: reasons.append('controller_error')
    if numerical_invalid: reasons.append('numerical_invalid')
    if not scored.get('raw_goal_success', False): reasons.append('goal_not_met')
    return (not reasons), reasons
