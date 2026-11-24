'''G1 independent scorer g1_scoring_v1: generic, strict upright-box v2 and sweep v3 with ROLE-ONLY changes.

Sources (read-only, copied logic): rsi/scorer.py, rsi/xtrainer_lift_box/lift_box_scorer.py,
rsi/xtrainer_sweep/sweep/scorer_v3.py. Task objects/table/gravity stay in the ORIGINAL SOURCE FRAME; thresholds,
geometry, tolerances, dwell, drift and tilt are unchanged. Only the robot body set (explicit pelvis subtree) and the sweep
held-contact bodies (thumb vs opposing_digits groups) differ. Privileged: never imported by controllers.
'''
import math
import numpy as np
import mujoco
from twin.practice.layouts import body_vertices, world_vertices
from twin.practice.scoring import _inside_profile
from . import g1_roles
from .g1_roles import roles

VERSION = 'g1_scoring_v1'
TOL = .001


# ---------------------------------------------------------------- generic (rsi/scorer.py, robot set = explicit)
def _support_edges(model, data, robot):
    objs = {b: model.body(b).name[7:] for b in range(model.nbody) if model.body(b).name.startswith('object_')}
    table = model.geom('table').id
    edges, touching = {}, set()
    for i, c in enumerate(data.contact):
        ga, gb = int(c.geom1), int(c.geom2)
        a, b = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gb])
        if a not in objs and b not in objs:
            continue
        if c.dist <= .002:
            if a in objs and b in robot: touching.add(objs[a])
            if b in objs and a in robot: touching.add(objs[b])
        w = np.zeros(6); mujoco.mj_contactForce(model, data, i, w)
        f = c.frame.reshape(3, 3).T @ w[:3]
        for child, sup, sg, force in ((b, a, ga, f), (a, b, gb, -f)):
            if child in objs and force[2] > .01 and force[2] / max(np.linalg.norm(force), 1e-12) > .1:
                name = 'table' if sg == table else objs.get(sup)
                if name: edges.setdefault(objs[child], set()).add(name)
    return edges, touching


def generic_snapshot(task, model, data, robot=None):
    robot = roles(model)['robot'] if robot is None else robot
    record = task.record
    edges, touching = _support_edges(model, data, robot)
    checks, rel = {}, []
    for kind, child, parent in task.goals:
        key = f'{kind}:{child}:{parent}'
        cb = data.body('object_' + child)
        if kind in ('inside', 'nested'):
            sup = data.body('object_' + parent)
            local = (world_vertices(model, data, 'object_' + child) - sup.xpos) @ sup.xmat.reshape(3, 3)
            ok, m = _inside_profile(local, record['container_collision_geometry'][parent], TOL, partial=kind == 'nested')
            if kind == 'nested':
                ok = ok and m.get('insertion_m', -1) > .015 and parent in edges.get(child, ())
            else:
                ok = ok and bool(edges.get(child))
        elif kind == 'on':
            sup = data.body('object_' + parent)
            sv = world_vertices(model, data, 'object_' + parent)
            cv = world_vertices(model, data, 'object_' + child)
            lo, hi = sv[:, :2].min(0), sv[:, :2].max(0)
            ok = (parent in edges.get(child, ()) and cv[:, 2].min() > sv[:, 2].max() - .01
                  and np.all(cb.xpos[:2] > lo) and np.all(cb.xpos[:2] < hi))
        elif kind == 'in_drawer':
            d = data.body('object_' + parent)
            v = body_vertices(model, 'object_' + parent)
            p = (cb.xpos - d.xpos) @ d.xmat.reshape(3, 3)
            ok = bool(np.all(p[:2] > v[:, :2].min(0) + .005) and np.all(p[:2] < v[:, :2].max(0) - .005)
                      and v[:, 2].min() - .01 < p[2] < v[:, 2].max() + .01 and parent in edges.get(child, ()))
        else:
            raise ValueError(kind)
        checks[key] = bool(ok); rel.append(bool(ok))
    for n in task.upright:
        z = data.body('object_' + n).xmat.reshape(3, 3)[2, 2]
        checks[f'upright:{n}'] = bool(np.degrees(np.arccos(np.clip(z, -1, 1))) <= 15)
    checks['released'] = not touching
    names = sorted({g[1] for g in task.goals} | {g[2] for g in task.goals})
    poses = {n: data.body('object_' + n).xpos.tolist() for n in names}
    return {'time': float(data.time), 'checks': checks, 'relations': rel, 'poses': poses,
            'valid': all(checks.values()), 'support': {k: sorted(v) for k, v in edges.items()}}


def generic_evaluate(history, drift_limit=.005):
    final = history[-1]
    drift = max(float(np.linalg.norm(np.array(final['poses'][n]) - np.array(h['poses'][n])))
                for h in history for n in final['poses'])
    success = all(h['valid'] for h in history) and drift < drift_limit
    partial = float(np.mean(final['relations'])) if final['relations'] else 0.
    failed = sorted({k for h in history for k, v in h['checks'].items() if not v})
    return {'success': bool(success), 'partial': partial, 'drift_m': drift, 'failed_checks': failed, 'final': final}


# ---------------------------------------------------------------- strict upright box v2 (lift_box_scorer.py)
LONG_AXIS_MAX_DEG = 15.
UPRIGHT_MAX_DEG = 15.
DRIFT_M = .005
SNAPSHOTS = 11
DWELL_S = .5
BOX_VERSION = 'g1_lift_box_strict_v2'


def tilt_deg(axis_z):
    return float(np.degrees(np.arccos(np.clip(abs(float(axis_z)), 0., 1.))))


def box_snapshot(task, model, data, robot=None):
    s = generic_snapshot(task, model, data, robot)
    cm = data.body('object_carton').xmat.reshape(3, 3)
    bm = data.body('object_box').xmat.reshape(3, 3)
    long_tilt = tilt_deg(cm[2, 1])
    box_tilt = float(np.degrees(np.arccos(np.clip(bm[2, 2], -1., 1.))))
    s['original_checks'] = dict(s['checks'])
    s['checks']['long_axis_vertical:carton'] = bool(long_tilt <= LONG_AXIS_MAX_DEG)
    s['checks']['upright:box'] = bool(box_tilt <= UPRIGHT_MAX_DEG)
    s['carton_long_axis_tilt_deg'] = long_tilt
    s['box_tilt_deg'] = box_tilt
    s['valid'] = all(s['checks'].values())
    return s


def box_evaluate(history, numerical_invalid=False, controller_error=None):
    final = history[-1]
    drift = max(float(np.linalg.norm(np.array(final['poses'][n]) - np.array(h['poses'][n])))
        for h in history for n in final['poses'])
    failed = sorted({k for h in history for k, v in h['checks'].items() if not v})
    times = np.array([h.get('time', np.nan) for h in history], float)
    dwell_ok = bool(len(history) == SNAPSHOTS and np.all(np.isfinite(times)) and np.all(np.diff(times) > 0)
        and times[-1] - times[0] >= DWELL_S - 1e-6)
    if not dwell_ok: failed.append('dwell_time')
    if drift >= DRIFT_M: failed.append('dwell_drift')
    if numerical_invalid: failed.append('numerical_invalid')
    if controller_error: failed.append('controller_exception')
    ok = dwell_ok and all(h['valid'] for h in history) and drift < DRIFT_M and not numerical_invalid and not controller_error
    raw = all(all(h['original_checks'].values()) for h in history) and drift < DRIFT_M
    return {'scorer': BOX_VERSION, 'success': bool(ok), 'raw_original_success': bool(raw), 'drift_m': drift,
        'dwell_times': times.tolist(), 'failed_checks': failed,
        'carton_long_axis_tilt_deg': final['carton_long_axis_tilt_deg'],
        'box_tilt_deg': final['box_tilt_deg'], 'final': final}


# ---------------------------------------------------------------- sweep v3 (scorer_v3.py; roles adapted only)
VERTEX_MARGIN = .002
Z_LO, Z_HI = -.016, .060
PUSH_MIN = .05
LIFT_MAX = .05
TILT_MAX = 25.
STEP_SLACK = .001
DWELL_SAMPLES, DWELL_DT, DWELL_SPAN, TIME_TOL = 11, .05, .5, 1e-6
SUPPORT_MIN_N = .01
SWEEP_VERSION = 'g1_sweep_v3'


def net_support_force(fz):
    '''Net signed world-z force (N) of cube-pan contacts on the cube. Order/split invariant (math.fsum); non-finite fails closed.'''
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


def _sets(model, rl=None):
    rl = roles(model) if rl is None else rl
    body = lambda n: model.body(n).id
    free = [model.body(n).id for n, _, _ in _free_slices(model)]
    return {'cube': body('object_orange_cube'), 'brush': body('object_brush'), 'pan': body('object_dustpan'),
        'robot': set(rl['robot']), 'lg': dict(rl['groups']['left']), 'rg': dict(rl['groups']['right']),
        'table': model.geom('table').id, 'free': free}


def _contacts(model, data, S):
    out = {'robot_cube': False, 'brush_cube': False, 'cube_table': False, 'pan_cube_support': False,
        'lg_pan': set(), 'rg_brush': set(), 'pan_cube_fz': []}
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
                f = f if b2 == S['cube'] else -f
                out['pan_cube_fz'].append(float(f[2]))
        if S['pan'] in pair:
            for b in pair:
                if b in S['lg']: out['lg_pan'].add(S['lg'][b])
        if S['brush'] in pair:
            for b in pair:
                if b in S['rg']: out['rg_brush'].add(S['rg'][b])
    out['pan_cube_net_fz'] = net_support_force(out['pan_cube_fz'])
    out['pan_cube_support'] = bool(out['pan_cube_net_fz'] > SUPPORT_MIN_N)
    return out


class Monitor:
    '''Trusted per-step evidence and exact state guard (v3 protocol: create after reset, pre_step() before every physics step
    and before scoring, update() after every physics step). Covers FREE object bodies only: articulated/slide-jointed drawer
    DOFs need the separate trusted runtime guard; do not claim this guard covers them.'''

    def __init__(self, model, data, robot_roles=None):
        self.S = _sets(model, robot_roles)
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
        for b in self.S['free']:
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
            if len(C['rg_brush']) == 2: self.push += d
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


def sweep_snapshot(model, data, robot_roles=None):
    S = _sets(model, robot_roles)
    C = _contacts(model, data, S)
    cube = data.body('object_orange_cube').xpos.copy()
    inside, vloc = cube_inside(model, data)
    tilt = float(np.degrees(np.arccos(np.clip(data.body('object_dustpan').xmat.reshape(3, 3)[2, 2], -1, 1))))
    checks = {
        'cube_in_pan': bool(inside and C['pan_cube_support'] and not C['cube_table']),
        'pan_held': bool(len(C['lg_pan']) == 2 and tilt <= TILT_MAX),
        'brush_held': bool(len(C['rg_brush']) == 2),
        'no_robot_cube_now': not C['robot_cube'],
    }
    return {'time': float(data.time), 'checks': checks, 'cube': cube.tolist(),
        'cube_pan_local': _pan_local(model, data, cube).tolist(),
        'cube_vertex_local_min': vloc.min(0).tolist(), 'cube_vertex_local_max': vloc.max(0).tolist(),
        'cube_vertices_inside': inside, 'pan_support': C['pan_cube_support'], 'cube_table': C['cube_table'],
        'pan_cube_net_fz_N': C['pan_cube_net_fz'], 'pan_cube_n_contacts': len(C['pan_cube_fz']),
        'pan_cube_max_fz_N': max(C['pan_cube_fz']) if C['pan_cube_fz'] else None, 'scorer_version': SWEEP_VERSION,
        'pan_tilt_deg': tilt, 'pan_groups': sorted(C['lg_pan']), 'brush_groups': sorted(C['rg_brush']), 'valid': all(checks.values())}


def dwell_time_ok(history):
    try:
        t = [float(h['time']) for h in history]
    except (KeyError, TypeError, ValueError):
        return False
    if len(t) < DWELL_SAMPLES or not all(math.isfinite(x) for x in t):
        return False
    steps = np.diff(t)
    return bool(np.all(steps >= DWELL_DT - TIME_TOL) and t[-1] - t[0] >= DWELL_SPAN - TIME_TOL)


def sweep_evaluate(history, monitor, numerical_invalid=False):
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
    return {'scorer': SWEEP_VERSION, 'raw_goal_success': bool(raw), 'partial': float(partial), 'drift_m': drift,
        'failed_checks': failed, 'episode_checks': episode, 'monitor': monitor, 'final': final}


def sweep_final_success(scored, controller_error, numerical_invalid, scoring_error=None):
    reasons = []
    if scoring_error: reasons.append('scoring_error')
    if controller_error: reasons.append('controller_error')
    if numerical_invalid: reasons.append('numerical_invalid')
    if not scored.get('raw_goal_success', False): reasons.append('goal_not_met')
    return (not reasons), reasons
