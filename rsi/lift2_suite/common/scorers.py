'''Role-mapped independent scorers for the LIFT2 suite (privileged: controllers never import these).

generic_snapshot == rsi.scorer.snapshot (inside / nested / on / in_drawer / upright / released, TOL .001, 15 deg upright,
contact touch dist <= .002, support force rule) with ONE change: the robot body set is passed explicitly (the whole LIFT2 base
subtree: base_link, cameras, brackets, arms, fingers) instead of the source name rule "left_/right_". Goal geometry, thresholds,
support edges, dwell and drift come from the source (evaluate is rsi.scorer.evaluate, imported unchanged).
box_strict_snapshot == lift_box_strict_v2.snapshot with the same role mapping; thresholds and evaluate semantics unchanged.
Fail closed: check_roles() requires every non-world body to be an object_* body or a robot body.
'''
import numpy as np
import mujoco
from twin.practice.layouts import body_vertices, world_vertices
from twin.practice.scoring import _inside_profile
from rsi.scorer import TOL, evaluate  # noqa: F401  (re-exported unchanged)
from rsi import scorer as source_scorer
from rsi import tasks as source_tasks  # noqa: F401

BOX_VERSION = 'lift_box_strict_v2_lift2roles'
LONG_AXIS_MAX_DEG = 15.
UPRIGHT_MAX_DEG = 15.
SNAPSHOTS, DWELL_S, DRIFT_M = 11, .5, .005


def robot_bodies_of(model, base='base_link'):
    b0 = model.body(base).id; out = set()
    for b in range(1, model.nbody):
        a = b
        while a and a != b0:
            a = int(model.body_parentid[a])
        if a == b0:
            out.add(b)
    return out


def check_roles(model, robot):
    objs = {b for b in range(1, model.nbody) if model.body(b).name.startswith('object_')}
    other = [model.body(b).name for b in range(1, model.nbody) if b not in objs and b not in robot]
    if other or objs & set(robot):
        raise ValueError(f'unmapped bodies {other}')
    return objs


def support_edges(model, data, robot):
    '''Source _support_edges with an explicit robot set. Objects include static fixtures and slide parts (every object_* body).'''
    objs = {b: model.body(b).name[7:] for b in range(model.nbody) if model.body(b).name.startswith('object_')}
    table = model.geom('table').id
    edges, touching = {}, set()
    for i, c in enumerate(data.contact[:data.ncon]):
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


def generic_snapshot(task, model, data, robot):
    '''task: source Task (goals, upright, record). Same relations as rsi.scorer.snapshot.'''
    record = task.record
    edges, touching = support_edges(model, data, robot)
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


def tilt_deg(axis_z):
    return float(np.degrees(np.arccos(np.clip(abs(float(axis_z)), 0., 1.))))


def box_strict_snapshot(task, model, data, robot):
    s = generic_snapshot(task, model, data, robot)
    cm = data.body('object_carton').xmat.reshape(3, 3); bm = data.body('object_box').xmat.reshape(3, 3)
    long_tilt = tilt_deg(cm[2, 1]); box_tilt = float(np.degrees(np.arccos(np.clip(bm[2, 2], -1., 1.))))
    s['original_checks'] = dict(s['checks'])
    s['checks']['long_axis_vertical:carton'] = bool(long_tilt <= LONG_AXIS_MAX_DEG)
    s['checks']['upright:box'] = bool(box_tilt <= UPRIGHT_MAX_DEG)
    s['carton_long_axis_tilt_deg'] = long_tilt; s['box_tilt_deg'] = box_tilt
    s['valid'] = all(s['checks'].values())
    return s


def box_strict_evaluate(history, numerical_invalid=False, controller_error=None):
    final = history[-1]
    drift = max(float(np.linalg.norm(np.array(final['poses'][n]) - np.array(h['poses'][n]))) for h in history for n in final['poses'])
    failed = sorted({k for h in history for k, v in h['checks'].items() if not v})
    times = np.array([h.get('time', np.nan) for h in history], float)
    dwell_ok = bool(len(history) == SNAPSHOTS and np.all(np.isfinite(times)) and np.all(np.diff(times) > 0) and times[-1] - times[0] >= DWELL_S - 1e-6)
    if not dwell_ok: failed.append('dwell_time')
    if drift >= DRIFT_M: failed.append('dwell_drift')
    if numerical_invalid: failed.append('numerical_invalid')
    if controller_error: failed.append('controller_exception')
    ok = dwell_ok and all(h['valid'] for h in history) and drift < DRIFT_M and not numerical_invalid and not controller_error
    raw = all(all(h['original_checks'].values()) for h in history) and drift < DRIFT_M
    return {'scorer': BOX_VERSION, 'success': bool(ok), 'raw_original_success': bool(raw), 'drift_m': drift, 'dwell_times': times.tolist(),
            'failed_checks': failed, 'carton_long_axis_tilt_deg': final['carton_long_axis_tilt_deg'], 'box_tilt_deg': final['box_tilt_deg'], 'final': final}
