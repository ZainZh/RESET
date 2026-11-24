"""Independent scorer from simulator truth. Controllers never import this.

Success: every goal relation holds, task objects are released (no robot contact),
required objects stay upright, and all of it holds over a 0.5 s dwell with <5 mm drift.
Partial score = fraction of goal relations satisfied at the final sample.
"""
import numpy as np
import mujoco
from twin.practice.layouts import body_vertices, world_vertices
from twin.practice.scoring import _inside_profile

TOL = .001


def _support_edges(model, data):
    objs = {b: model.body(b).name[7:] for b in range(model.nbody) if model.body(b).name.startswith('object_')}
    robot = {b for b in range(model.nbody) if model.body(b).name.startswith(('left_', 'right_'))}
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


def snapshot(task, model, data):
    record = task.record
    edges, touching = _support_edges(model, data)
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


def evaluate(history, drift_limit=.005):
    final = history[-1]
    drift = max(float(np.linalg.norm(np.array(final['poses'][n]) - np.array(h['poses'][n])))
                for h in history for n in final['poses'])
    success = all(h['valid'] for h in history) and drift < drift_limit
    partial = float(np.mean(final['relations'])) if final['relations'] else 0.
    failed = sorted({k for h in history for k, v in h['checks'].items() if not v})
    return {'success': bool(success), 'partial': partial, 'drift_m': drift, 'failed_checks': failed, 'final': final}
