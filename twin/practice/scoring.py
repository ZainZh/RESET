"""External full-task scorer. Never import this module into policy code.

snapshot() is a JSON-serializable immutable observation of scorer truth.
Runtime must supply 11 snapshots at t, t+.05, ..., t+.50 after the program
returns, within its episode budget; no task success is granted by a program.
Geometry/contact predicates must hold at every sample, including the final
live state. Synthetic tests are software validation, never task evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .layouts import TASK_OBJECTS, body_vertices, world_vertices

CONFIG_PATH = Path(__file__).resolve().parents[2] / 'configs/practice/scoring.json'


def _config():
    raw = CONFIG_PATH.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _inside_profile(points, profile, tolerance, partial=False):
    """All exterior convex-envelope points must fit the piecewise cavity.

    Add intersections of hull edges with every profile plane and rim before
    checking. Thus a cup cannot pass by putting only its center/base inside;
    below-rim exterior material must fit at every profile slope break. Above
    the rim is allowed only for nesting. Uses body's own transformed geometry.
    """
    points = np.asarray(points)
    levels = np.array(profile['profile_z_m'], dtype=float)
    wall = float(profile['wall_thickness_m'])
    floor, rim = float(levels[0]), float(levels[-1])
    # Profile z[0] is top of the bottom collision slab in these exports.
    # Inspect exported collision bottom surface rather than add wall twice.
    bottom = float(points[:, 2].min())
    if bottom < floor - tolerance:
        return False, dict(bottom_m=bottom, cavity_floor_m=floor, rim_m=rim, reason='below_cavity_floor')
    if not partial and np.max(points[:, 2]) > rim + tolerance:
        return False, dict(top_m=float(points[:, 2].max()), rim_m=rim, reason='above_rim')
    hull = ConvexHull(points)
    edges = {tuple(sorted((int(a), int(b)))) for face in hull.simplices
             for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))}
    extra = []
    for a, b in edges:
        pa, pb = points[a], points[b]
        if abs(pb[2] - pa[2]) < 1e-12:
            continue
        for level in levels:
            ratio = (level - pa[2]) / (pb[2] - pa[2])
            if 0 < ratio < 1:
                extra.append(pa + ratio * (pb - pa))
    p = np.concatenate((points, np.asarray(extra).reshape(-1, 3)))
    if partial:
        p = p[p[:, 2] <= rim + 1e-10]
    if not len(p):
        return False, dict(reason='no_inserted_material', insertion_m=rim - bottom)
    radius = np.linalg.norm(p[:, :2] - np.array(profile['axis_xy_m']), axis=1)
    allowed = np.interp(p[:, 2], levels, profile['outer_radius_m']) - wall
    clearance = float(np.min(allowed - radius))
    return bool(clearance >= -tolerance), dict(radial_clearance_m=clearance, bottom_m=bottom,
                                              cavity_floor_m=floor, rim_m=rim, insertion_m=rim - bottom)


def _support_graph(model, data, config):
    """Directed loaded contacts: supported body -> lower supporting body.

    Contact frame normal points geom1 -> geom2; full contact wrench is rotated
    to world. Only an upward resultant on the supported body contributes.
    Table is world body 0, but only the named table geom can anchor a chain.
    """
    edges, contacts, robot_contacts = {}, [], set()
    objects = {model.body('object_' + n).id: n for n in sum((list(x) for x in TASK_OBJECTS.values()), [])
               if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'object_' + n) >= 0}
    normal = np.array([0., 0., 1.])
    table = model.geom('table').id
    robot_ids = {i for i in range(model.nbody) if model.body(i).name.startswith(('left_', 'right_'))}
    for i, contact in enumerate(data.contact):
        ga, gb = int(contact.geom1), int(contact.geom2)
        a, b = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gb])
        if a not in objects and b not in objects:
            continue
        if contact.dist <= config['robot_contact_clearance_m']:
            if a in objects and b in robot_ids:
                robot_contacts.add(objects[a])
            if b in objects and a in robot_ids:
                robot_contacts.add(objects[b])
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        world_force = contact.frame.reshape(3, 3).T @ wrench[:3]
        for child, support, support_geom, force in ((b, a, ga, world_force), (a, b, gb, -world_force)):
            if child not in objects:
                continue
            up = float(force @ normal)
            magnitude = float(np.linalg.norm(force))
            support_name = 'table' if support_geom == table else objects.get(support)
            if (support_name and up >= config['minimum_support_force_N'] and
                    up / max(magnitude, 1e-15) >= config['minimum_support_normal_up']):
                edges.setdefault(objects[child], set()).add(support_name)
        contacts.append(dict(a=objects.get(a, model.body(a).name), b=objects.get(b, model.body(b).name),
                             distance_m=float(contact.dist), force_N=world_force.tolist()))
    return {k: sorted(v) for k, v in edges.items()}, contacts, sorted(robot_contacts)


def _anchored(name, edges, allowed, visited=None):
    if name == 'table':
        return True
    visited = set() if visited is None else visited
    if name in visited or name not in allowed:
        return False
    return any(_anchored(parent, edges, allowed, visited | {name}) for parent in edges.get(name, []))


def snapshot(task, model, data, record) -> dict:
    """Capture one scorer sample without stepping or mutating simulation state.

    Public result schema: task,time_sec,config_sha256,valid,checks,measurements,
    poses (position + scalar-first quaternion), velocities, support_edges.
    `evaluate` consumes these directly; runtime need not inspect their internals.
    """
    names = TASK_OBJECTS[task]
    config, config_hash = _config()
    checks, measures, poses, velocities = {}, {}, {}, {}
    finite = all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.qacc, data.xpos, data.xmat))
    warnings = {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number}
    # Conservatively fail any MuJoCo warning: overflow and numerical repairs
    # invalidate evidence, even if the last repaired qpos is finite.
    checks['numerical_validity'] = bool(finite and np.isfinite(data.time) and not warnings)
    measures['warnings'] = warnings
    if not finite:
        return dict(task=task, time_sec=float(data.time), config_sha256=config_hash, valid=False,
                    checks=checks, measurements=measures, poses={}, velocities={}, support_edges={})
    edges, contacts, robot_contacts = _support_graph(model, data, config)
    measures['contacts'] = contacts
    checks['released_no_robot_support'] = not robot_contacts
    measures['robot_contacts'] = robot_contacts
    relevant_contacts = [c for c in contacts]
    checks['no_deep_penetration'] = all(c['distance_m'] >= -config['max_penetration_m'] for c in relevant_contacts)
    for n in names:
        b = data.body('object_' + n)
        rotation = b.xmat.reshape(3, 3)
        poses[n] = np.r_[b.xpos, np.roll(Rotation.from_matrix(rotation).as_quat(), 1)].tolist()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, model.body('object_' + n).id, velocity, 0)
        velocities[n] = velocity.tolist()  # angular, then linear (MuJoCo convention)
        checks[n + ':slow'] = bool(np.linalg.norm(velocity[:3]) <= config['max_angular_speed_rad_s'] and
                                    np.linalg.norm(velocity[3:]) <= config['max_linear_speed_m_s'])
    containers = record['container_collision_geometry']
    targets = [('yellow_cup', 'blue_cup'), ('purple_cup', 'yellow_cup')] if task == 'stack_cups' else [(n, 'bowl') for n in names if n != 'bowl']
    upright_names = names if task == 'stack_cups' else ('bowl',)
    for n in upright_names:
        up = data.body('object_' + n).xmat.reshape(3, 3)[:, 2]
        angle = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        measures[n + ':upright_deg'] = angle
        checks[n + ':upright'] = angle <= config['upright_max_deg']
    contained = {}
    for child, parent in targets:
        support = data.body('object_' + parent)
        vertices = world_vertices(model, data, 'object_' + child)
        local = (vertices - support.xpos) @ support.xmat.reshape(3, 3)
        inside, metrics = _inside_profile(local, containers[parent], config['geometry_tolerance_m'], partial=task == 'stack_cups')
        relation = child + '_in_' + parent
        checks[relation + ':geometry'] = inside
        if task == 'stack_cups':
            checks[relation + ':insertion'] = metrics.get('insertion_m', -np.inf) > config['minimum_insertion_m']
            checks[relation + ':direct_support'] = parent in edges.get(child, [])
        contained[child] = inside
        measures[relation] = metrics
    allowed = set(names) | {'table'}
    # A cube may rest on another contained cube, but cannot be supported by a
    # robot/escaped object. Every chain must terminate in the actual table.
    for n in names:
        checks[n + ':supported'] = _anchored(n, edges, allowed)
    if task != 'stack_cups':
        for n in contained:
            def reaches_bowl(child, seen):
                if child == 'bowl':
                    return True
                if child in seen or not contained.get(child, False):
                    return False
                return any(reaches_bowl(p, seen | {child}) for p in edges.get(child, []))
            checks[n + ':container_support'] = reaches_bowl(n, set())
    return dict(task=task, time_sec=float(data.time), config_sha256=config_hash,
                valid=bool(all(checks.values())), checks=checks, measurements=measures,
                poses=poses, velocities=velocities, support_edges=edges)


def evaluate(task, model, data, record, history: list[dict]) -> dict:
    """Score final live state plus mandatory, contiguous final dwell evidence."""
    config, config_hash = _config()
    current = snapshot(task, model, data, record)
    checks = {'final_state': current['valid'], 'history_schema': False, 'dwell_coverage': False,
              'dwell_all_valid': False, 'stable_position': False, 'stable_orientation': False}
    measurements = dict(final=current, dwell_samples=len(history))
    try:
        samples = list(history)
        checks['history_schema'] = bool(samples and all(s['task'] == task and s['config_sha256'] == config_hash and
                                                       isinstance(s['valid'], bool) for s in samples))
        times = np.asarray([s['time_sec'] for s in samples], dtype=float)
        tolerance = config['sample_time_tolerance_sec']
        checks['dwell_coverage'] = bool(len(times) >= 11 and np.isfinite(times).all() and
            abs(times[-1] - data.time) <= 1e-6 and
            times[-1] - times[0] >= config['dwell_sec'] - 1e-6 and
            times[-1] - times[0] <= config['dwell_sec'] + tolerance and
            np.all(np.abs(np.diff(times) - config['sample_period_sec']) <= tolerance))
        checks['dwell_all_valid'] = bool(samples and all(s['valid'] and all(s['checks'].values()) for s in samples))
        drift, angular = {}, {}
        for n in TASK_OBJECTS[task]:
            p = np.asarray([s['poses'][n] for s in samples] + [current['poses'][n]], dtype=float)
            if not np.isfinite(p).all():
                raise ValueError('nonfinite dwell pose')
            # Maximum pairwise excursion, not just endpoint difference.
            drift[n] = float(np.linalg.norm(p[:, None, :3] - p[None, :, :3], axis=-1).max())
            r = Rotation.from_quat(np.roll(p[:, 3:], -1, axis=1)).as_matrix()
            relative = r[:, None].transpose(0, 1, 3, 2) @ r[None, :]
            angular[n] = float(np.degrees(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))).max())
        checks['stable_position'] = all(x <= config['max_position_drift_m'] for x in drift.values())
        checks['stable_orientation'] = all(x <= config['max_rotation_drift_deg'] for x in angular.values())
        measurements.update(position_drift_m=drift, rotation_drift_deg=angular, sample_times_sec=times.tolist())
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        measurements['history_error'] = str(exc)
    success = bool(all(checks.values()))
    return dict(success=success, checks=checks, measurements=measurements,
                reason='full task stable and released over final dwell' if success else '; '.join(k for k, v in checks.items() if not v))
