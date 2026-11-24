"""Offline, no-step tabletop layouts. Simulator truth is reset/scorer-only.

CLI: python -m twin.practice.layouts TASK OUTPUT (OUTPUT must not exist).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
TASK_OBJECTS = {
    'collect_apple': ('apple', 'bowl'),
    'collect_cubes': ('bowl', 'yellow_cube', 'red_cube_left', 'blue_cube', 'orange_cube', 'red_cube_right'),
    'stack_cups': ('yellow_cup', 'purple_cup', 'blue_cup'),
}
OFFSETS = dict(collect_apple=10000, collect_cubes=20000, stack_cups=30000)
SPLITS = {'DEV': ((101, .4), (102, 1.), (103, 1.6)),
          'dev3': ((201, .4), (202, 1.), (203, 1.6)),
          'VAL': ((301, .4), (302, 1.), (303, 1.6)),
          'TEST': ((901, .4), (902, .4), (903, 1.), (904, 1.), (905, 1.6), (906, 1.6))}


def body_vertices(model, name):
    """Rigid body-local convex envelope of MJCF visual/collision geometry.

    Mesh hull includes all mesh extrema. Primitive boxes conservatively enclose
    curved primitives; shipped task objects are meshes/boxes. No dynamics.
    """
    bid = model.body(name).id
    points = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid:
            continue
        kind = model.geom_type[gid]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            p = np.asarray(model.mesh_vert[start:start + count], dtype=float)
        else:
            size = model.geom_size[gid].copy()
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                size[:] = size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                size = np.array([size[0], size[0], size[1]])
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                size = np.array([size[0], size[0], size[0] + size[1]])
            elif kind not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                raise ValueError(f'unsupported object geom type {kind}')
            p = np.array([[x, y, z] for x in (-size[0], size[0])
                          for y in (-size[1], size[1]) for z in (-size[2], size[2])])
        rotation = Rotation.from_quat(np.roll(model.geom_quat[gid], -1)).as_matrix()
        points.append(p @ rotation.T + model.geom_pos[gid])
    if not points:
        raise ValueError(f'{name}: missing geometry')
    p = np.unique(np.concatenate(points), axis=0)
    return p[ConvexHull(p).vertices]


def world_vertices(model, data, name, vertices=None):
    p = body_vertices(model, name) if vertices is None else vertices
    body = data.body(name)
    return p @ body.xmat.reshape(3, 3).T + body.xpos


def _disjoint(a, b, margin=.002):
    """Separating axes for convex support footprints, with 2 mm clearance."""
    a, b = a[ConvexHull(a).vertices], b[ConvexHull(b).vertices]
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for e in edges:
            axis = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            aa, bb = a @ axis, b @ axis
            if aa.max() + margin <= bb.min() or bb.max() + margin <= aa.min():
                return True
    return False


def _point_footprint_distance(point, points):
    polygon = points[ConvexHull(points).vertices]
    hull = ConvexHull(polygon)
    if np.all(hull.equations[:, :2] @ point + hull.equations[:, 2] <= 0):
        return 0.
    end = np.roll(polygon, -1, axis=0)
    edge = end - polygon
    ratio = np.clip(np.sum((point - polygon) * edge, axis=1) / np.sum(edge * edge, axis=1), 0, 1)
    return float(np.min(np.linalg.norm(polygon + ratio[:, None] * edge - point, axis=1)))


def _visible(model, data, bid, points):
    camera = model.camera('main').id
    origin = data.cam_xpos[camera]
    rot = data.cam_xmat[camera].reshape(3, 3)
    # MJCF camera intrinsic fields are metric sensor coordinates.
    resolution = model.cam_resolution[camera]
    sensor = model.cam_sensorsize[camera]
    intrinsic = model.cam_intrinsic[camera]
    p = points[np.linspace(0, len(points) - 1, min(96, len(points)), dtype=int)]
    local = (p - origin) @ rot
    depth = -local[:, 2]
    uv = local[:, :2] / np.maximum(depth[:, None], 1e-12) * intrinsic[:2] - intrinsic[2:]
    in_frame = (depth > .02) & np.all(np.abs(uv) < sensor / 2 * .98, axis=1)
    visible = 0
    for target, valid in zip(p, in_frame):
        if not valid:
            continue
        direction = target - origin
        direction /= np.linalg.norm(direction)
        geom = np.array([-1], dtype=np.int32)
        mujoco.mj_ray(model, data, origin, direction, np.ones(6, dtype=np.uint8), 1, -1, geom)
        if geom[0] >= 0 and model.geom_bodyid[geom[0]] == bid:
            visible += 1
    return dict(samples=len(p), visible=visible, fraction=visible / len(p), resolution=resolution.tolist())


def _workspace_reach(model, data, target):
    """Position-only robot kinematic reach with bounded scalar joint limits.

    Scratch mj_kinematics/mj_comPos/Jacobians inspect robot chain only. No
    contact/dynamics or policy outcome. This is not a collision-free motion or
    grasp-orientation certificate; it verifies a point 60 mm above the top.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    evidence = {}
    for arm in ('left', 'right'):
        joints = [model.joint(f'{arm}_j{i}').id for i in range(1, 7)]
        qa = model.jnt_qposadr[joints]
        va = model.jnt_dofadr[joints]
        limits = model.jnt_range[joints]
        site = model.site(arm + '_tcp').id
        initial = np.clip(data.qpos[qa], limits[:, 0] + 1e-8, limits[:, 1] - 1e-8)
        best = float('inf')
        evaluations = 0
        def residual(q):
            scratch.qpos[qa] = q
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            return scratch.site_xpos[site] - target
        def jacobian(q):
            residual(q)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, scratch, jp, jr, site)
            return jp[:, va]
        for start in (initial, np.clip(np.zeros(6), limits[:, 0] + 1e-8, limits[:, 1] - 1e-8)):
            fit = least_squares(residual, start, jac=jacobian,
                                bounds=(limits[:, 0], limits[:, 1]), max_nfev=60,
                                ftol=1e-7, xtol=1e-7, gtol=1e-7)
            best = min(best, float(np.linalg.norm(fit.fun)))
            evaluations += fit.nfev
            if best <= .005:
                break
        evidence[arm] = dict(residual_m=best, evaluations=evaluations, reachable=best <= .005)
    return evidence


def validate_layout(model, data, record, task, vertices=None):
    """Pure geometry checks after mj_forward; never calls mj_step."""
    names = TASK_OBJECTS[task]
    table = model.geom('table').id
    basis = data.geom_xmat[table].reshape(3, 3)
    center = data.geom_xpos[table] + basis[:, 2] * model.geom_size[table, 2]
    reasons, evidence, footprints = [], {}, {}
    for name in names:
        body = 'object_' + name
        p = world_vertices(model, data, body, None if vertices is None else vertices[name])
        local = (p - center) @ basis
        footprints[name] = local[:, :2]
        gap = float(local[:, 2].min())
        if abs(gap - .0002) > .0001:
            reasons.append(f'{name}:unsupported_table')
        if np.any(np.abs(local[:, :2]) > model.geom_size[table, :2] - .003):
            reasons.append(f'{name}:outside_table')
        reaches = {}
        for arm in ('left', 'right'):
            shoulder = data.body(arm + '_base').xpos + data.body(arm + '_base').xmat.reshape(3, 3)[:, 2] * .2234
            distance = float(np.linalg.norm(data.body(body).xpos - shoulder))
            reaches[arm] = distance
        if not any(.12 <= d <= .75 for d in reaches.values()):
            reasons.append(f'{name}:outside_reach_envelopes')
        for arm in ('left', 'right'):
            if _point_footprint_distance(data.body(arm + '_base').xpos[:2], p[:, :2]) < .10:
                reasons.append(f'{name}:robot_base_clearance')
        visibility = _visible(model, data, model.body(body).id, p)
        if visibility['visible'] < 3 or visibility['fraction'] < .10:
            reasons.append(f'{name}:not_visible')
        target = data.body(body).xpos.copy()
        target += basis[:, 2] * (.06 + local[:, 2].max() - float((target - center) @ basis[:, 2]))
        kinematic_reach = _workspace_reach(model, data, target)
        if not any(v['reachable'] for v in kinematic_reach.values()):
            reasons.append(f'{name}:no_kinematic_workspace_reach')
        evidence[name] = dict(table_gap_m=gap, reach_shoulder_distance_m=reaches,
                              reach_target_world_m=target.tolist(), kinematic_reach=kinematic_reach, visibility=visibility)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if not _disjoint(footprints[name], footprints[other]):
                reasons.append(f'{name}/{other}:overlapping_support_footprints')
    # Catch arm/table penetration as well as footprint separation. mj_forward
    # computes contact candidates without integrating the task.
    object_ids = {model.body('object_' + n).id for n in names}
    for contact in data.contact:
        a, b = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        if contact.dist < -.0005 and ((a in object_ids) != (b in object_ids)):
            reasons.append(f'penetration:{model.geom(contact.geom1).name}/{model.geom(contact.geom2).name}')
    return dict(valid=not reasons, reasons=reasons, objects=evidence,
                reach_check='shoulder annulus and bounded position IK (5 mm residual); not grasp orientation/path certificate')


def make_layouts(task: str, output: Path) -> dict:
    """Write audit.jsonl and manifest.json into a fresh directory.

    On exhaustion audit remains and failure.json records the missing layout;
    no incomplete success manifest is emitted. All rejected draws include
    exact perturbations, final rigid poses, and every validity reason.
    """
    names = TASK_OBJECTS[task]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene = ROOT / 'runs/camera_reference_20261025/review_v2/scenes' / task / 'scene.xml'
    record_path = ROOT / 'runs/physics_v2' / task / 'reconstruction.json'
    reference_path = ROOT / 'runs/physics_v2' / task / 'reference.npz'
    record = json.loads(record_path.read_text())
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    # Match runtime reset: first measured encoder row only, open fingers.
    with np.load(reference_path, allow_pickle=False) as archive:
        first = np.array(archive['joints'][0], dtype=float)
    if first.shape != (2, 6) or not np.isfinite(first).all():
        raise ValueError('invalid first encoder row')
    for a, arm in enumerate(('left', 'right')):
        for j in range(1, 7):
            joint = model.joint(f'{arm}_j{j}')
            if not model.jnt_range[joint.id, 0] <= first[a, j - 1] <= model.jnt_range[joint.id, 1]:
                raise ValueError('initial encoder outside joint limits')
            data.qpos[joint.qposadr[0]] = first[a, j - 1]
        for j in (1, 2):
            data.qpos[model.joint(f'{arm}_gripper{j}').qposadr[0]] = 0
    reset_qpos = data.qpos.copy()
    mujoco.mj_forward(model, data)
    table = model.geom('table').id
    basis = data.geom_xmat[table].reshape(3, 3).copy()
    top = data.geom_xpos[table] + basis[:, 2] * model.geom_size[table, 2]
    original = {n: (data.body('object_' + n).xpos.copy(), data.body('object_' + n).xmat.reshape(3, 3).copy()) for n in names}
    vertices = {n: body_vertices(model, 'object_' + n) for n in names}
    for n in names:
        body = model.body('object_' + n).id
        if model.body_jntnum[body] != 1 or model.jnt_type[model.body_jntadr[body]] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f'{n} must have a free joint')
    manifest = dict(schema_version=1, task=task, rng='numpy.PCG64', layouts=[],
                    inputs_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (scene, record_path, reference_path)},
                    initial_encoders=first.tolist(), table_basis=basis.tolist(), max_attempts=200, task_dynamics_executed=False)
    with (output / 'audit.jsonl').open('x') as audit:
        for split, entries in SPLITS.items():
            for index, scale in entries:
                seed = OFFSETS[task] + index
                rng = np.random.Generator(np.random.PCG64(seed))
                for attempt in range(1, 201):
                    poses, draws = {}, {}
                    mujoco.mj_resetData(model, data)
                    data.qpos[:] = reset_qpos
                    for n in names:
                        extent = (.04 if n == 'bowl' else .05) * scale
                        dx, dy = rng.uniform(-extent, extent, size=2)
                        angle = rng.uniform(-np.pi, np.pi) if n == 'bowl' else rng.uniform(-np.pi / 6 * scale, np.pi / 6 * scale)
                        p0, r0 = original[n]
                        r = Rotation.from_rotvec(basis[:, 2] * angle).as_matrix() @ r0
                        p = p0 + basis[:, :2] @ np.array([dx, dy])
                        # Support lowest rigid-geometry point on the tilted top.
                        lowest = np.min((vertices[n] @ r.T + p - top) @ basis[:, 2])
                        p += basis[:, 2] * (.0002 - lowest)
                        quat = np.roll(Rotation.from_matrix(r).as_quat(), 1)
                        pose = np.r_[p, quat]
                        joint = model.body('object_' + n).jntadr[0]
                        adr = model.jnt_qposadr[joint]
                        data.qpos[adr:adr + 7] = pose
                        poses['object_' + n] = pose.tolist()
                        draws[n] = dict(translation_table_m=[float(dx), float(dy)], yaw_rad=float(angle))
                    mujoco.mj_forward(model, data)
                    validity = validate_layout(model, data, record, task, vertices)
                    proposal = dict(id=f'{split}_{index}', split=split, seed=seed, scale=scale,
                                    attempt=attempt, draws=draws, poses=poses, validity=validity)
                    audit.write(json.dumps(proposal, allow_nan=False) + '\n')
                    audit.flush()
                    if validity['valid']:
                        manifest['layouts'].append(proposal)
                        break
                else:
                    failure = dict(task=task, id=f'{split}_{index}', seed=seed, reason='200 rejected proposals; ranges unchanged')
                    (output / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
                    raise RuntimeError(str(failure))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=tuple(TASK_OBJECTS))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    make_layouts(args.task, args.output)
