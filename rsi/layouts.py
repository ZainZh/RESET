"""Deterministic layout sampling around the demonstration's first-frame poses.

A layout is (task, split, index) -> {object_<name>: [x y z qw qx qy qz]} for free bodies.
Objects shift in the world XY plane and rotate about world Z; samples that
interpenetrate other objects or the robot, or leave the table, are rejected.
Splits: dev (seed 0) free use; val (seed 1) adoption only; test (seed 2) sealed.
"""
import os
import zlib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .tasks import get

SPLITS = {'demo': None, 'dev': 0, 'val': 1, 'test': 2}
TEST_ENV = 'RSI_UNSEAL_TEST'


def _free_adr(model, body):
    jid = int(model.body_jntadr[model.body(body).id])
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f'{body} is not a free body')
    return int(model.jnt_qposadr[jid])


def demo_poses(model):
    data = mujoco.MjData(model)
    out = {}
    for b in range(model.nbody):
        name = model.body(b).name
        if name.startswith('object_') and model.body_jntnum[b] == 1 and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE:
            adr = _free_adr(model, name)
            out[name] = data.qpos[adr:adr + 7].copy()
    return out


def _valid(model, data, record, names, initial_joints):
    from twin.oneshot_scene import set_bimanual
    set_bimanual(model, data, initial_joints, np.ones(2))
    mujoco.mj_forward(model, data)
    objects = {model.body('object_' + n).id for n in names}
    task_bodies = {b for b in range(model.nbody) if model.body(b).name.startswith('object_')}
    for c in data.contact:
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if not ({b1, b2} & objects):
            continue
        other = b2 if b1 in task_bodies else b1
        if model.geom(c.geom1).name == 'table' or model.geom(c.geom2).name == 'table':
            continue
        if c.dist < -.001:
            return False
    table = record['table']
    R = np.array(table['rotation']); center = np.array(table['center']); half = np.array(table['half_size'])
    for n in names:
        local = R.T @ (data.body('object_' + n).xpos - center)
        if np.any(np.abs(local[:2]) > half[:2] - .03):
            return False
    return True


def sample(task_name, split, index, model=None, record=None):
    task = get(task_name)
    if split == 'test' and os.environ.get(TEST_ENV) != '1':
        raise PermissionError('TEST split is sealed')
    model = model or mujoco.MjModel.from_xml_path(str(task.world / 'scene.xml'))
    record = record or task.record
    base = demo_poses(model)
    if split == 'demo':
        return {k: v.tolist() for k, v in base.items()}, 0
    initial = np.load(task.world / 'reference.npz')['joints'][0]
    rng = np.random.default_rng([SPLITS[split], index, zlib.crc32(task_name.encode())])
    data = mujoco.MjData(model)
    for attempt in range(500):
        mujoco.mj_resetData(model, data)
        poses = {}
        for name, pose in base.items():
            p, q = pose[:3].copy(), pose[3:].copy()
            key = name[len('object_'):]
            if key in task.movable:
                dx, dy, dyaw = task.movable[key]
                p[:2] += rng.uniform([-dx, -dy], [dx, dy])
                yaw = np.radians(rng.uniform(-dyaw, dyaw)) if dyaw else 0.
                R = Rotation.from_euler('z', yaw) * Rotation.from_quat(np.roll(q, -1))
                q = np.roll(R.as_quat(), 1)
            poses[name] = np.r_[p, q]
            adr = _free_adr(model, name)
            data.qpos[adr:adr + 7] = poses[name]
        if _valid(model, data, record, list(task.movable), initial):
            return {k: v.tolist() for k, v in poses.items()}, attempt + 1
    raise RuntimeError(f'no valid layout for {task_name} {split} {index}')
