# MuJoCo trusted backend for the G1 suite (r2). Privileged side only; controllers never import this.
# Loads a b1 world + one layout at reset, applies object poses ONCE, then arms NonRobotGuard for every non-robot DOF (free bodies and the drawer slide).
# After arming: no object writes, no xfrc/qfrc, no mocap/weld/attachments; only the 26 drive actuators (ctrl) and mj_step change the world.
# Public side gets PublicBackend closures only (no bound methods, so no __self__ leak). Closures are still introspectable in-process: real isolation is the per-seed parent process.
# Disclosed: fixed pelvis support and robot-only gravcomp (in the world, untouched); this is not a balance or walking claim.
# head_cam RGB-D producer is implemented with the source observe conventions (T = cam_xmat @ diag(1,-1,-1)); it was NOT executed or validated here.
import math, os
import numpy as np
from .robot_adapter import ENCODER_NAMES, check_encoders
from .public_io import PublicBackend
from .trusted_runner import NonRobotGuard

HEAD_CAMERA = 'head_cam'
EXPECTED_NU, EXPECTED_NEQ = 26, 10


class NumericalFailure(RuntimeError):
    pass


class BackendFault(RuntimeError):
    pass


class TrustedHandle:
    # held only by the trusted harness (scorer snapshots, guard, info). Never passed to the controller.
    def __init__(self, mj, model, data, guard, robot_bodies, info, camera_fn):
        self.mj, self.model, self.data, self.guard = mj, model, data, guard
        self.robot_bodies, self.info, self.camera_calibration = robot_bodies, info, camera_fn


def _free_objects(mj, model):
    out = {}
    for b in range(model.nbody):
        name = model.body(b).name
        if name.startswith('object_') and int(model.body_jntnum[b]) == 1 and int(model.jnt_type[int(model.body_jntadr[b])]) == int(mj.mjtJoint.mjJNT_FREE):
            out[name] = int(model.jnt_qposadr[int(model.body_jntadr[b])])
    return out


def build_trusted(scene_xml, layout_poses, width=640, height=480, family=None):
    # layout_poses: {object_name: 7 floats xyz + wxyz} from the DEV layout; returns (PublicBackend, TrustedHandle)
    import mujoco as mj
    from ..scoring import g1_roles
    model = mj.MjModel.from_xml_path(str(scene_xml))
    data = mj.MjData(model)
    if int(model.nu) != EXPECTED_NU or int(model.neq) != EXPECTED_NEQ or int(model.nmocap) != 0:
        raise BackendFault('unexpected_world:nu=%d neq=%d nmocap=%d' % (model.nu, model.neq, model.nmocap))
    dt = float(model.opt.timestep)
    if abs(round(0.05 / dt) * dt - 0.05) > 1e-9:
        raise BackendFault('timestep_not_integral_in_0.05s')
    act, qadr = {}, {}
    for n in ENCODER_NAMES:
        a = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, 'a_' + n)
        j = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)
        if a < 0 or j < 0 or int(model.actuator_trnid[a, 0]) != j:
            raise BackendFault('drive_actuator_joint_mismatch:' + n)
        act[n], qadr[n] = a, int(model.jnt_qposadr[j])
    rl = g1_roles.roles(model)  # fail-closed classification of every body
    robot_bodies = set(int(b) for b in rl['robot'])
    free = _free_objects(mj, model)
    if set(layout_poses) != set(free):
        raise BackendFault('layout_objects_mismatch:%s vs %s' % (sorted(layout_poses), sorted(free)))
    mj.mj_resetData(model, data)
    for name, pose in layout_poses.items():
        p = np.asarray(pose, float)
        if p.shape != (7,) or not np.all(np.isfinite(p)) or abs(np.linalg.norm(p[3:]) - 1.0) > 1e-6:
            raise BackendFault('bad_layout_pose:' + name)
        data.qpos[free[name]:free[name] + 7] = p
    mj.mj_forward(model, data)
    for n in ENCODER_NAMES:  # servos hold the reset posture until the controller commands otherwise
        data.ctrl[act[n]] = data.qpos[qadr[n]]
    guard = NonRobotGuard(model, data, robot_bodies, EXPECTED_NEQ, EXPECTED_NU)
    guard.arm()  # object init is finished: every non-robot qpos/qvel (incl. drawer slide) is frozen against writes
    cid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, HEAD_CAMERA)
    if cid < 0:
        raise BackendFault('head_cam_missing')
    fovy = math.radians(float(model.cam_fovy[cid]))
    fy = (height / 2.0) / math.tan(fovy / 2.0)
    K = np.array([[fy, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1.0]])
    state = {'renderer': None}
    ncon0 = int(data.ncon)
    robot_contacts0 = sum(1 for c in data.contact[:ncon0] if (int(model.geom_bodyid[c.geom1]) in robot_bodies) != (int(model.geom_bodyid[c.geom2]) in robot_bodies) or (int(model.geom_bodyid[c.geom1]) in robot_bodies and int(model.geom_bodyid[c.geom2]) in robot_bodies and c.dist < 0))
    info = {'scene': str(scene_xml), 'dt': dt, 'n_robot_bodies': len(robot_bodies), 'free_objects': sorted(free), 'K_nominal_from_model': K.tolist(), 'resolution': [width, height],
            'robot_related_contacts_at_reset': robot_contacts0, 'camera_convention': 'T_world_from_camera = cam_xmat @ diag(1,-1,-1), t = cam_xpos (source MotorWorld.observe)'}

    def time_fn():
        return float(data.time)

    def encoders_fn():
        v = {n: float(data.qpos[qadr[n]]) for n in ENCODER_NAMES}
        if not all(math.isfinite(x) for x in v.values()):
            raise NumericalFailure('nonfinite encoder')
        return v

    def camera_fn():
        mj.mj_forward(model, data)
        T = np.eye(4)
        T[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = data.cam_xpos[cid]
        return K.copy(), T

    def step_fn(targets):
        t = check_encoders(targets)
        for n, v in t.items():
            a = act[n]
            if int(model.actuator_ctrllimited[a]):
                lo, hi = model.actuator_ctrlrange[a]
                if v < lo - 1e-9 or v > hi + 1e-9:
                    raise BackendFault('target_out_of_ctrlrange:' + n)
                v = min(max(v, lo), hi)
            data.ctrl[a] = v
        mj.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)) and np.all(np.isfinite(data.qacc))):
            raise NumericalFailure('non-finite physics state')

    def observe_fn():
        if os.environ.get('MUJOCO_GL', '') in ('', 'disable'):
            raise BackendFault('rendering_disabled')
        Kc, T = camera_fn()
        if state['renderer'] is None:
            state['renderer'] = mj.Renderer(model, height=height, width=width)
        r = state['renderer']
        r.disable_depth_rendering()
        r.update_scene(data, camera=HEAD_CAMERA)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        try:
            r.update_scene(data, camera=HEAD_CAMERA)
            depth = r.render().copy()
        finally:
            r.disable_depth_rendering()
        if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8 or depth.shape != (height, width) or not np.isfinite(depth).all() or np.any(depth <= 0):
            raise BackendFault('invalid_observation_frame')
        return dict(rgb=rgb, depth=depth, K=Kc, T_world_from_camera=T, encoders=encoders_fn(), time_sec=time_fn())

    from .trusted_runner import make_public_backend
    monitor, step_guard = None, guard
    if family == 'sweep':  # r3: sweep Monitor created after reset/arm and placed in the step path through CompositeGuard
        from ..scoring import g1_scorer as S
        from .scoring_wire import CompositeGuard
        monitor = S.Monitor(model, data, rl)
        step_guard = CompositeGuard(guard, monitor, model, data)
    cb = make_public_backend(dt, time_fn, encoders_fn, observe_fn, step_fn, step_guard)
    h = TrustedHandle(mj, model, data, guard, robot_bodies, info, camera_fn)
    h.monitor = monitor
    return cb, h
