"""Trusted target-robot MuJoCo world (NOT given to policy). Generic over robot spec: arbitrary joint names,
per-arm position servos, master gripper joint with robot-internal joint-equality gearing.

Physics after reset: only robot actuator controls + mj_step. Fail-closed validity:
- model contract at load: no mocap; every equality is a JOINT coupling between two robot joints of one arm;
  actuators only on declared robot joints; objects are free bodies with no robot ancestry.
- each step: finite state, no MuJoCo warnings, xfrc_applied/qfrc_applied == 0, |qvel| <= 100,
  arm/finger limit excursions (.005 rad / .0002 m), robot-table/env, robot-self, robot-robot penetration <= 1 mm
  (same tolerances as source twin.practice.runtime.MotorWorld), and object state integrity: object qpos/qvel must
  equal the value left by the previous mj_step (no teleport/write between steps); physical model parameters are
  hashed at load and re-checked at synchronize.
"""
import hashlib, math
import numpy as np
import mujoco
from twin.practice.runtime import NumericalFailure, ObservationError
from .kinematics import TaskKinematics, CanonicalKinematics


class IntegrityViolation(NumericalFailure):
    pass


ARMS = ('left', 'right')


def _model_hash(m):
    h = hashlib.sha256()
    for a in (m.body_mass, m.body_inertia, m.geom_friction, m.geom_size, m.geom_solref, m.geom_solimp, m.geom_contype, m.geom_conaffinity,
        m.geom_pos, m.geom_quat, m.body_pos, m.body_quat, m.opt.gravity, m.eq_active0, m.eq_data, m.actuator_gainprm, m.actuator_biasprm,
        m.actuator_forcerange, m.jnt_range, m.dof_damping, m.body_gravcomp, m.dof_frictionloss, m.dof_armature, m.jnt_axis, m.jnt_solref, m.jnt_solimp,
        m.jnt_stiffness, m.eq_solref, m.eq_solimp, m.eq_obj1id, m.eq_obj2id, m.eq_type, m.geom_condim, m.geom_margin, m.geom_gap, m.body_parentid):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(np.array([m.opt.timestep, m.opt.impratio, m.opt.tolerance, m.opt.iterations, m.opt.cone]).tobytes())
    return h.hexdigest()


class TargetWorld:
    def __init__(self, scene, robot, T_task_base):
        self.model = m = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(m)
        self.robot, self.renderer, self.dt = robot, None, float(m.opt.timestep)
        if m.nmocap:
            raise ValueError('mocap bodies prohibited')
        base = m.body(robot.BASE_BODY).id
        self.robot_bodies, self.arm_of_body = set(), {}
        for b in range(1, m.nbody):
            a = b
            while a and a != base:
                a = int(m.body_parentid[a])
            if a == base:
                self.robot_bodies.add(b)
        self.object_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith('object_')}
        if self.object_bodies & self.robot_bodies or any(b not in self.robot_bodies | self.object_bodies for b in range(1, m.nbody)):
            raise ValueError('every body must be robot (base subtree) or object_*')
        if m.nflex:
            raise ValueError('flex bodies are not supported by this world (cloth is a blocked task)')
        # object hierarchy: an object body is a world child or a child of another object body; its joints may be free (movable object),
        # slide/hinge/ball (articulated fixture part, passive) or none (static fixture). No robot ancestry (checked above).
        ok_types = (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_BALL)
        for b in self.object_bodies:
            if int(m.body_parentid[b]) != 0 and int(m.body_parentid[b]) not in self.object_bodies:
                raise ValueError('object body parent must be world or another object body')
            for k in range(int(m.body_jntnum[b])):
                jt = int(m.jnt_type[int(m.body_jntadr[b]) + k])
                if jt not in ok_types or (jt == mujoco.mjtJoint.mjJNT_FREE and (m.body_jntnum[b] != 1 or m.body_parentid[b] != 0)):
                    raise ValueError('unsupported object joint layout')
        if m.nmocap or m.ntendon and any(m.eq_type[e] == mujoco.mjtEq.mjEQ_TENDON for e in range(m.neq)):
            raise ValueError('mocap / tendon equalities prohibited')
        self.qa, self.da, self.aids, self.ga, self.gaid, ranges, arm_joints = [], [], [], [], [], [], {}
        for arm in ARMS:
            spec = robot.ARMS[arm]
            js = [m.joint(j).id for j in spec['joints'] + [spec['gripper'], *spec['followers']]]
            for j in js:
                arm_joints[j] = arm
            for jn in spec['joints']:
                j = m.joint(jn); self.qa.append(int(j.qposadr[0])); self.da.append(int(j.dofadr[0])); ranges.append(m.jnt_range[j.id].copy())
                self.aids.append(self._actuator(jn))
            g = m.joint(spec['gripper']); self.ga.append(int(g.qposadr[0])); self.gaid.append(self._actuator(spec['gripper']))
            for b in self.robot_bodies:
                nm = m.body(b).name
                if nm.startswith(arm + '_'):
                    self.arm_of_body[b] = arm
        if m.nu != len(self.aids) + len(self.gaid):
            raise ValueError('undeclared actuators')
        # equalities: robot-internal joint couplings (same arm) are the gripper gearing; every other equality must be a preserved source
        # TASK mechanic that references no robot body/joint (object attachment / robot-object welds are forbidden).
        self.robot_eq, self.task_eq = [], []
        for e in range(m.neq):
            t, o1, o2 = int(m.eq_type[e]), int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            if t == mujoco.mjtEq.mjEQ_JOINT:
                j1, j2 = o1 in arm_joints, o2 in arm_joints
                if j1 or j2:
                    if not (j1 and j2 and arm_joints[o1] == arm_joints[o2]):
                        raise ValueError('robot joint equalities must couple joints of one arm')
                    self.robot_eq.append(e); continue
                bodies = {int(m.jnt_bodyid[o1])} | ({int(m.jnt_bodyid[o2])} if o2 >= 0 else set())
            elif t in (mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD):
                bodies = {o1, o2}
            else:
                raise ValueError('unsupported equality type')
            if bodies & self.robot_bodies:
                raise ValueError('equality links a robot body to a task body/world (attachment forbidden)')
            self.task_eq.append(e)
        self.qa, self.da, self.aids, self.ga, self.gaid = (np.array(x, int) for x in (self.qa, self.da, self.aids, self.ga, self.gaid))
        self.ranges = np.array(ranges).reshape(2, 6, 2)
        self.finger_jids = np.array([m.joint(j).id for arm in ARMS for j in (robot.ARMS[arm]['gripper'], *robot.ARMS[arm]['followers'])])
        # finger excursions are measured in master-slide metres: follower joints are divided by their gearing |polycoef[1]|
        gear = {int(m.eq_obj1id[e]): abs(float(m.eq_data[e, 1])) for e in self.robot_eq}
        self.finger_scale = np.array([gear.get(int(j), 1.) for j in self.finger_jids])
        self.arm_jids = np.array([m.joint(j).id for arm in ARMS for j in robot.ARMS[arm]['joints']])
        # every non-robot DOF is integrity-protected: free (7/6), ball (4/3), slide/hinge (1/1); static fixtures have none
        qsz = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}; vsz = {int(mujoco.mjtJoint.mjJNT_FREE): 6, int(mujoco.mjtJoint.mjJNT_BALL): 3}
        oq, ov = [], []
        for j in range(m.njnt):
            if int(m.jnt_bodyid[j]) in self.object_bodies:
                jt = int(m.jnt_type[j]); oq += list(range(int(m.jnt_qposadr[j]), int(m.jnt_qposadr[j]) + qsz.get(jt, 1))); ov += list(range(int(m.jnt_dofadr[j]), int(m.jnt_dofadr[j]) + vsz.get(jt, 1)))
        self.obj_q, self.obj_v = np.array(sorted(oq), int), np.array(sorted(ov), int)
        robot_dofs = {int(m.jnt_dofadr[j]) for j in range(m.njnt) if int(m.jnt_bodyid[j]) in self.robot_bodies}
        if set(self.obj_v.tolist()) & robot_dofs or len(self.obj_v) + len(robot_dofs) != m.nv:
            raise ValueError('every DOF must be a robot DOF or a protected object DOF')
        table = m.geom('table').id
        self.geom_role = []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b in self.robot_bodies:
                self.geom_role.append(('robot', self.arm_of_body.get(b, 'body')))
            elif b in self.object_bodies:
                self.geom_role.append(('object', None))
            else:
                self.geom_role.append(('table' if g == table else 'env', None))
        self.kinematics = TaskKinematics(robot.URDF, T_task_base, robot.ARMS).set_cameras(robot.URDF, robot.CAMERA_LINKS)
        self.model_hash = _model_hash(m)
        self.cam = m.camera(robot.OBS_CAMERA).id
        self.invalid, self.invalid_reason = False, None
        self._reset_evidence()

    def _actuator(self, joint):
        m = self.model; jid = m.joint(joint).id
        ids = [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] == jid]
        if len(ids) != 1:
            raise ValueError(f'expected one actuator on {joint}')
        return ids[0]

    def _reset_evidence(self):
        self.evidence = {'arm_limit_excursion_rad': 0., 'finger_limit_excursion_m': 0.,
            'penetration_m': {k: 0. for k in ('robot_table', 'robot_env', 'robot_self', 'robot_robot', 'robot_object', 'object_table')},
            'witness': {}}

    def fail(self, reason, cls=NumericalFailure):
        self.invalid = True; self.invalid_reason = self.invalid_reason or reason
        raise cls(reason)

    # ---------------------------------------------------------------- state
    def reset(self, poses, home):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        home = np.asarray(home, float).reshape(2, 6)
        if np.any(home < self.ranges[:, :, 0]) or np.any(home > self.ranges[:, :, 1]):
            raise ValueError('home outside joint limits')
        d.qpos[self.qa] = home.ravel()
        for arm in ARMS:  # fingers open: master and gearing consistent with the equality polycoefs
            spec = self.robot.ARMS[arm]; q = self.robot.GRIPPER_OPEN_M
            d.qpos[m.joint(spec['gripper']).qposadr[0]] = q
            for e in self.robot_eq:
                if m.eq_obj2id[e] == m.joint(spec['gripper']).id:
                    d.qpos[m.jnt_qposadr[m.eq_obj1id[e]]] = m.eq_data[e, 0] + m.eq_data[e, 1] * q
        for name, pose in poses.items():
            b = m.body(name).id
            if b not in self.object_bodies or m.body_jntnum[b] != 1 or m.jnt_type[m.body_jntadr[b]] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError('layout may reset only free object bodies (fixtures and slide parts stay as built)')
            adr = m.jnt_qposadr[m.body_jntadr[b]]; d.qpos[adr:adr + 7] = pose
        d.ctrl[self.aids] = home.ravel(); d.ctrl[self.gaid] = self.robot.GRIPPER_OPEN_M
        mujoco.mj_forward(m, d)
        self._obj_state = (d.qpos[self.obj_q].copy(), d.qvel[self.obj_v].copy())  # protect from the first command on
        self.check_state()
        for i, arm in enumerate(ARMS):
            s = m.site(f'{arm}_tcp').id; T = self.kinematics.fk(arm, home[i])
            if not (np.allclose(T[:3, 3], d.site_xpos[s], atol=1e-6) and np.allclose(T[:3, :3], d.site_xmat[s].reshape(3, 3), atol=1e-6)):
                raise ValueError('URDF FK disagrees with MuJoCo TCP site')

    def encoders(self):
        return self.data.qpos[self.qa].reshape(2, 6).copy()

    def gripper_positions(self):
        return self.data.qpos[self.ga].copy()

    def _guard(self):
        d = self.data
        if self._obj_state is not None and not (np.array_equal(d.qpos[self.obj_q], self._obj_state[0]) and np.array_equal(d.qvel[self.obj_v], self._obj_state[1])):
            self.fail('object state written outside physics', IntegrityViolation)
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):
            self.fail('external force detected', IntegrityViolation)
        if np.any(self.model.eq_active0 != 1) or (self.model.neq and np.any(d.eq_active != 1)):
            self.fail('equality activation changed', IntegrityViolation)

    def check_state(self):
        m, d = self.model, self.data
        if not all(np.isfinite(x).all() for x in (d.qpos, d.qvel, d.qacc, d.ctrl)) or not math.isfinite(d.time):
            self.fail('non-finite physics state')
        if any(w.number for w in d.warning):
            self.fail('MuJoCo warning: ' + str([i for i, w in enumerate(d.warning) if w.number]))
        if np.max(np.abs(d.qvel)) > 100:
            self.fail('unbounded state velocity')
        self._guard()
        ev = self.evidence
        for key, jids, tol in (('arm_limit_excursion_rad', self.arm_jids, .005), ('finger_limit_excursion_m', self.finger_jids, .0002)):
            v = d.qpos[m.jnt_qposadr[jids]]; r = m.jnt_range[jids]
            e = np.maximum(0., np.maximum(r[:, 0] - v, v - r[:, 1]))
            x = float(np.max(e / (self.finger_scale if key.startswith('finger') else 1.)))
            ev[key] = max(ev[key], x)
            if x > tol:
                self.fail(f'{key} {x:.5f}')
        for c in d.contact[:d.ncon]:
            (k1, a1), (k2, a2) = self.geom_role[c.geom1], self.geom_role[c.geom2]
            kinds = {k1, k2}
            if kinds == {'robot'}:
                kind = 'robot_self' if a1 == a2 or 'body' in (a1, a2) else 'robot_robot'
            elif 'robot' in kinds:
                kind = 'robot_' + (kinds - {'robot'}).pop()
            elif kinds == {'object', 'table'}:
                kind = 'object_table'
            else:
                continue
            depth = max(0., -float(c.dist))
            if depth > ev['penetration_m'][kind]:
                ev['penetration_m'][kind] = depth
                ev['witness'][kind] = {'t': float(d.time), 'geoms': [int(c.geom1), int(c.geom2)], 'bodies': [m.body(m.geom_bodyid[g]).name for g in (c.geom1, c.geom2)]}
            if kind in ('robot_table', 'robot_env', 'robot_self', 'robot_robot') and depth > .001:
                self.fail(f'{kind} penetration {depth:.4f} m')

    def synchronize(self):
        self._guard()
        mujoco.mj_forward(self.model, self.data)
        if _model_hash(self.model) != self.model_hash:
            self.fail('physical model parameters changed', IntegrityViolation)
        self.check_state()

    def step(self, targets, openings):
        q = np.asarray(targets, float).reshape(2, 6); o = np.asarray(openings, float).reshape(2)
        if not (np.isfinite(q).all() and np.isfinite(o).all()):
            raise ValueError('non-finite command')
        if np.any(q < self.ranges[:, :, 0]) or np.any(q > self.ranges[:, :, 1]):
            raise ValueError('joint target out of range')
        if np.any(o < 0) or np.any(o > 1):
            raise ValueError('finger target out of range')
        self.check_state()
        d = self.data
        d.ctrl[self.aids] = q.ravel(); d.ctrl[self.gaid] = o * self.robot.GRIPPER_OPEN_M
        mujoco.mj_step(self.model, d)
        self._obj_state = (d.qpos[self.obj_q].copy(), d.qvel[self.obj_v].copy())
        mujoco.mj_forward(self.model, d)  # refresh contacts/frames at the encoder time (source synchronize)
        self.check_state()

    # ---------------------------------------------------------------- sensors
    def intrinsics(self, cam):
        m = self.model; c = m.camera(cam).id; w, h = (int(x) for x in m.cam_resolution[c]) if m.cam_resolution[c].min() > 1 else (640, 480)  # free cameras without resolution: 640x480
        f = (h / 2) / math.tan(math.radians(m.cam_fovy[c]) / 2)
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.]]), w, h

    def render(self, cam, depth=True, recoveries=2):
        # Driver-level recovery (post-baseline): a broken EGL context returns all-dark RGB / invalid depth and stays broken.
        # On an invalid frame the renderer is closed and recreated and the SAME physics state is rendered again (no stepping);
        # each recovery is recorded in evidence; a frame that is still invalid raises. Never synthesises pixels or depth.
        for k in range(recoveries + 1):
            try:
                rgb, dep, K, T = self._render_once(cam, depth)
                if int(rgb.max()) > 20:
                    return rgb, dep, K, T
                err = 'all-dark RGB frame'
            except ObservationError as e:
                err = str(e)
            if k == recoveries:
                raise ObservationError(err)
            self.evidence.setdefault('render_recoveries', []).append({'t': float(self.data.time), 'camera': cam, 'reason': err})
            if self.renderer is not None:
                self.renderer.close(); self.renderer = None

    def _render_once(self, cam, depth=True):

        K, w, h = self.intrinsics(cam)
        if self.renderer is None or (self.renderer.width, self.renderer.height) != (w, h):
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        r = self.renderer
        r.disable_depth_rendering(); r.update_scene(self.data, camera=cam); rgb = r.render().copy()
        dep = None
        if depth:
            r.enable_depth_rendering()
            try:
                r.update_scene(self.data, camera=cam); dep = r.render().copy()
            finally:
                r.disable_depth_rendering()
            if dep.shape != (h, w) or not np.isfinite(dep).all() or np.any(dep <= 0):
                raise ObservationError('invalid metric depth frame')
        c = self.model.camera(cam).id
        T = np.eye(4); T[:3, :3] = self.data.cam_xmat[c].reshape(3, 3) @ np.diag([1., -1., -1.]); T[:3, 3] = self.data.cam_xpos[c]
        return rgb, dep, K, T

    def observe(self, camera=None):
        """camera None -> robot.OBS_CAMERA (head, fixed). Wrist cameras (A1, post-baseline): pose from encoder FK
        through the URDF camera chain (what a calibrated real robot reports), checked against the render."""
        self.synchronize()
        cam = camera or self.robot.OBS_CAMERA
        if cam not in self.robot.POLICY_CAMERAS:
            raise ValueError(f'camera {cam} is not a policy camera')
        rgb, dep, K, T = self.render(cam)
        if rgb.dtype != np.uint8:
            raise ObservationError('invalid RGB frame')
        if cam != self.robot.OBS_CAMERA:
            arm = cam.split('_')[0]
            Tk = self.kinematics.camera_fk(arm, self.encoders()[ARMS.index(arm)])
            if not (np.allclose(Tk[:3, 3], T[:3, 3], atol=1e-5) and np.allclose(Tk[:3, :3], T[:3, :3], atol=1e-5)):
                raise ObservationError('camera FK disagrees with render')
            T = Tk
        return dict(rgb=rgb, depth=dep, K=K, T_world_from_camera=T, encoders=self.encoders(), time_sec=float(self.data.time), camera=cam)

    def robot_view(self):
        return RobotView(self)

    def close(self):
        if self.renderer is not None:
            self.renderer.close(); self.renderer = None


class RobotView:
    """Policy-visible robot-only surface (what a calibrated real robot exposes): FK/IK model, encoders, gripper positions, dt, sim time
    and sensor observation. It holds no reference to model/data/objects as attributes; trusted stepping and scoring keep the real world."""
    __slots__ = ('kinematics', 'dt', '_enc', '_gp', '_obs', '_time')

    def __init__(self, world):
        # policy-facing FK/IK use SOURCE CANONICAL TCP axes (approach +z); world.kinematics stays PHYSICAL (reset site check, camera_fk)
        self.kinematics, self.dt = CanonicalKinematics(world.kinematics), world.dt
        self._enc, self._gp, self._obs = world.encoders, world.gripper_positions, world.observe
        self._time = lambda: float(world.data.time)

    def encoders(self):
        return self._enc()

    def gripper_positions(self):
        return self._gp()

    def observe(self, camera=None):
        return self._obs(camera)

    @property
    def time(self):
        return self._time()
