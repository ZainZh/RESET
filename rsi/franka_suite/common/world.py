"""Trusted Franka MuJoCo world (NOT given to policy). Same contract as the LIFT2 TargetWorld
(rsi/lift2_suite/common/embodiment/world.py), adapted to 7-joint arms, one finger master + one mimic follower per hand and
ONE fixed policy camera (main_cam on the robot body).

Physics after reset: only robot actuator controls + mj_step. Fail-closed validity, every step:
- finite state, no MuJoCo warnings, xfrc_applied/qfrc_applied == 0, |qvel| <= 100;
- arm/finger limit excursions (.005 rad / .0002 m);
- robot-table/env (desk borders), robot-robot (cross-arm) and robot-self (any contact the model still generates between robot
  geoms) penetration <= 1 mm (same tolerances as the source MotorWorld / LIFT2 TargetWorld);
- object state integrity (object qpos/qvel equal the value left by the previous mj_step) and a hash of physical model parameters.
Evidence: deepest robot-object, object-table contact.
"""
import hashlib, math
import numpy as np
import mujoco
from twin.practice.runtime import NumericalFailure, ObservationError
from .kinematics import FrankaKinematics
from . import franka as R

ARMS = ('left', 'right')
NJ = R.NJ


class IntegrityViolation(NumericalFailure):
    pass


def _model_hash(m):
    h = hashlib.sha256()
    for a in (m.body_mass, m.body_inertia, m.geom_friction, m.geom_size, m.geom_solref, m.geom_solimp, m.geom_contype, m.geom_conaffinity,
        m.geom_pos, m.geom_quat, m.body_pos, m.body_quat, m.opt.gravity, m.eq_active0, m.eq_data, m.actuator_gainprm, m.actuator_biasprm,
        m.actuator_forcerange, m.jnt_range, m.dof_damping, m.body_gravcomp, m.dof_frictionloss, m.dof_armature, m.jnt_axis, m.jnt_solref, m.jnt_solimp,
        m.jnt_stiffness, m.eq_solref, m.eq_solimp, m.eq_obj1id, m.eq_obj2id, m.eq_type, m.geom_condim, m.geom_margin, m.geom_gap, m.body_parentid):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(np.array([m.opt.timestep, m.opt.impratio, m.opt.tolerance, m.opt.iterations, m.opt.cone]).tobytes())
    return h.hexdigest()


class FrankaWorld:
    def __init__(self, scene, T_task_base):
        self.model = m = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(m)
        self.renderer, self.dt = None, float(m.opt.timestep)
        if m.nmocap or m.nflex:
            raise ValueError('mocap / flex prohibited')
        base = m.body(R.BASE_BODY).id
        self.robot_bodies = set()
        for b in range(1, m.nbody):
            a = b
            while a and a != base:
                a = int(m.body_parentid[a])
            if a == base:
                self.robot_bodies.add(b)
        self.object_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith('object_')}
        if self.object_bodies & self.robot_bodies or any(b not in self.robot_bodies | self.object_bodies for b in range(1, m.nbody)):
            raise ValueError('every body must be robot (base subtree) or object_*')
        ok_types = (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_BALL)
        for b in self.object_bodies:
            if int(m.body_parentid[b]) != 0 and int(m.body_parentid[b]) not in self.object_bodies:
                raise ValueError('object body parent must be world or another object body')
            for k in range(int(m.body_jntnum[b])):
                jt = int(m.jnt_type[int(m.body_jntadr[b]) + k])
                if jt not in ok_types or (jt == mujoco.mjtJoint.mjJNT_FREE and (m.body_jntnum[b] != 1 or m.body_parentid[b] != 0)):
                    raise ValueError('unsupported object joint layout')
        self.arm_of_body = {}
        for b in self.robot_bodies:
            nm = m.body(b).name
            for arm in ARMS:
                if nm.startswith(arm + '_'): self.arm_of_body[b] = arm
        self.qa, self.aids, self.ga, self.gaid, arm_joints = [], [], [], [], {}
        for arm in ARMS:
            spec = R.ARMS[arm]
            for jn in spec['joints'] + [spec['gripper'], *spec['followers']]:
                arm_joints[m.joint(jn).id] = arm
            for jn in spec['joints']:
                self.qa.append(int(m.joint(jn).qposadr[0])); self.aids.append(self._actuator(jn))
            self.ga.append(int(m.joint(spec['gripper']).qposadr[0])); self.gaid.append(self._actuator(spec['gripper']))
        if m.nu != len(self.aids) + len(self.gaid):
            raise ValueError('undeclared actuators')
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
        self.qa, self.aids, self.ga, self.gaid = (np.array(x, int) for x in (self.qa, self.aids, self.ga, self.gaid))
        self.ranges = np.array([m.jnt_range[m.joint(j).id] for arm in ARMS for j in R.ARMS[arm]['joints']]).reshape(2, NJ, 2)
        self.finger_jids = np.array([m.joint(j).id for arm in ARMS for j in (R.ARMS[arm]['gripper'], *R.ARMS[arm]['followers'])])
        self.arm_jids = np.array([m.joint(j).id for arm in ARMS for j in R.ARMS[arm]['joints']])
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
        self.kinematics = FrankaKinematics.from_robot_xml(T_task_base)
        self.model_hash = _model_hash(m)
        self.cam = m.camera(R.OBS_CAMERA).id
        self.invalid, self.invalid_reason = False, None
        self._obj_state = None
        self._reset_evidence()

    def _actuator(self, joint):
        m = self.model; jid = m.joint(joint).id
        ids = [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] == jid]
        if len(ids) != 1:
            raise ValueError(f'expected one actuator on {joint}')
        return ids[0]

    def _reset_evidence(self):
        self.evidence = {'arm_limit_excursion_rad': 0., 'finger_limit_excursion_m': 0.,
            'penetration_m': {k: 0. for k in ('robot_table', 'robot_env', 'robot_self', 'robot_robot', 'robot_object', 'object_table', 'object_env')},
            'witness': {}}

    def fail(self, reason, cls=NumericalFailure):
        self.invalid = True; self.invalid_reason = self.invalid_reason or reason
        raise cls(reason)

    def reset(self, poses, home):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        home = np.asarray(home, float).reshape(2, NJ)
        if np.any(home < self.ranges[:, :, 0]) or np.any(home > self.ranges[:, :, 1]):
            raise ValueError('home outside joint limits')
        d.qpos[self.qa] = home.ravel()
        for arm in ARMS:
            spec = R.ARMS[arm]; q = R.GRIPPER_OPEN_M  # open at reset
            d.qpos[m.joint(spec['gripper']).qposadr[0]] = q
            for e in self.robot_eq:
                if m.eq_obj2id[e] == m.joint(spec['gripper']).id:
                    d.qpos[m.jnt_qposadr[m.eq_obj1id[e]]] = m.eq_data[e, 0] + m.eq_data[e, 1] * q
        for name, pose in poses.items():
            b = m.body(name).id
            if b not in self.object_bodies or m.body_jntnum[b] != 1 or m.jnt_type[m.body_jntadr[b]] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError('layout may reset only free object bodies')
            adr = m.jnt_qposadr[m.body_jntadr[b]]; d.qpos[adr:adr + 7] = pose
        d.ctrl[self.aids] = home.ravel(); d.ctrl[self.gaid] = R.GRIPPER_OPEN_M
        mujoco.mj_forward(m, d)
        self._obj_state = (d.qpos[self.obj_q].copy(), d.qvel[self.obj_v].copy())
        self.check_state()
        for i, arm in enumerate(ARMS):
            s = m.site(f'{arm}_tcp').id; T = self.kinematics.fk(arm, home[i])
            if not (np.allclose(T[:3, 3], d.site_xpos[s], atol=1e-6) and np.allclose(T[:3, :3], d.site_xmat[s].reshape(3, 3), atol=1e-6)):
                raise ValueError('numeric FK disagrees with MuJoCo TCP site')

    def encoders(self):
        return self.data.qpos[self.qa].reshape(2, NJ).copy()

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

    def contact_depths(self):
        '''Deepest contact per role pair at the current state (no failure side effects).'''
        m, d = self.model, self.data; out = {}
        for c in d.contact[:d.ncon]:
            kind = self._kind(c)
            if kind:
                out[kind] = max(out.get(kind, 0.), max(0., -float(c.dist)))
        return out

    def _kind(self, c):
        (k1, a1), (k2, a2) = self.geom_role[c.geom1], self.geom_role[c.geom2]
        kinds = {k1, k2}
        if kinds == {'robot'}:
            return 'robot_self' if a1 == a2 or 'body' in (a1, a2) else 'robot_robot'
        if 'robot' in kinds:
            return 'robot_' + (kinds - {'robot'}).pop()
        if kinds == {'object', 'table'}:
            return 'object_table'
        if kinds == {'object', 'env'}:
            return 'object_env'
        return None

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
            x = float(np.max(np.maximum(0., np.maximum(r[:, 0] - v, v - r[:, 1]))))
            ev[key] = max(ev[key], x)
            if x > tol:
                self.fail(f'{key} {x:.5f}')
        for c in d.contact[:d.ncon]:
            kind = self._kind(c)
            if not kind:
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
        q = np.asarray(targets, float).reshape(2, NJ); o = np.asarray(openings, float).reshape(2)
        if not (np.isfinite(q).all() and np.isfinite(o).all()):
            raise ValueError('non-finite command')
        if np.any(q < self.ranges[:, :, 0]) or np.any(q > self.ranges[:, :, 1]):
            raise ValueError('joint target out of range')
        if np.any(o < 0) or np.any(o > 1):
            raise ValueError('finger target out of range')
        self.check_state()
        d = self.data
        d.ctrl[self.aids] = q.ravel(); d.ctrl[self.gaid] = R.GRIPPER_CLOSED_M + o * (R.GRIPPER_OPEN_M - R.GRIPPER_CLOSED_M)
        mujoco.mj_step(self.model, d)
        self._obj_state = (d.qpos[self.obj_q].copy(), d.qvel[self.obj_v].copy())
        mujoco.mj_forward(self.model, d)
        self.check_state()

    # ---------------------------------------------------------------- sensors
    def intrinsics(self, cam):
        m = self.model; c = m.camera(cam).id; w, h = (int(x) for x in m.cam_resolution[c])
        f = (h / 2) / math.tan(math.radians(m.cam_fovy[c]) / 2)
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.]]), w, h

    def render(self, cam, depth=True, recoveries=2):
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
        self.synchronize()
        cam = camera or R.OBS_CAMERA
        if cam not in R.POLICY_CAMERAS:
            raise ValueError(f'camera {cam} is not a policy camera')
        rgb, dep, K, T = self.render(cam)
        return dict(rgb=rgb, depth=dep, K=K, T_world_from_camera=T, encoders=self.encoders(), time_sec=float(self.data.time), camera=cam)

    def close(self):
        if self.renderer is not None:
            self.renderer.close(); self.renderer = None


class RobotView:
    """Policy-visible robot-only surface: FK/IK model (canonical TCP axes), encoders, gripper positions, dt, sim time, observe."""
    __slots__ = ('kinematics', 'dt', '_enc', '_gp', '_obs', '_time')

    def __init__(self, world):
        self.kinematics, self.dt = world.kinematics, world.dt
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
