"""Trusted force-only MuJoCo runtime, with a small public sensor/robot facade.

The episode subprocess owns physics and perception. The parent reserves the
attempt and enforces the 300-second wall cap, including rendering/perception.
No task dynamics occur during import, preparation, or input freeze.
"""
from contextlib import contextmanager
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from .engine import (ContractError, SIM_CAPS, budget_root, json_bytes, load_inputs,
                     read_json, require, reserve_attempt, verify_candidate, write_new)
from .motion import KinematicsError, RobotKinematics, numeric, rotation, waypoint


class ObservationError(RuntimeError):
    pass


class NumericalFailure(RuntimeError):
    pass


class EpisodeLimit(RuntimeError):
    pass


class MotionError(RuntimeError):
    pass


def servo_config(record):
    """Extract only trusted servo scalars; never expose the record to policy."""
    keys = ('finger_velocity_damping_Ns_m', 'opening_velocity_limit_m_s',
            'opening_stop_buffer_m', 'finger_force_limit_N')
    require(all(k in record for k in keys), 'reconstruction lacks validated finger servo settings')
    config = {k: float(record[k]) for k in keys}
    require(all(math.isfinite(v) for v in config.values()), 'non-finite finger servo settings')
    require(config[keys[0]] >= 0 and config[keys[1]] > 0 and config[keys[2]] >= 0
            and 0 < config[keys[3]] <= 8, 'invalid finger servo settings')
    return config


class MotorWorld:
    """Trusted world; never supplied to policy or retained by robot kinematics."""
    def __init__(self, scene, servo):
        import mujoco
        self.mj = mujoco
        self.model = m = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(m)
        self.renderer = None
        self.servo = servo_config(servo)
        require(m.nmocap == 0 and m.neq == 0, 'mocap/equality constraints prohibited')
        qa, da, aids, fqa, fda, faids, ranges = [], [], [], [], [], [], []
        for arm in ('left', 'right'):
            for j in range(1, 7):
                joint, act = m.joint(f'{arm}_j{j}'), m.actuator(f'{arm}_motor{j}')
                self._check_motor(joint, act, 100.)
                qa.append(int(joint.qposadr[0])); da.append(int(joint.dofadr[0]))
                aids.append(act.id); ranges.append(m.jnt_range[joint.id].copy())
            for j in (1, 2):
                joint, act = m.joint(f'{arm}_gripper{j}'), m.actuator(f'{arm}_gripper_motor{j}')
                self._check_motor(joint, act, self.servo['finger_force_limit_N'])
                require(m.jnt_range[joint.id, 0] <= 0 <= m.jnt_range[joint.id, 1], 'open fingers outside range')
                fqa.append(int(joint.qposadr[0])); fda.append(int(joint.dofadr[0])); faids.append(act.id)
        self.qa, self.da, self.aids, self.fqa, self.fda, self.faids = (
            np.asarray(x, dtype=int) for x in (qa, da, aids, fqa, fda, faids))
        require(m.nu == 16 and len(set(aids + faids)) == 16, 'unexpected motor set')
        self.ranges = np.asarray(ranges).reshape(2, 6, 2)
        self.dt = float(m.opt.timestep)
        require(math.isfinite(self.dt) and 0 < self.dt <= .005, 'unsupported physics timestep')
        require(abs(round(.05 / self.dt) * self.dt - .05) < 1e-10, 'scorer sampling requires integral 0.05 s steps')
        self.travel = float(m.numeric('gripper_closing_travel').data[0])
        require(math.isfinite(self.travel) and 0 < self.travel <= .04, 'invalid finger travel')
        require(self.servo['opening_stop_buffer_m'] < self.travel, 'opening buffer exceeds travel')
        self.finger_jids = np.array([m.joint(f'{a}_gripper{j}').id for a in ('left', 'right') for j in (1, 2)])
        self.arm_jids = np.array([m.joint(f'{a}_j{j}').id for a in ('left', 'right') for j in range(1, 7)])
        self._geom_roles = []
        for gid in range(m.ngeom):
            names, bid = [], int(m.geom_bodyid[gid])
            while bid:
                names.append(m.body(bid).name or '')
                bid = int(m.body_parentid[bid])
            arm = next((a for a in ('left', 'right') if any(n.startswith(a + '_') for n in names)), None)
            self._geom_roles.append((arm, any(n.startswith('object_') for n in names),
                                     m.geom(gid).name == 'table' or 'table' in names))
        self._reset_numerical_evidence()
        for index, jid in enumerate([m.joint(f'{a}_gripper{j}').id for a in ('left', 'right') for j in (1, 2)]):
            end = (-1 if index % 2 == 0 else 1) * self.travel
            require(m.jnt_range[jid, 0] - 1e-9 <= end <= m.jnt_range[jid, 1] + 1e-9, 'finger close outside range')
        self.cam = m.camera('main').id
        require(m.cam_bodyid[self.cam] == 0, 'main observation camera must be fixed')
        self.width, self.height = map(int, m.cam_resolution[self.cam])
        sx, sy = m.cam_sensorsize[self.cam]
        fx, fy, cx, cy = m.cam_intrinsic[self.cam]
        require(self.width > 0 and self.height > 0 and sx > 0 and sy > 0 and fx > 0 and fy > 0,
                'missing numerical camera intrinsics')
        self.K = np.array([[fx / sx * self.width, 0, self.width / 2 - cx / sx * self.width],
                           [0, fy / sy * self.height, self.height / 2 + cy / sy * self.height], [0, 0, 1.]])
        self.kinematics = RobotKinematics.from_model(m)

    def _check_motor(self, joint, act, limit):
        m, mj = self.model, self.mj
        require(m.jnt_type[joint.id] in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE)
                and m.jnt_limited[joint.id], 'expected limited scalar motor joint')
        require(m.actuator_trnid[act.id, 0] == joint.id
                and m.actuator_trntype[act.id] == mj.mjtTrn.mjTRN_JOINT
                and m.actuator_dyntype[act.id] == mj.mjtDyn.mjDYN_NONE
                and m.actuator_gaintype[act.id] == mj.mjtGain.mjGAIN_FIXED
                and m.actuator_biastype[act.id] == mj.mjtBias.mjBIAS_NONE
                and m.actuator_gainprm[act.id, 0] == 1
                and np.array_equal(m.actuator_gear[act.id], [1, 0, 0, 0, 0, 0])
                and m.actuator_forcelimited[act.id]
                and np.array_equal(m.actuator_forcerange[act.id], [-limit, limit]),
                'expected bounded direct-force motor convention')

    def reset(self, reference, poses):
        m, d, mj = self.model, self.data, self.mj
        mj.mj_resetData(m, d)
        # Only the first encoder row is read. No recorded gripper/trajectory use.
        with np.load(reference, allow_pickle=False) as archive:
            first = numeric(archive['joints'][0], (2, 6), 'initial encoders')
        require(np.all(first >= self.ranges[:, :, 0]) and np.all(first <= self.ranges[:, :, 1]),
                'initial encoders outside limits')
        d.qpos[self.qa] = first.ravel()
        d.qpos[self.fqa] = 0
        self.finger_targets = d.qpos[self.fqa].copy()
        self._reset_numerical_evidence()
        for name, values in poses.items():
            require(name.startswith('object_'), 'layout may only reset named objects')
            body = m.body(name).id
            require(m.body_jntnum[body] == 1, 'layout body must have one free joint')
            jid = int(m.body_jntadr[body])
            require(m.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE, 'layout may reset only free bodies')
            pose = numeric(values, (7,), 'layout pose')
            require(abs(np.linalg.norm(pose[3:]) - 1) < 1e-6, 'invalid layout quaternion')
            adr = int(m.jnt_qposadr[jid])
            d.qpos[adr:adr + 7] = pose
        mj.mj_forward(m, d)
        self.check_state()
        # Validate the extracted robot-only chain against reset encoders, no dynamics.
        for i, arm in enumerate(('left', 'right')):
            sid = m.site(f'{arm}_tcp').id
            T = self.kinematics.fk(arm, first[i])
            require(np.allclose(T[:3, 3], d.site_xpos[sid], atol=1e-7) and
                    np.allclose(T[:3, :3], d.site_xmat[sid].reshape(3, 3), atol=1e-7),
                    'encoder FK disagrees with robot frame contract')

    def encoders(self):
        return self.data.qpos[self.qa].reshape(2, 6).copy()

    def _reset_numerical_evidence(self):
        self.numerical_invalid = False
        self.numerical_evidence = {
            'arm_limit_excursion_rad': {'maximum': 0., 'witness': None},
            'finger_limit_excursion_m': {'maximum': 0., 'witness': None},
            'penetration_m': {kind: {'maximum': 0., 'witness': None}
                              for kind in ('robot_table', 'robot_self', 'robot_robot')}}

    def _track_geometry(self):
        m, d = self.model, self.data
        failures = []
        for key, jids, addresses, tolerance in (
                ('arm_limit_excursion_rad', self.arm_jids, self.qa, .005),
                ('finger_limit_excursion_m', self.finger_jids, self.fqa, .0002)):
            values, ranges = d.qpos[addresses], m.jnt_range[jids]
            excursions = np.maximum(0., np.maximum(ranges[:, 0] - values, values - ranges[:, 1]))
            index = int(np.argmax(excursions))
            maximum = float(excursions[index])
            metric = self.numerical_evidence[key]
            if maximum > metric['maximum']:
                metric.update(maximum=maximum, witness={
                    'time_sec': float(d.time), 'joint_id': int(jids[index]),
                    'joint': m.joint(jids[index]).name, 'value': float(values[index]),
                    'range': ranges[index].tolist(),
                    'side': 'lower' if values[index] < ranges[index, 0] else 'upper'})
            if metric['maximum'] > tolerance:
                failures.append(key)
        for contact in d.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if min(g1, g2) < 0:
                continue
            a1, obj1, table1 = self._geom_roles[g1]
            a2, obj2, table2 = self._geom_roles[g2]
            # Object geometry belongs exclusively to the external scorer.
            if obj1 or obj2:
                continue
            kind = ('robot_self' if a1 == a2 else 'robot_robot') if a1 and a2 else (
                'robot_table' if (a1 and table2) or (a2 and table1) else None)
            if kind is None:
                continue
            depth = max(0., -float(contact.dist))
            metric = self.numerical_evidence['penetration_m'][kind]
            if depth > metric['maximum']:
                metric.update(maximum=depth, witness={
                    'time_sec': float(d.time), 'geom_ids': [g1, g2],
                    'geoms': [m.geom(g).name for g in (g1, g2)],
                    'bodies': [m.body(m.geom_bodyid[g]).name for g in (g1, g2)],
                    'position_m': contact.pos.tolist(), 'distance_m': float(contact.dist)})
        for kind, metric in self.numerical_evidence['penetration_m'].items():
            if metric['maximum'] > .001:
                failures.append(kind)
        if failures:
            self.numerical_invalid = True
            # Policy may catch action errors; numerical measurements and contact
            # identities stay in trusted evidence, never in the public error.
            raise NumericalFailure('robot limit or forbidden-contact tolerance exceeded')

    def check_state(self):
        d = self.data
        if not all(np.isfinite(x).all() for x in (d.qpos, d.qvel, d.qacc, d.ctrl, d.qfrc_bias)) or not math.isfinite(d.time):
            self.numerical_invalid = True
            raise NumericalFailure('non-finite physics state')
        self._track_geometry()
        if any(w.number for w in d.warning):
            self.numerical_invalid = True
            raise NumericalFailure('MuJoCo warning: ' + str([(i, w.number) for i, w in enumerate(d.warning) if w.number]))
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):
            self.numerical_invalid = True
            raise NumericalFailure('external force detected')
        if np.max(np.abs(d.qvel)) > 100:
            self.numerical_invalid = True
            raise NumericalFailure('unbounded state velocity')

    def synchronize(self):
        # mj_step integrates qpos after its forward pass. Refresh derived frames
        # and contacts at the current encoder time before rendering/scoring.
        self.mj.mj_forward(self.model, self.data)
        self.check_state()

    def observe(self):
        self.synchronize()
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height=self.height, width=self.width)
        self.renderer.disable_segmentation_rendering()
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera='main')
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(self.data, camera='main')
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ObservationError('invalid RGB frame')
        if depth.shape != (self.height, self.width) or not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ObservationError('invalid metric depth frame')
        T = np.eye(4)
        T[:3, :3] = self.data.cam_xmat[self.cam].reshape(3, 3) @ np.diag([1., -1., -1.])
        T[:3, 3] = self.data.cam_xpos[self.cam]
        return dict(rgb=rgb, depth=depth, K=self.K.copy(), T_world_from_camera=T,
                    encoders=self.encoders(), time_sec=float(self.data.time))

    def step(self, joints, openings):
        q = numeric(joints, (2, 6), 'joint targets')
        opening = numeric(openings, (2,), 'finger targets')
        require(np.all(q >= self.ranges[:, :, 0]) and np.all(q <= self.ranges[:, :, 1]), 'joint target out of range')
        require(np.all((opening >= 0) & (opening <= 1)), 'finger target out of range')
        self.check_state()
        d = self.data
        kp = np.tile([1500, 1800, 1500, 700, 700, 500], 2)
        kd = np.tile([65, 70, 60, 18, 18, 14], 2)
        d.ctrl[self.aids] = np.clip(kp * (q.ravel() - d.qpos[self.qa]) - kd * d.qvel[self.da]
                                    + d.qfrc_bias[self.da], -100, 100)
        sign = np.tile([-1., 1.], 2)
        buffer = self.servo['opening_stop_buffer_m']
        closure = buffer + np.repeat(1 - opening, 2) * (self.travel - buffer)
        # Match the validated servo convention: closing is immediate, opening
        # reduces the previous target closure by at most speed * physics dt.
        closure = np.maximum(closure, sign * self.finger_targets
                             - self.servo['opening_velocity_limit_m_s'] * self.dt)
        self.finger_targets = sign * closure
        force = self.servo['finger_force_limit_N']
        d.ctrl[self.faids] = np.clip(1600 * (self.finger_targets - d.qpos[self.fqa])
                                     - self.servo['finger_velocity_damping_Ns_m'] * d.qvel[self.fda]
                                     + d.qfrc_bias[self.fda], -force, force)
        self.mj.mj_step(self.model, d)
        self.synchronize()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


class Evidence:
    """Incremental, fsynced evidence survives failed/killed episode processes."""
    def __init__(self, output):
        self.output = Path(output)
        self.trace = (self.output / 'trace.jsonl').open('xb')
        self.state_dir = self.output / 'states'
        self.frame_dir = self.output / 'frames'
        self.state_dir.mkdir(); self.frame_dir.mkdir()
        self.states = self.frames = 0

    def log(self, event, **data):
        row = {'event': event, 'wall_monotonic': time.monotonic(), **data}
        self.trace.write(json_bytes(row).replace(b'\n', b'') + b'\n')
        self.trace.flush(); os.fsync(self.trace.fileno())

    def state(self, world, targets, openings, steps):
        d = world.data
        path = self.state_dir / f'{self.states:06d}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, time_sec=d.time, qpos=d.qpos.copy(), qvel=d.qvel.copy(),
                                ctrl=d.ctrl.copy(), targets=targets.copy(), openings=openings.copy(), steps=steps,
                                numerical_evidence=json_bytes(world.numerical_evidence).decode())
            stream.flush(); os.fsync(stream.fileno())
        self.states += 1

    def frame(self, observation):
        ident = f'frame_{self.frames:05d}'
        with (self.frame_dir / f'{ident}.npz').open('xb') as stream:
            np.savez_compressed(stream, **observation)
            stream.flush(); os.fsync(stream.fileno())
        self.frames += 1
        self.log('frame', frame=ident, time_sec=observation['time_sec'])
        return ident

    def close(self):
        self.trace.close()


class _Controller:
    """Trusted executor. Public facade below exposes only bound sensor/action calls."""
    def __init__(self, world, perception, evidence, task, deadline):
        self.world, self.perception, self.evidence = world, perception, evidence
        self.cap, self.deadline = SIM_CAPS[task], deadline
        self.targets = world.encoders()
        self.openings = np.ones(2)
        self.steps, self.last_state_step, self.calls = 0, -1, 0
        self.dwell = False
        self.last_observation_time = None
        self.evidence.state(world, self.targets, self.openings, self.steps)

    def check(self):
        if time.monotonic() >= self.deadline:
            raise EpisodeLimit('wall time exhausted')
        self.world.check_state()
        if self.world.data.time > self.cap + 1e-9:
            raise EpisodeLimit('simulation time exceeded')

    def duration_steps(self, duration):
        self.check()
        value = float(duration)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('duration must be positive and finite')
        count = int(math.ceil(value / self.world.dt - 1e-10))
        limit = self.cap if self.dwell else self.cap - .5
        if self.world.data.time + count * self.world.dt > limit + 1e-9:
            raise EpisodeLimit('action would exhaust duration reserved for final dwell')
        return count

    def step(self):
        self.check()
        self.world.step(self.targets, self.openings)
        self.steps += 1
        sample_steps = int(round(.05 / self.world.dt))
        if self.steps % sample_steps == 0:
            self.evidence.state(self.world, self.targets, self.openings, self.steps)
            self.last_state_step = self.steps

    def log(self, event, **data):
        self.check()
        self.calls += 1
        if self.calls > 10000:
            raise EpisodeLimit('policy IO/log call budget exhausted')
        require(isinstance(event, str) and len(event) < 200, 'invalid log event')
        require(len(json_bytes(data)) < 256000, 'oversized policy log')
        self.evidence.log('policy', stage_event=event, time_sec=float(self.world.data.time), data=data)

    def observe(self):
        self.log('observe')
        # A new capture on each call; repeated captures at one physics instant are explicit.
        obs = self.world.observe()
        frame = self.evidence.frame(obs)
        obs['source_frame'] = frame
        return obs

    def locate(self):
        obs = self.observe()
        try:
            objects = self.perception.locate(obs)
            if not isinstance(objects, dict) or not objects:
                raise ObservationError('perception returned no object evidence')
            for name, obj in objects.items():
                require(isinstance(name, str) and isinstance(obj, dict), 'invalid object evidence')
                numeric(obj['position'], (3,), 'perceived position')
                rotation(obj['rotation'])
                extent = numeric(obj['extent'], (3,), 'perceived extent')
                confidence = float(obj['confidence'])
                require(np.all(extent > 0) and math.isfinite(confidence) and 0 < confidence <= 1,
                        'invalid perception extent/confidence')
                require(bool(obj.get('source_frame')), 'missing perception frame provenance')
            self.check()
            self.log('perception', objects=objects)
            return objects
        except (EpisodeLimit, NumericalFailure):
            raise
        except Exception as exc:
            raise ObservationError(f'{type(exc).__name__}: {exc}') from exc

    def tcp(self, arm):
        self.log('tcp', arm=arm)
        index = self.arm_index(arm)
        return self.world.kinematics.fk(arm, self.world.encoders()[index]).copy()

    @staticmethod
    def arm_index(arm):
        if arm not in ('left', 'right'):
            raise ValueError('arm must be left or right')
        return ('left', 'right').index(arm)

    def move_tcp(self, arm, position, orientation, duration=1.):
        index = self.arm_index(arm)
        p, R = numeric(position, (3,), 'TCP target'), rotation(orientation)
        count = self.duration_steps(duration)
        if duration < .1:
            raise MotionError('Cartesian move duration below 0.1 seconds')
        start = self.world.kinematics.fk(arm, self.world.encoders()[index])
        self.log('move_tcp_begin', arm=arm, position=p, rotation=R, duration=duration)
        # Preflight every path waypoint without touching physics. Unreachable paths
        # fail before motor execution; servo IK still uses measured encoders below.
        nway = max(2, int(math.ceil(duration / .025)))
        seed = self.world.encoders()[index]
        for k in range(1, nway + 1):
            self.check()
            wp, wr = waypoint(start, p, R, min(1., k / nway / .8))
            seed = self.world.kinematics.ik(arm, wp, wr, seed)
        period = max(1, int(round(.02 / self.world.dt)))
        rate = 2.0  # rad/s, targets independently limited from physical motor torques
        for k in range(count):
            if k % period == 0:
                encoders = self.world.encoders()[index]
                wp, wr = waypoint(start, p, R, min(1., (k + period) / count / .8))
                solution = self.world.kinematics.ik(arm, wp, wr, encoders)
                # Per-step interpolation is below; never jump an encoder target.
                desired = solution
            delta = np.clip(desired - self.targets[index], -rate * self.world.dt, rate * self.world.dt)
            self.targets[index] += delta
            self.step()
        achieved = self.world.kinematics.fk(arm, self.world.encoders()[index])
        pe = float(np.linalg.norm(p - achieved[:3, 3]))
        re = float(np.linalg.norm(Rotation.from_matrix(R @ achieved[:3, :3].T).as_rotvec()))
        self.log('move_tcp_end', arm=arm, position_error_m=pe, rotation_error_rad=re)
        if pe > .015 or re > .12:
            raise MotionError(f'Cartesian servo tracking failed: {pe:.5f}m, {re:.5f}rad')

    def gripper(self, arm, opening, duration=.5):
        index = self.arm_index(arm)
        value = float(opening)
        require(math.isfinite(value) and 0 <= value <= 1, 'opening must be in [0,1]')
        count = self.duration_steps(duration)
        start = float(self.openings[index])
        self.log('gripper', arm=arm, opening=value, duration=duration)
        for k in range(count):
            self.openings[index] = start + (value - start) * (k + 1) / count
            self.step()

    def wait(self, duration):
        count = self.duration_steps(duration)
        self.log('wait', duration=duration)
        for _ in range(count):
            self.step()


class PolicyIO:
    """Architectural boundary, not a hostile-code sandbox.

    Private bound closures contain trusted execution state. No public model,
    data, layout, scorer, recording or episode-seed attributes exist.
    """
    __slots__ = ('__observe', '__locate', '__tcp', '__move', '__grip', '__wait', '__log')

    def __init__(self, controller):
        self.__observe, self.__locate, self.__tcp = controller.observe, controller.locate, controller.tcp
        self.__move, self.__grip = controller.move_tcp, controller.gripper
        self.__wait, self.__log = controller.wait, controller.log

    def observe(self):
        return self.__observe()

    def locate(self):
        return self.__locate()

    def tcp(self, arm):
        return self.__tcp(arm)

    def move_tcp(self, arm, position, rotation, duration=1.):
        return self.__move(arm, position, rotation, duration)

    def gripper(self, arm, opening, duration=.5):
        return self.__grip(arm, opening, duration)

    def wait(self, duration):
        return self.__wait(duration)

    def log(self, event, **data):
        return self.__log(event, **data)


def _load_program(candidate_dir):
    # Never emit __pycache__ into the immutable candidate closure.
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        path = Path(candidate_dir) / 'program.py'
        spec = importlib.util.spec_from_file_location('_practice_frozen_program', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        require(callable(getattr(module, 'run', None)), 'candidate does not export run')
        return module
    finally:
        sys.dont_write_bytecode = old


def _episode_child(task, candidate_dir, layout, output, frozen, deadline):
    """Child entry also used by synthetic tests with isolated injected modules."""
    # Isolate all children (including SAM subprocess) for the parent's watchdog.
    if multiprocessing.current_process().name != 'MainProcess':
        os.setsid()
    output = Path(output)
    evidence = Evidence(output)
    world = perception = controller = None
    result = dict(success=False, task_success=False, engineering_valid=False, status='failure', reason='',
                  task=task, layout_id=layout['id'], split=layout['split'], steps=0, sim_time_sec=0.,
                  numerical_invalid=False, inputs_sha256=frozen['inputs_sha256'],
                  candidate_sha256=verify_candidate(candidate_dir, task)['candidate_sha256'])
    try:
        from . import perception as perception_module, scoring
        evidence.log('initialize', task=task, layout_id=layout['id'], policy_inputs='RGBD and encoders only')
        params = read_json(Path(candidate_dir) / 'params.json')
        item = frozen['tasks'][task]
        record = read_json(item['record'])
        world = MotorWorld(item['scene'], servo_config(record))
        world.reset(item['reference'], layout['poses'])
        # Perception receives only frozen metadata/config, never world/scorer/layout.
        perception = perception_module.Perception(task, params.get('perception', {}), output / 'perception')
        controller = _Controller(world, perception, evidence, task, deadline)
        controller.observe()
        program = _load_program(candidate_dir)
        diagnostic = program.run(PolicyIO(controller), params)
        require(isinstance(diagnostic, dict), 'candidate run must return a diagnostic dict')
        evidence.log('program_return', diagnostic=diagnostic, grants_task_success=False)
        controller.check()
        controller.dwell = True
        history = []
        # Include both endpoints: eleven snapshots span exactly the required 0.5 s.
        for sample in range(11):
            if sample:
                controller.wait(.05)
            world.synchronize()
            snap = scoring.snapshot(task, world.model, world.data, record)
            history.append(snap)
            evidence.log('scorer_snapshot', index=sample, snapshot=snap)
        score = scoring.evaluate(task, world.model, world.data, record, history)
        require(isinstance(score, dict) and type(score.get('success')) is bool
                and all(key in score for key in ('checks', 'measurements', 'reason')), 'invalid external scorer result')
        json_bytes(score)  # reject non-finite or unserializable scorer measurements
        world.check_state()
        verify_candidate(candidate_dir, task)
        # Recheck every frozen dependency, including checkpoint/code, after execution.
        load_inputs(frozen['campaign'])
        result.update(success=score['success'], task_success=score['success'], engineering_valid=True,
                      status='completed', reason=score['reason'], score=score, program_diagnostic=diagnostic)
    except BaseException as exc:
        result.update(reason=f'{type(exc).__name__}: {exc}', numerical_invalid=isinstance(exc, NumericalFailure),
                      status='numerical_invalid' if isinstance(exc, NumericalFailure) else (
                          'timeout' if isinstance(exc, EpisodeLimit) else 'failure'))
        evidence.log('failure', reason=result['reason'], traceback=traceback.format_exc())
    finally:
        if world is not None:
            result['numerical_evidence'] = world.numerical_evidence
            result['sim_time_sec'] = float(world.data.time) if math.isfinite(world.data.time) else None
            result['steps'] = int(round(world.data.time / world.dt)) if math.isfinite(world.data.time) else 0
        if world is not None:
            try:
                evidence.state(world, controller.targets if controller else world.encoders(),
                               controller.openings if controller else np.ones(2), result['steps'])
            except Exception as exc:
                result.update(success=False, task_success=False, engineering_valid=False,
                              status='failure', reason=f'state evidence failure: {exc}')
        for resource in (perception, world):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    result.update(success=False, task_success=False, engineering_valid=False,
                                  status='failure', reason=f'resource close failed: {exc}')
        result['frames'] = evidence.frames
        if world is not None and world.numerical_invalid:
            result.update(success=False, task_success=False, engineering_valid=False,
                          numerical_invalid=True, status='numerical_invalid')
        result['state_samples'] = evidence.states
        evidence.log('result', **result)
        write_new(output / 'result.json', result)
        evidence.close()


def _partial_progress(output):
    for path in sorted((Path(output) / 'states').glob('*.npz'), reverse=True):
        try:
            with np.load(path, allow_pickle=False) as state:
                sample_time = float(state['time_sec'])
                if math.isfinite(sample_time):
                    metrics = json.loads(str(state['numerical_evidence'])) if 'numerical_evidence' in state else None
                    return int(state['steps']), sample_time, metrics
        except (OSError, ValueError, EOFError, KeyError):
            continue
    return 0, 0., None


def _kill_group(process):
    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            process.kill()
        process.join(timeout=5)
        if process.is_alive():
            process.kill(); process.join(timeout=5)


@contextmanager
def _task_lock(campaign, task):
    """One episode per task, at most three tasks concurrently, across processes."""
    import fcntl
    with (budget_root(campaign) / f'.running_{task}.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(f'task already running: {task}') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_episode(task, candidate_dir, layout, output, frozen_inputs):
    """Reserve once, execute frozen policy in a watchdog process, retain all failures."""
    start = time.monotonic()
    actual = load_inputs(frozen_inputs['campaign'])
    require(actual == frozen_inputs, 'caller supplied altered frozen inputs')
    with _task_lock(actual['campaign'], task):
        reservation = reserve_attempt(task, candidate_dir, layout, output, actual)
        deadline = start + actual['protocol']['wall_cap_sec']
        process = multiprocessing.get_context('spawn').Process(
            target=_episode_child, args=(task, str(candidate_dir), layout, str(output), actual, deadline))
        path = Path(output) / 'result.json'
        watchdog_reason = None
        try:
            process.start()
            process.join(timeout=max(0., deadline - time.monotonic()))
            if process.is_alive():
                watchdog_reason = 'wall cap exhausted (including perception); child process group killed'
                _kill_group(process)
            elif process.exitcode != 0:
                watchdog_reason = f'episode process exited {process.exitcode}'
        except BaseException as exc:
            watchdog_reason = f'parent interruption/start failure: {type(exc).__name__}: {exc}'
            if process.pid:
                _kill_group(process)
        if not path.exists():
            observed_steps, observed_time, numerical_evidence = _partial_progress(output)
            result = dict(success=False, task_success=False, engineering_valid=False,
                          status='timeout' if 'wall cap exhausted' in (watchdog_reason or '') else 'failure',
                          reason=watchdog_reason or 'episode exited without a result', numerical_invalid=None,
                          steps=observed_steps, sim_time_sec=observed_time, task=task, layout_id=layout['id'], split=layout['split'],
                          numerical_evidence=numerical_evidence,
                          inputs_sha256=actual['inputs_sha256'], candidate_sha256=reservation['candidate_sha256'],
                          state_evidence='last durable state; actual final step/time may be later')
            write_new(path, result)
        result = read_json(path)
        if watchdog_reason and (result.get('success') or result.get('engineering_valid')):
            # A child must exit cleanly within the wall budget even after writing a score.
            override = {**result, 'success': False, 'task_success': False, 'engineering_valid': False,
                        'status': 'timeout' if 'wall cap' in watchdog_reason else 'failure', 'reason': watchdog_reason}
            write_new(Path(output) / 'watchdog_failure.json', override)
            # Preserve original result bytes; summarize must honor this terminal override.
            result = override
        write_new(Path(output) / 'PROCESS.json', {'exitcode': process.exitcode, 'watchdog_reason': watchdog_reason,
                                                 'wall_time_sec': time.monotonic() - start})
        return result
