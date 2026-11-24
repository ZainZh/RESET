# Public controller IO for the G1 suite (r2). Pure code: no MuJoCo import, no model/data/scorer/layout, no backend or guard object.
# The trusted harness hands over a PublicBackend: exactly five callbacks (dt, time, encoders, observe, step). Privileged world/guard/scorer objects stay in the
# trusted harness registry/closures; PublicIO keeps only these callables, so no model/data/scorer/layout attribute is reachable through ordinary fields.
# Source-compatible surface: observe(), tcp(side), move(side, position, rotation, duration), grip(side, opening, duration), home(side), wait(duration), log(event, **data), demo, instruction, objects.
# Poses are in the unchanged SOURCE frame; the adapter/hand module applies the fixed rigid transform. opening: 1 open, 0 closed.
# Physics-path faults are LATCHED: step (numerical, sim time cap, backend fault) AND observe (backend fault, malformed observation). Once set, every later step/observe re-raises it even if the controller swallowed it.
# Planning errors (IK invalid, unsupported grasp profile) are NOT latched: source controllers legitimately try alternative candidates; they stay visible in the log.
# The sim time cap is enforced BEFORE stepping: a step that would end beyond cap (+1e-9 s float tolerance) is refused, so the world never advances past the cap.
import math
import numpy as np
from .robot_adapter import (RobotAdapter, AdapterError, UnsupportedGraspProfile, IncompatibleBaseline, GRASP_PROFILE,
                            ENCODER_NAMES, ARM, SIDES, check_encoders)

DEMO_KEYS = ('events', 'home', 'objects0')
OBS_KEYS = ('rgb', 'depth', 'K', 'T_world_from_camera', 'encoders', 'time_sec')
JOINT_RATE = 2.5  # rad/s per-joint target slew, same as the source IO
CAP_TOL = 1e-9


class Limit(RuntimeError):
    pass


class PublicBackend:
    # the only thing PublicIO receives from the trusted side: plain callables, nothing else
    __slots__ = ('dt', 'time', 'encoders', 'observe', 'step')

    def __init__(self, dt, time, encoders, observe, step):
        for f in (time, encoders, observe, step):
            if not callable(f):
                raise AdapterError('callback_not_callable')
        self.dt, self.time, self.encoders, self.observe, self.step = float(dt), time, encoders, observe, step


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    ang = math.acos(max(-1.0, min(1.0, c)))
    if ang < 1e-9:
        return np.zeros(3)
    if math.pi - ang < 1e-6:
        w, v = np.linalg.eigh((R + R.T) / 2.0)
        return v[:, int(np.argmax(w))] * ang
    a = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(ang))
    return a * ang


def _expmap(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * K @ K


def waypoint(T0, p, R, s):
    pos = T0[:3, 3] + (p - T0[:3, 3]) * s
    return pos, _expmap(_rotvec(R @ T0[:3, :3].T) * s) @ T0[:3, :3]


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    return v


class RobotOnlyView:
    # Stand-in for the source io._w used by box b8 (io._w.kinematics, io._w.encoders(), RobotCollision). The source hooks assume a six-joint xTrainer
    # kinematics and collision model; they are NOT a G1 model. No live World/model/data is reachable through this object. Any use raises
    # IncompatibleBaseline so b8 fails honestly (counted as an incompatibility failure) instead of being silently modified.
    def __init__(self, why='source six-joint xTrainer kinematics/collision hooks have no G1 equivalent in the baseline'):
        self._why = why

    @property
    def kinematics(self):
        raise IncompatibleBaseline('io._w.kinematics: ' + self._why)

    def encoders(self):
        raise IncompatibleBaseline('io._w.encoders(): ' + self._why)


class PublicIO:
    def __init__(self, task, cb, adapter, demo, cap, instruction='', objects=None):
        if not isinstance(cb, PublicBackend):
            raise AdapterError('callbacks_type')
        if not isinstance(adapter, RobotAdapter):
            raise AdapterError('adapter_type')
        if adapter.task != task:
            raise AdapterError('adapter_task_mismatch')
        if not isinstance(demo, dict) or any(k not in demo for k in DEMO_KEYS):
            raise AdapterError('demo_priors_incomplete')
        cap = float(cap)
        if not math.isfinite(cap) or cap <= 0:
            raise AdapterError('bad_cap')
        self.task, self.adapter, self._cap, self._cb = task, adapter, cap, cb
        self.dt = float(cb.dt)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise AdapterError('bad_dt')
        self.demo = demo  # fixed one-shot priors decoded by the harness via source FK; never from current DEV objects
        self.instruction = instruction
        self.objects = objects or {}
        self.error = None  # latched physics-path error
        self.targets = self._enc()
        self.home_q = dict(self.targets)
        self.openings = {s: 1.0 for s in SIDES}
        self.frames = []
        self.records = []
        self._w = RobotOnlyView()

    def _enc(self):
        # encoder read is a physics-path callback: a fault here latches
        if self.error is not None:
            raise self.error
        try:
            return check_encoders(self._cb.encoders())
        except Exception as e:
            self.error = e
            raise

    def now(self):
        return float(self._cb.time())

    def _step(self):
        if self.error is not None:
            raise self.error
        try:
            t = float(self._cb.time())
            if not math.isfinite(t):
                raise AdapterError('bad_sim_time')
            if t + self.dt > self._cap + CAP_TOL:
                raise Limit('sim time cap')
            self._cb.step(dict(self.targets))
        except Exception as e:
            self.error = e
            raise

    def _side(self, side):
        if side not in SIDES:
            raise AdapterError('bad_side')
        return side

    def observe(self, keep=True):
        if self.error is not None:
            raise self.error
        try:
            o = self._cb.observe()
            if not isinstance(o, dict) or any(k not in o for k in OBS_KEYS):
                raise AdapterError('observation_incomplete')
            out = {k: o[k] for k in OBS_KEYS}
        except Exception as e:
            self.error = e
            raise
        if keep:
            self.frames.append(out['rgb'])
        return out

    def tcp(self, side):
        self._side(side)
        return self.adapter.tcp(self._enc(), side)

    def move(self, side, position, rotation, duration=1.5):
        self._side(side)
        self.adapter.profile()  # unsupported task fails before any motion
        start = self.tcp(side)
        p, R = np.asarray(position, float), np.asarray(rotation, float)
        enc = self._enc()
        for k in range(1, 21):  # preflight: every waypoint must be kinematically reachable before any motion
            wp, wr = waypoint(start, p, R, k / 20.0)
            enc.update(self.adapter.plan(enc, side, wp, wr))
        n = max(1, int(math.ceil(float(duration) / self.dt)))
        period = max(1, int(round(0.02 / self.dt)))
        names = [f'{side}_{a}_joint' for a in ARM]
        desired = {a: self.targets[a] for a in names}
        for k in range(n):
            if k % period == 0:
                wp, wr = waypoint(start, p, R, min(1.0, (k + period) / n / 0.8))
                desired = self.adapter.plan(self._enc(), side, wp, wr)
            for a in names:
                self.targets[a] += float(np.clip(desired[a] - self.targets[a], -JOINT_RATE * self.dt, JOINT_RATE * self.dt))
            self._step()
        return float(np.linalg.norm(self.tcp(side)[:3, 3] - p))

    def grip(self, side, opening, duration=0.5):
        self._side(side)
        if isinstance(opening, (bool, np.bool_)) or not isinstance(opening, (int, float, np.integer, np.floating)) or not 0.0 <= float(opening) <= 1.0:
            raise AdapterError('bad_opening')
        s = self.openings[side]
        n = max(1, int(math.ceil(float(duration) / self.dt)))
        for k in range(n):
            o = s + (float(opening) - s) * (k + 1) / n
            self.targets.update(self.adapter.hand_targets(self.task, side, o))
            self.openings[side] = o
            self._step()

    def home(self, side, duration=2.0):
        self._side(side)
        names = [f'{side}_{a}_joint' for a in ARM]
        q0 = {a: self.targets[a] for a in names}
        n = max(1, int(math.ceil(float(duration) / self.dt)))
        for k in range(n):
            for a in names:
                self.targets[a] = q0[a] + (self.home_q[a] - q0[a]) * (k + 1) / n
            self._step()

    def wait(self, duration):
        for _ in range(max(1, int(math.ceil(float(duration) / self.dt)))):
            self._step()

    def log(self, event, **data):
        self.records.append({'t': self.now(), 'event': event, **{k: _plain(v) for k, v in data.items()}})
