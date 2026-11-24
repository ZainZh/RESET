"""Franka copy of rsi/lift2_suite/common/embodiment/facade.py (unchanged logic; RobotView from the Franka world). run(io) facade for the target robot: same calls and semantics as rsi.env.IO (observe/tcp/move/grip/home/wait/
demo/log, instruction, objects, home_q, targets, openings). move() is the source code path (preflight 20
waypoints, 20 ms IK re-solve from encoders, 2.5 rad/s target slew, near-solution fallback) on TaskKinematics.
Opening is normalized (1 open, 0 closed) and mapped by the World to the robot's master gripper joint.
Also records a 20 Hz command/encoder trace (trusted evidence, not visible to policy logic).
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from twin.practice.motion import waypoint, KinematicsError
from .world import RobotView

_TIMEOUT = set()  # id(io) of episodes that hit the sim time cap: trusted latch, a controller that catches Limit cannot clear it
_TRUSTED = {}  # id(io) -> real TargetWorld; runner-private, never an attribute of the policy-facing io (pop it when the episode ends)

ARMS = ('left', 'right')


class Limit(RuntimeError):
    pass


class IO:
    def __init__(self, world, task, cap, log, demo):
        _TRUSTED[id(self)] = world
        self._w, self._cap, self._log = RobotView(world), cap, log
        self.targets = world.encoders(); self.home_q = world.encoders(); self.openings = np.ones(2)
        self.instruction = task.instruction
        self.objects = task.objects
        self.demo = demo
        self.frames = []; self.record_every = 0.; self._next_frame = 0.
        self.trace = []; self._next_trace = 0.
        self.frame_hook = None  # trusted video recorder (presentation renders only)

    def _step(self):
        w = _TRUSTED[id(self)]
        if w.data.time > self._cap:
            _TIMEOUT.add(id(self))
            raise Limit('sim time cap')
        w.step(self.targets, self.openings)
        t = w.data.time
        if t >= self._next_trace:
            self._next_trace += .05
            self.trace.append(np.r_[t, self.targets.ravel(), self.openings, self._w.encoders().ravel(), self._w.gripper_positions()])
        if self.frame_hook is not None:
            self.frame_hook(w, self)
        if self.record_every and t >= self._next_frame:
            self._next_frame += self.record_every; self.frames.append(self._w.observe()['rgb'])

    def observe(self, keep=True, camera=None):
        o = self._w.observe(camera)
        if keep: self.frames.append(o['rgb'])
        return o

    def tcp(self, arm):
        i = ARMS.index(arm)
        return self._w.kinematics.fk(arm, self._w.encoders()[i])

    def move(self, arm, p, R, duration=1.5):
        i = ARMS.index(arm)
        start = self.tcp(arm); p = np.asarray(p, float); R = np.asarray(R, float)
        n = max(1, int(math.ceil(duration / self._w.dt))); period = max(1, int(round(.02 / self._w.dt)))
        seed = self._w.encoders()[i]
        for k in range(1, 21):  # preflight
            wp, wr = waypoint(start, p, R, k / 20); seed = self._ik(arm, wp, wr, seed)
        desired = self.targets[i]
        for k in range(n):
            if k % period == 0:
                wp, wr = waypoint(start, p, R, min(1., (k + period) / n / .8))
                desired = self._ik(arm, wp, wr, self._w.encoders()[i])
            self.targets[i] += np.clip(desired - self.targets[i], -2.5 * self._w.dt, 2.5 * self._w.dt)
            self._step()
        T = self.tcp(arm)
        return float(np.linalg.norm(T[:3, 3] - p))

    def _ik(self, arm, p, R, seed):
        kin = self._w.kinematics
        try:
            return kin.ik(arm, p, R, seed, max_iterations=400)
        except KinematicsError:
            q = np.asarray(seed, float)
            for _ in range(400):
                T = kin.fk(arm, q); dp = p - T[:3, 3]
                dr = Rotation.from_matrix(R @ T[:3, :3].T).as_rotvec()
                J = kin.jacobian(arm, q); J[3:] *= .15
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[dp, dr * .15])
                lim = kin.limits[ARMS.index(arm)]
                q = np.clip(q + dq * min(1., .15 / max(np.abs(dq).max(), 1e-12)), lim[:, 0] + 1e-6, lim[:, 1] - 1e-6)
            T = kin.fk(arm, q)
            if np.linalg.norm(p - T[:3, 3]) < .005 and np.linalg.norm(Rotation.from_matrix(R @ T[:3, :3].T).as_rotvec()) < .05:
                return q
            raise

    def grip(self, arm, opening, duration=.5):
        i = ARMS.index(arm); s = float(self.openings[i])
        n = max(1, int(math.ceil(duration / self._w.dt)))
        for k in range(n):
            self.openings[i] = s + (opening - s) * (k + 1) / n; self._step()

    def home(self, arm, duration=2.0):
        i = ARMS.index(arm)
        q0 = self.targets[i].copy(); q1 = self.home_q[i]
        n = max(1, int(math.ceil(duration / self._w.dt)))
        for k in range(n):
            self.targets[i] = q0 + (q1 - q0) * (k + 1) / n
            self._step()

    def wait(self, duration):
        for _ in range(max(1, int(math.ceil(duration / self._w.dt)))): self._step()

    def log(self, event, **data):
        self._log.append({'t': float(self._w.time), 'event': event, **{k: _plain(v) for k, v in data.items()}})


def _plain(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)): return v.item()
    return v
