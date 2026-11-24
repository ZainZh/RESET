# Trusted harness scaffolding for the G1 suite (r2). Privileged side only; controllers never import this and never receive its objects.
# No MuJoCo import: model/data are duck-typed arrays supplied by the trusted world. Simulation only, DEV only.
# Disclosed support: fixed pelvis and robot-only gravcomp; this is not a balance or walking claim.
# Object initialization happens only before arm(); afterwards only actuators and the physics step may change the world.
# r2: the guard/world/scorer live in closures created by make_public_backend(); PublicIO receives only five callables. In-process Python cannot be a
# security boundary (closures are introspectable); the real isolation is the fail-closed per-seed parent harness. This removes the ordinary-attribute leak only.
import numpy as np
from .public_io import Limit, PublicBackend
from .robot_adapter import check_encoders, AdapterError

CAPS = {'collect_apple': 30.0, 'collect_cubes': 90.0, 'stack_cups': 30.0, 'put_mug_on_box': 45.0,
        'build_tower': 60.0, 'open_drawer_and_place_object': 60.0, 'sweep_objects': 40.0, 'lift_box': 40.0}


class GuardViolation(RuntimeError):
    pass


def _nonrobot_indices(model, robot_bodies):
    q, v = [], []
    for j in range(int(model.njnt)):
        if int(model.jnt_bodyid[j]) in robot_bodies:
            continue
        nq, nv = {0: (7, 6), 1: (4, 3)}.get(int(model.jnt_type[j]), (1, 1))
        qa, da = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        q += list(range(qa, qa + nq))
        v += list(range(da, da + nv))
    return np.array(q, int), np.array(v, int)


class NonRobotGuard:
    # Protects EVERY non-robot coordinate (free bodies and articulated ones such as the drawer slide) from any write between physics steps,
    # and rejects force injection, added constraints/attachments and mocap bodies.
    def __init__(self, model, data, robot_bodies, expected_neq, n_actuators=26):
        self.model, self.data = model, data
        self.qi, self.vi = _nonrobot_indices(model, set(int(b) for b in robot_bodies))
        if int(model.nmocap) != 0:
            raise GuardViolation('mocap_present')
        if int(model.nu) != n_actuators:
            raise GuardViolation('unexpected_actuator_set')
        if int(model.neq) != int(expected_neq):
            raise GuardViolation('unexpected_equality_count')
        self.neq0 = int(model.neq)
        self.armed = False

    def arm(self):
        # called once after object initialization and before the controller runs
        self.q, self.v = self.data.qpos[self.qi].copy(), self.data.qvel[self.vi].copy()
        self.armed = True

    def pre_step(self):
        if not self.armed:
            raise GuardViolation('guard_not_armed')
        if int(self.model.neq) != self.neq0 or int(self.model.nmocap) != 0:
            raise GuardViolation('constraint_added')
        if np.any(self.data.xfrc_applied) or np.any(self.data.qfrc_applied):
            raise GuardViolation('force_injection')
        if not (np.array_equal(self.data.qpos[self.qi], self.q) and np.array_equal(self.data.qvel[self.vi], self.v)):
            raise GuardViolation('nonrobot_coordinate_rewritten')

    def post_step(self):
        self.q, self.v = self.data.qpos[self.qi].copy(), self.data.qvel[self.vi].copy()


def make_public_backend(dt, time_fn, encoders_fn, observe_fn, step_fn, guard):
    # Every physical step passes through the guard; the guard object itself is captured only by the closure, never stored on the public object.
    def step(targets):
        check_encoders(targets)
        guard.pre_step()
        step_fn(targets)
        guard.post_step()
    return PublicBackend(dt, time_fn, encoders_fn, observe_fn, step)


def _flags(e):
    return isinstance(e, GuardViolation) or 'Numerical' in type(e).__name__


def run_episode(io, controller_run, snapshot_fn, cap, n=11, interval=0.05):
    # Latches controller, numerical, guard and cap errors; the 11-sample dwell must fit fully inside the cap (float tolerance), otherwise no dwell is taken and the episode is vetoed.
    res = {'controller_error': None, 'latched_error': None, 'time_limit': False, 'numerical_invalid': False,
           'dwell_ok': False, 'dwell_reason': '', 'history': []}
    try:
        controller_run(io)
    except Exception as e:
        res['controller_error'] = type(e).__name__ + ': ' + str(e)
    steps = int(round(interval / io.dt))
    if io.error is not None:
        res['dwell_reason'] = 'latched_before_dwell'
    elif steps < 1 or abs(steps * io.dt - interval) > 1e-9:
        res['dwell_reason'] = 'interval_not_integral'
    elif io.now() + (n - 1) * steps * io.dt > cap + 1e-9:
        res['dwell_reason'] = 'dwell_exceeds_cap'
    else:
        try:
            for k in range(n):
                if k:
                    for _ in range(steps):
                        io._step()
                res['history'].append(snapshot_fn())
            res['dwell_ok'], res['dwell_reason'] = True, 'ok'
        except Exception as e:
            res['dwell_reason'] = 'dwell_error:' + type(e).__name__
            if io.error is None:
                io.error = e
    e = io.error
    if e is not None:
        res['latched_error'] = type(e).__name__ + ': ' + str(e)
        res['time_limit'] = isinstance(e, Limit)
        res['numerical_invalid'] = _flags(e)
    return res


def vetoes(res):
    v = []
    if res['controller_error']:
        v.append('controller_error')
    if res['latched_error']:
        v.append('latched_error')
    if res['time_limit']:
        v.append('time_limit')
    if res['numerical_invalid']:
        v.append('numerical_invalid')
    if not res['dwell_ok']:
        v.append('dwell_incomplete:' + res['dwell_reason'])
    return v
