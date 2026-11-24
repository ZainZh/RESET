"""Episode runner: one world, one layout, one controller, independent score.

Controllers get an `io` object only: observe() (RGB-D + K + camera pose + encoders),
tcp(arm), move(arm, p, R, duration), grip(arm, opening, duration), wait(s),
demo (public demo priors), and log(). No simulator state, scorer or layout.
"""
import importlib.util, json, math, os, time, traceback
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from twin.practice.runtime import MotorWorld, NumericalFailure
from twin.practice.motion import waypoint, KinematicsError
from . import scorer
from .tasks import get
from .layouts import sample


class Limit(RuntimeError):
    pass


class IO:
    def __init__(self, world, task, cap, log):
        self._w, self._cap, self._log = world, cap, log
        self.targets = world.encoders(); self.home_q = world.encoders(); self.openings = np.ones(2)
        self.instruction = task.instruction
        self.objects = task.objects
        self.demo = demo_priors(task, world)
        self.frames = []; self.record_every = 0.; self._next_frame = 0.

    def _step(self):
        if self._w.data.time > self._cap:
            raise Limit('sim time cap')
        self._w.step(self.targets, self.openings)
        if self.record_every and self._w.data.time >= self._next_frame:
            self._next_frame += self.record_every; self.frames.append(self._w.observe()['rgb'])

    def observe(self, keep=True):
        o = self._w.observe()
        if keep: self.frames.append(o['rgb'])
        return o

    def tcp(self, arm):
        i = ('left', 'right').index(arm)
        return self._w.kinematics.fk(arm, self._w.encoders()[i])

    def move(self, arm, p, R, duration=1.5):
        i = ('left', 'right').index(arm)
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
        try:
            return self._w.kinematics.ik(arm, p, R, seed, max_iterations=400)
        except KinematicsError:
            # Accept a near solution (<5 mm, <0.05 rad); otherwise the target is unreachable.
            q = np.asarray(seed, float)
            for _ in range(400):
                T = self._w.kinematics.fk(arm, q); dp = p - T[:3, 3]
                dr = Rotation.from_matrix(R @ T[:3, :3].T).as_rotvec()
                J = self._w.kinematics.jacobian(arm, q); J[3:] *= .15
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[dp, dr * .15])
                lim = self._w.kinematics.limits[('left', 'right').index(arm)]
                q = np.clip(q + dq * min(1., .15 / max(np.abs(dq).max(), 1e-12)), lim[:, 0] + 1e-6, lim[:, 1] - 1e-6)
            T = self._w.kinematics.fk(arm, q)
            if np.linalg.norm(p - T[:3, 3]) < .005 and np.linalg.norm(Rotation.from_matrix(R @ T[:3, :3].T).as_rotvec()) < .05:
                return q
            raise

    def grip(self, arm, opening, duration=.5):
        i = ('left', 'right').index(arm); s = float(self.openings[i])
        n = max(1, int(math.ceil(duration / self._w.dt)))
        for k in range(n):
            self.openings[i] = s + (opening - s) * (k + 1) / n; self._step()

    def home(self, arm, duration=2.0):
        i = ('left', 'right').index(arm)
        q0 = self.targets[i].copy(); q1 = self.home_q[i]
        n = max(1, int(math.ceil(duration / self._w.dt)))
        for k in range(n):
            self.targets[i] = q0 + (q1 - q0) * (k + 1) / n
            self._step()

    def wait(self, duration):
        for _ in range(max(1, int(math.ceil(duration / self._w.dt)))): self._step()

    def log(self, event, **data):
        self._log.append({'t': float(self._w.data.time), 'event': event, **{k: _plain(v) for k, v in data.items()}})


def _plain(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)): return v.item()
    return v


def demo_priors(task, world):
    """Public priors from the demonstration: per-arm TCP pose at each gripper close/open."""
    ref = np.load(task.world / 'reference.npz')
    t, q, g = ref['timestamps'], ref['joints'], ref['gripper']
    events = []
    for a, arm in enumerate(('left', 'right')):
        for k in range(1, len(t)):
            for kind, cond in (('close', g[k - 1, a] > .5 >= g[k, a]), ('open', g[k - 1, a] <= .5 < g[k, a])):
                if cond:
                    T = world.kinematics.fk(arm, q[k, a])
                    events.append({'arm': arm, 'kind': kind, 'time': float(t[k]), 'position': T[:3, 3], 'rotation': T[:3, :3]})
    events.sort(key=lambda e: e['time'])
    home = {arm: world.kinematics.fk(arm, q[0, a]) for a, arm in enumerate(('left', 'right'))}
    names = [str(n) for n in ref['object_names']]
    objects0 = {n: ref['object_poses'][0, i, :3, 3] for i, n in enumerate(names)}
    for e in events:  # demo object positions at the event (tracked from the real recording)
        k = int(np.argmin(np.abs(ref['camera_timestamps'] - e['time'])))
        e['objects'] = {n: ref['object_poses'][k, i, :3, 3] for i, n in enumerate(names)}
    return {'events': events, 'home': home, 'objects0': objects0}


def load_controller(path):
    spec = importlib.util.spec_from_file_location('controller_' + str(abs(hash(str(path)))), path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _warm_renderer(world, tries=8):
    """Create the EGL context before any controller code runs; retry with random backoff."""
    import random
    for k in range(tries):
        try:
            world.observe(); return
        except Exception as e:
            # invalid depth frames appear when several contexts share a GPU on rs2: treat like an EGL failure
            if 'EGL' not in type(e).__name__ and 'EGL' not in str(e) and type(e).__name__ != 'ObservationError': raise
            try:
                if world.renderer is not None: world.renderer.close()
            except Exception:
                pass
            world.renderer = None
            time.sleep(random.uniform(.3, 1.0) * (k + 1))
    raise RuntimeError('infra_error: EGL context could not be created')


def run(task_name, controller_path, split='dev', index=0, out=None, video=False):
    task = get(task_name); t0 = time.time()
    world = MotorWorld(task.world / 'scene.xml', task.record)
    poses, _ = sample(task_name, split, index, world.model, task.record)
    world.reset(task.world / 'reference.npz', poses)
    _warm_renderer(world)
    log = []; io = IO(world, task, task.cap_sec, log); io.record_every = .7 if video else 0.; result = {'task': task_name, 'split': split, 'index': index}
    try:
        load_controller(controller_path).run(io)
        result['error'] = None
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'; log.append({'event': 'exception', 'trace': traceback.format_exc()[-1500:]})
    try:
        history = []
        for k in range(11):
            if k: io.wait(.05)
            world.synchronize(); history.append(scorer.snapshot(task, world.model, world.data))
        result.update(scorer.evaluate(history, .06 if any(g[0] == 'inside' for g in task.goals) else .005))
    except Exception as e:
        result.update(success=False, partial=0., scoring_error=f'{type(e).__name__}: {e}')
    result.update(sim_time=float(world.data.time), wall=time.time() - t0)
    if out:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        (out / 'result.json').write_text(json.dumps(result, indent=1, default=_plain))
        (out / 'log.json').write_text(json.dumps(log, indent=1, default=_plain))
        if video and io.frames:
            import cv2
            idx = np.linspace(0, len(io.frames) - 1, 12).astype(int) if len(io.frames) >= 12 else list(range(len(io.frames))) + [len(io.frames) - 1] * (12 - len(io.frames))
            sheet = [cv2.resize(io.frames[i], (320, 180)) for i in idx]
            grid = np.concatenate([np.concatenate(sheet[i:i + 4], 1) for i in range(0, 12, 4)], 0)
            cv2.imwrite(str(out / 'frames.jpg'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    world.close()
    return result


def _egl_devices():
    """EGL device ids to spread workers over (RSI_EGL_DEVICES, comma separated).
    Many EGL contexts on one GPU hang or return invalid depth frames on rs2, so workers are spread."""
    return [int(x) for x in os.environ.get('RSI_EGL_DEVICES', '0,1,2,3,4').split(',') if x.strip()]


def batch(task_name, controller_path, split='dev', indices=range(8), out_root=None, workers=8, video=True):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    devs = _egl_devices()
    jobs = [(task_name, str(controller_path), split, i, (Path(out_root) / f'{split}_{i:03d}') if out_root else None, video, .4 * (n % workers), devs[n % len(devs)])
            for n, i in enumerate(indices)]
    # spawn, not fork: children forked from a parent that already imported mujoco fail with EGL_BAD_ALLOC
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
        results = list(ex.map(_run_star, jobs))
    for n, r in enumerate(results):  # infrastructure failures are not controller failures: rerun serially
        if 'infra_error' in str(r.get('error')) or 'EGL' in str(r.get('error')):
            results[n] = _run_star(jobs[n][:6] + (0., jobs[n][7])); results[n]['rerun_after_infra_error'] = True
    return results


def _run_star(args):
    time.sleep(args[6]) if len(args) > 6 else None
    if len(args) > 7: os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args[7])  # read when the renderer is created
    return run(*args[:6])
