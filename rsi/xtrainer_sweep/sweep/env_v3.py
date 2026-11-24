"""[env_v3] Runner v2 + scorer v3: identical to env.py except the scorer import. env.py stays frozen.
sweep_objects task spec, layouts and episode runner (privileged side; controllers never import this).

Reuses rsi.env.IO (the same controller interface as the frozen tasks), rsi.layouts validity checks and
twin MotorWorld. Adds a trusted per-step Monitor whose evidence only the scorer reads; the policy-facing io
object holds no reference to it (the runner keeps it in a module-private registry keyed by id(io)).

Outputs per episode (runner v2): <out_root>/<split>_<idx>/attempt_<k>/{result.json,stderr.log,episode/{result.json,log.json}}; never
overwritten. Retries happen only for infrastructure errors (EGL), into a new attempt directory; the
episode's episode.json names the attempt used and lists all attempts.
"""
import json, os, time, traceback, zlib
os.environ.setdefault('MUJOCO_GL', 'egl')
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from twin.practice.runtime import MotorWorld
from rsi import env as base
from rsi.layouts import SPLITS, TEST_ENV, demo_poses, _free_adr, _valid
from . import scorer_v3 as scorer   # v3: net cube-pan support force (the only difference vs env.py)

ROOT = Path(__file__).resolve().parents[3]
WORLD = Path(os.environ.get('SWEEP_WORLD', ROOT / 'worlds/xtrainer/sweep_objects'))


@dataclass(frozen=True)
class Task:
    name: str
    instruction: str
    movable: dict
    cap_sec: float
    objects: dict = field(default_factory=dict)
    world: Path = WORLD

    @property
    def record(self):
        return json.loads((self.world / 'reconstruction.json').read_text())


TASK = Task('sweep_objects', 'Sweep the orange block into the dustpan with the brush.',
    # dustpan is held by the (unrandomized) left gripper in the demo, so it is not randomized
    movable={'orange_cube': (.05, .05, 30), 'brush': (.04, .04, 20)}, cap_sec=40,
    objects={'orange_cube': {'kind': 'block', 'color': 'orange', 'size': [.032, .032, .032]},
        'brush': {'kind': 'hand_brush', 'color': 'orange', 'size': [.34, .03, .035]},
        'dustpan': {'kind': 'dustpan', 'color': 'grey', 'size': [.164, .26, .028]}})


def sample(split, index, model=None, record=None, task=TASK):
    if split == 'test' and os.environ.get(TEST_ENV) != '1':
        raise PermissionError('TEST split is sealed')
    model = model or mujoco.MjModel.from_xml_path(str(task.world / 'scene.xml'))
    record = record or task.record
    b0 = demo_poses(model)
    if split == 'demo':
        return {k: v.tolist() for k, v in b0.items()}, 0
    initial = np.load(task.world / 'reference.npz')['joints'][0]
    rng = np.random.default_rng([SPLITS[split], index, zlib.crc32(task.name.encode())])
    data = mujoco.MjData(model)
    for attempt in range(500):
        mujoco.mj_resetData(model, data)
        poses = {}
        for name, pose in b0.items():
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
            mujoco.mj_forward(model, data)
            if scorer.cube_outside_pan(model, data):
                return {k: v.tolist() for k, v in poses.items()}, attempt + 1
    raise RuntimeError(f'no valid layout for {task.name} {split} {index}')


_MONITORS = {}  # id(io) -> Monitor; runner-private, never an attribute of the policy-facing io


class MonitoredIO(base.IO):
    """rsi.env.IO plus a trusted monitor update after every physics step (monitor kept outside the object)."""

    def __init__(self, world, task, cap, log, monitor):
        super().__init__(world, task, cap, log)
        _MONITORS[id(self)] = monitor

    def _step(self):
        m = _MONITORS[id(self)]
        m.pre_step(self._w.model, self._w.data)
        super()._step()
        m.update(self._w.model, self._w.data)


def run(controller_path, split='dev', index=0, out=None, video=False):
    task = TASK; t0 = time.time()
    world = MotorWorld(task.world / 'scene.xml', task.record)
    poses, _ = sample(split, index, world.model, task.record)
    world.reset(task.world / 'reference.npz', poses)
    monitor = scorer.Monitor(world.model, world.data)   # legitimate state recorded immediately after reset
    base._warm_renderer(world)
    log = []; io = MonitoredIO(world, task, task.cap_sec, log, monitor); io.record_every = .7 if video else 0.
    result = {'task': task.name, 'split': split, 'index': index, 'world': str(task.world)}
    try:
        base.load_controller(controller_path).run(io)
        result['error'] = None
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'; log.append({'event': 'exception', 'trace': traceback.format_exc()[-1500:]})
    scoring_error = None
    try:
        history = []
        monitor.pre_step(world.model, world.data)   # catches writes after the controller's last step
        for k in range(11):
            if k:
                # dwell without the cap: the controller has returned; physics continues with held targets
                for _ in range(int(round(.05 / world.dt))):
                    monitor.pre_step(world.model, world.data)
                    world.step(io.targets, io.openings); monitor.update(world.model, world.data)
            world.synchronize(); history.append(scorer.snapshot(world.model, world.data))
        result.update(scorer.evaluate(history, monitor.summary(), numerical_invalid=world.numerical_invalid))
    except Exception as e:
        scoring_error = f'{type(e).__name__}: {e}'
        result.update(raw_goal_success=False, partial=0., scoring_error=scoring_error)
    ok, reasons = scorer.final_success(result, result['error'], world.numerical_invalid, scoring_error)
    result.update(success=ok, fail_reasons=reasons, numerical_invalid=bool(world.numerical_invalid),
        sim_time=float(world.data.time), wall=time.time() - t0)
    if out:
        out = Path(out)
        if out.exists():
            raise FileExistsError(f'{out} exists; attempts are never overwritten')
        out.mkdir(parents=True)
        (out / 'result.json').write_text(json.dumps(result, indent=1, default=base._plain))
        (out / 'log.json').write_text(json.dumps(log, indent=1, default=base._plain))
        if video and io.frames:
            import cv2
            idx = np.linspace(0, len(io.frames) - 1, 12).astype(int)
            sheet = [cv2.resize(io.frames[i], (320, 180)) for i in idx]
            grid = np.concatenate([np.concatenate(sheet[i:i + 4], 1) for i in range(0, 12, 4)], 0)
            cv2.imwrite(str(out / 'frames.jpg'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    _MONITORS.pop(id(io), None)
    world.close()
    return result


def _is_infra(r):
    return 'infra_error' in str(r.get('error')) or 'EGL' in str(r.get('error'))


ATTEMPT_TIMEOUT_SEC = 900.   # wall-clock bound per attempt (sim cap is 40 s; episodes take ~25-60 s wall)


def _child(controller_path, split, index, out, video, conn):
    """Episode in an isolated process; stderr goes to <attempt>/stderr.log so it survives a native crash."""
    import faulthandler, sys
    if out:
        f = open(Path(out) / 'stderr.log', 'w', buffering=1)
        os.dup2(f.fileno(), 2); sys.stderr = f
        faulthandler.enable(file=f, all_threads=True)
        out = Path(out) / 'episode'   # run() creates it; the attempt dir (reserved by the parent) holds stderr.log
    try:
        r = run(controller_path, split, index, out, video)
    except Exception as e:   # world construction/sampling failure: recorded, counted as failure
        r = {'task': TASK.name, 'split': split, 'index': index, 'success': False, 'raw_goal_success': False,
            'partial': 0., 'error': f'runner: {type(e).__name__}: {e}', 'fail_reasons': ['runner_error']}
    conn.send(r); conn.close()


def _attempt(controller_path, split, index, out, video, timeout=None):
    """One attempt in a spawned process. The parent reserves <out> (exclusive mkdir) before starting the child."""
    import multiprocessing as mp
    timeout = ATTEMPT_TIMEOUT_SEC if timeout is None else timeout
    if out:
        Path(out).mkdir(parents=True, exist_ok=False)
    ctx = mp.get_context('spawn')
    rx, tx = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(controller_path, split, index, out, video, tx)); p.start(); tx.close()
    r, timed_out = None, False
    try:
        if rx.poll(timeout):
            r = rx.recv()
        else:
            timed_out = True
    except EOFError:
        pass
    if timed_out:
        p.kill()
    p.join()
    if r is None:   # native crash, kill or timeout: fail closed, the seed stays in the denominator
        sig = -p.exitcode if p.exitcode is not None and p.exitcode < 0 else None
        kind = 'attempt_timeout' if timed_out else 'engine_crash'
        r = {'task': TASK.name, 'split': split, 'index': index, 'success': False, 'raw_goal_success': False, 'partial': 0.,
            'error': f'{kind}: child exit code {p.exitcode}' + (f' (signal {sig})' if sig else '') + (f' after {timeout:.0f} s' if timed_out else ''),
            'fail_reasons': [kind], 'signal': sig}
    r['exit_code'] = p.exitcode
    if out:
        (Path(out) / 'result.json').write_text(json.dumps(r, indent=1, default=base._plain))
    return r


def episode(controller_path, split, index, ep_dir, video=False, delay=0., max_attempts=2, timeout=None):
    """Run one episode, each attempt in its own process, in immutable attempt directories. The episode directory is
    reserved exclusively (an existing directory raises FileExistsError: nothing is overwritten). Retry only after an EGL
    infrastructure error; engine crashes and timeouts are recorded failures and are not retried."""
    time.sleep(delay)
    ep_dir = Path(ep_dir) if ep_dir else None
    if ep_dir:
        ep_dir.mkdir(parents=True, exist_ok=False)
    attempts = []
    for k in range(max_attempts):
        out = ep_dir / f'attempt_{k}' if ep_dir else None
        r = _attempt(controller_path, split, index, out, video, timeout)
        r['attempt'] = k; attempts.append({'attempt': k, 'success': r['success'], 'error': r.get('error'), 'infra': _is_infra(r),
            'exit_code': r.get('exit_code'), 'signal': r.get('signal')})
        if not _is_infra(r):
            break
    r['attempts'] = attempts; r['used_attempt'] = attempts[-1]['attempt']
    if ep_dir:
        (ep_dir / 'episode.json').write_text(json.dumps({'used_attempt': r['used_attempt'], 'attempts': attempts}, indent=1))
    return r


def batch(task_name, controller_path, split='dev', indices=range(8), out_root=None, workers=4, video=False):
    """Threads only dispatch; every attempt is its own spawned process, so one crash cannot abort the batch."""
    from concurrent.futures import ThreadPoolExecutor
    assert task_name == TASK.name
    jobs = [(str(controller_path), split, i, (Path(out_root) / f'{split}_{i:03d}') if out_root else None, video, .4 * (n % workers))
        for n, i in enumerate(indices)]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: episode(*j), jobs))
