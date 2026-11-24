# Changes (physics, IO, controllers, scorers, validity vetoes, child timeouts unchanged; frozen common files are still hashed/snapshotted from their original location):
#  1. layouts come from --layouts FILE with a --split LABEL (dev|test); indices 0..len-1; result/summary carry the split. The world's own
#     layouts.json must still equal the pin (unchanged world). DEV: --layouts SHA must equal the specs pin; TEST: must equal
#     layouts/test/DERIVE_REPORT.json for robot franka/task, and the file's source SHA must match.
#  2. episode dirs are named <split>_<iii>.
#  3. EGL on the evaluation machine: devices 0/1 (2/3 are broken Mesa nodes); slot pool size RESET_EGL_SLOTS (default 8), lock files $TMPDIR/reset_eval_franka_egl_dev{D}_s{K}.lock.
#  4. this file's own SHA and the eval layout file SHA are added to the START/END hashes; this file is copied into the snapshot.
'''Trusted episode runner for the Franka suite (simulation only, DEV only). Controllers never import this.

Usage:
  runner.py build TASK                         build worlds/franka/TASK (scene.xml, build.json, home.json, layouts.json)
  runner.py run TASK CONTROLLER OUT_ROOT INDICES [--workers N] [--devices 1,2]
Same contract as the LIFT2 suite runner (rsi/lift2_suite/common/runner.py): success = goal_success AND episode_valid;
controller exception, sim time cap, invalid physics/integrity and scoring failure veto success; one fresh child process per seed;
START/END hashes of every live input; exclusive-create artifacts; controller.py runs from an immutable snapshot copy.
Scorers: LIFT2 role-mapped generic / box_strict_v2 (robot set = whole base_link subtree) and the Franka role-mapped sweep_v3 copy.
Demo priors: source FK decode (LIFT2 priors.canonical_priors, read-only); the Franka TCP already uses the canonical axes.
Child process: memsafe.install() (numerically equivalent fit_plane, full_matrices=False) before the controller loads.
EGL: devices 1 and 2 only (local); per-episode flock on a Franka slot pool $TMPDIR/franka_egl_dev{D}_s{K}_20261109.lock (K < SLOTS).
'''
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

ME = Path(__file__).resolve()
ROOT = ME.parents[3]
HERE = ROOT / 'rsi/franka_suite/common'  # original frozen common dir
LAYOUTS = None  # set by CLI --layouts
SPLIT = None  # set by CLI --split
DERIVE_REPORT = ROOT / 'layouts/test/DERIVE_REPORT.json'
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')

from rsi.franka_suite.common import specs, franka as R, build_world as BW  # noqa: E402
from rsi.franka_suite.common import sweep_scorer_v3_franka as SW  # noqa: E402
from rsi.franka_suite.common.world import FrankaWorld, ARMS  # noqa: E402
from rsi.franka_suite.common.facade import IO, Limit, _plain, _TRUSTED, _TIMEOUT  # noqa: E402
from rsi.lift2_suite.common import scorers as S  # noqa: E402  role-mapped generic / box scorers (read-only)
from rsi.lift2_suite.common.embodiment.priors import canonical_priors  # noqa: E402  source FK decode (read-only)

WORLD_SET = os.environ.get('FRANKA_WORLD_SET', 'worlds_v2_parallel')  # 'worlds' (v1, robot x along task +x) or 'worlds_v2_parallel' (robot parallel to the board edges)
if WORLD_SET not in ('worlds', 'worlds_v2_parallel'): raise ValueError(f'unknown FRANKA_WORLD_SET {WORLD_SET}')
ALIGN = {'worlds': 'world_x', 'worlds_v2_parallel': 'board_edge'}[WORLD_SET]
WORLDS = ROOT / {'worlds_v2_parallel': 'worlds/franka'}.get(WORLD_SET, 'worlds/franka_v1')
HASHED = ['specs.py', 'runner.py', 'build_world.py', 'world.py', 'facade.py', 'kinematics.py', 'franka.py', 'memsafe.py', 'sweep_scorer_v3_franka.py']
DEP_FILES = ['rsi/perception.py', 'rsi/perception_geo.py', 'rsi/perception_sam3.py', 'rsi/env.py', 'rsi/layouts', 'twin/practice/motion.py',
    'twin/practice/layouts.py', 'twin/practice/scoring.py', 'twin/practice/runtime.py', 'rsi/scorer.py', 'rsi/tasks.py',
    'rsi/lift2_suite/common/scorers.py', 'rsi/lift2_suite/common/embodiment/priors.py', 'rsi/lift2_suite/common/embodiment/frames.py',
    'rsi/lift2_suite/common/specs.py', 'rsi/franka_suite/model/franka_robot.xml', 'rsi/franka_suite/model/franka_planning.xml',
    'rsi/franka_suite/model/meshes', 'rsi/franka_suite/model/build_robot.py']
EGL_VENDOR_JSON = ROOT / 'configs/egl/10_nvidia.json'
EGL_VALID = (0, 1)  # evaluation machine
SLOTS = int(os.environ.get('RESET_EGL_SLOTS', '8'))
_MON = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _dep_hash(p):
    p = Path(p)
    if not p.exists() and p.with_suffix('.py').is_file(): p = p.with_suffix('.py')
    if p.is_file(): return sha(p)
    if not p.is_dir(): raise RuntimeError(f'frozen dependency missing: {p}')
    h = hashlib.sha256()
    for f in sorted(x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts):
        h.update(str(f.relative_to(p)).encode() + b'\0' + sha(f).encode() + b'\0')
    return h.hexdigest()


class SetupError(RuntimeError):
    pass


def world_dir(t):
    return WORLDS / t


def build(t):
    s = specs.spec(t); wd = world_dir(t)
    info = BW.build(t, wd, s['board_xy'], align=ALIGN)
    (wd / 'home.json').write_text(json.dumps({'home_joints': np.asarray(R.home_for(t, WORLD_SET), float).tolist(),
        'definition': 'franka.home_for(task) code constant (HOME, or HOME_TASK for sweep_objects; same for every seed)'}, indent=1))
    shutil.copy2(s['layouts'], wd / 'layouts.json')
    return info


def dev_layouts(t):
    if LAYOUTS is None: raise SetupError('no --layouts file given')
    return json.loads(Path(LAYOUTS).read_text())['layouts']


def validate_setup(t):
    s = specs.spec(t); wd = world_dir(t); errs = []
    try:
        b = json.loads((wd / 'build.json').read_text())
    except Exception as e:
        raise SetupError(f'{t}: cannot read build.json in {wd}: {e}')
    if b.get('task') != t: errs.append('build task mismatch')
    if b.get('source_scene_sha256') != sha(Path(s['world']) / 'scene.xml'): errs.append('source scene hash differs from build record')
    if b.get('scene_sha256') != sha(wd / 'scene.xml'): errs.append('built scene.xml hash differs from build record')
    if b.get('robot_xml_sha256') != sha(BW.ROBOT_XML): errs.append('robot xml changed since build')
    if not (b.get('mechanics') or {}).get('ok'): errs.append('mechanics not ok')
    if list(b.get('board_xy_robot_frame', [])) != list(s['board_xy']): errs.append('station differs from specs')
    if b.get('station_align', 'world_x') != ALIGN: errs.append(f'station alignment {b.get("station_align")} != {ALIGN}')
    try:
        h = json.loads((wd / 'home.json').read_text())['home_joints']
        if not np.allclose(h, R.home_for(t, WORLD_SET), atol=0): errs.append('home.json differs from the franka.home_for(task) constant')
    except Exception as e:
        errs.append(f'home unreadable: {e}')
    try:
        if sha(wd / 'layouts.json') != s['layout_pin']: errs.append('layouts.json SHA differs from the pin')
        L = json.loads((wd / 'layouts.json').read_text())
        if L.get('task') != t or L.get('indices') != list(range(20)) or len(L['layouts']) != 20: errs.append('layouts not the DEV0-19 identity file')
    except Exception as e:
        errs.append(f'layouts unreadable: {e}')
    try:  # evaluation layout file (--layouts)
        lp = Path(LAYOUTS); E = json.loads(lp.read_text()); n = len(E.get('layouts', []))
        if E.get('task') != t or n < 1 or E.get('indices') != list(range(n)) or E.get('split') != SPLIT: errs.append(f'eval layout file is not the {SPLIT} 0..n-1 map for this task')
        if SPLIT == 'dev':
            if sha(lp) != s['layout_pin']: errs.append('DEV eval layout SHA differs from the pin')
        elif SPLIT == 'test':
            cells = [c for c in json.loads(DERIVE_REPORT.read_text())['cells'] if c['robot'] == 'franka' and c['task'] == t and (ROOT / c['setup']).resolve() == wd.resolve()]
            if len(cells) != 1 or cells[0].get('status') != 'ok': errs.append('no derived TEST layout record for this world')
            else:
                c = cells[0]
                if (ROOT / c['test_layouts']).resolve() != lp.resolve() or sha(lp) != c['test_layouts_sha256']: errs.append('TEST layout file differs from DERIVE_REPORT')
                if sha(ROOT / E['derived_from']) != E['derived_from_sha256'] or E['derived_from_sha256'] != c['test_source_sha256']: errs.append('TEST source SHA mismatch')
        else:
            errs.append(f'unknown split {SPLIT!r}')
    except Exception as e:
        errs.append(f'eval layouts unreadable: {e}')
    if errs:
        raise SetupError(f'{t}: invalid setup {wd}: ' + '; '.join(errs))
    return b


def station(t):
    return np.array(json.loads((world_dir(t) / 'build.json').read_text())['station_T_task_base'])


def load_controller(path):
    spec = importlib.util.spec_from_file_location('controller_' + hashlib.sha1(str(path).encode()).hexdigest()[:10], path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def warm(world, tries=6):
    import random
    for k in range(tries):
        try:
            world.observe(); return
        except Exception as e:
            if 'EGL' not in type(e).__name__ + str(e) and type(e).__name__ != 'ObservationError':
                raise
            world.close(); time.sleep(random.uniform(.3, 1.) * (k + 1))
    raise RuntimeError('infra_error: EGL context could not be created')


class MonitoredIO(IO):
    def _step(self):
        m, w = _MON[id(self)], _TRUSTED[id(self)]
        m.pre_step(w.model, w.data)
        super()._step()
        m.update(w.model, w.data)


def _dwell_step(world, io, monitor):
    for _ in range(int(round(.05 / world.dt))):
        if monitor: monitor.pre_step(world.model, world.data)
        world.step(io.targets, io.openings)
        if monitor: monitor.update(world.model, world.data)


def score(t, task, world, io, robot, monitor, controller_error):
    kind = specs.spec(t)['scorer']; hist = []
    if monitor: monitor.pre_step(world.model, world.data)
    for k in range(11):
        if k:
            if kind == 'sweep_v3': _dwell_step(world, io, monitor)
            else: io.wait(.05)
        world.synchronize()
        if kind == 'sweep_v3': hist.append(SW.snapshot(world.model, world.data))
        elif kind == 'box_strict_v2': hist.append(S.box_strict_snapshot(task, world.model, world.data, robot))
        else: hist.append(S.generic_snapshot(task, world.model, world.data, robot))
    if kind == 'sweep_v3':
        r = SW.evaluate(hist, monitor.summary(), numerical_invalid=world.invalid); r['goal_success'] = bool(r['raw_goal_success'])
    elif kind == 'box_strict_v2':
        r = S.box_strict_evaluate(hist, numerical_invalid=world.invalid, controller_error=controller_error); r['goal_success'] = bool(r['success'])
    else:
        r = S.evaluate(hist, .06 if any(g[0] == 'inside' for g in task.goals) else .005); r['goal_success'] = bool(r['success'])
    return hist, r


def make_world(t):
    return FrankaWorld(world_dir(t) / 'scene.xml', station(t))


def run(t, controller, index, out=None, frame_hook=None, keep_world=False, layout=None):
    validate_setup(t)
    before = (set(_MON), set(_TRUSTED), set(_TIMEOUT))
    world = make_world(t)
    try:
        return _run_world(t, world, controller, index, out, frame_hook, keep_world, layout)
    except BaseException:
        for reg, keep in zip((_MON, _TRUSTED, _TIMEOUT), before):
            for k in [k for k in reg if k not in keep]:
                reg.pop(k, None) if isinstance(reg, dict) else reg.discard(k)
        world.close(); raise


def _run_world(t, world, controller, index, out, frame_hook, keep_world, layout):
    from rsi.franka_suite.common import memsafe
    patch = memsafe.install()
    task = specs.source_task(t); t0 = time.time(); kind = specs.spec(t)['scorer']
    layout = dev_layouts(t)[index] if layout is None else layout
    robot = S.robot_bodies_of(world.model, R.BASE_BODY); S.check_roles(world.model, robot)
    world.reset(layout, json.loads((world_dir(t) / 'home.json').read_text())['home_joints'])
    monitor = SW.Monitor(world.model, world.data) if kind == 'sweep_v3' else None
    warm(world)
    log = []; cls = MonitoredIO if monitor else IO
    io = cls(world, task, task.cap_sec, log, canonical_priors(task)); io.frame_hook = frame_hook
    if monitor: _MON[id(io)] = monitor
    first = io.observe(keep=False)
    result = {'task': t, 'robot': 'franka1_tilted_umi', 'split': SPLIT, 'layouts_file': str(LAYOUTS), 'index': index, 'controller': str(controller), 'scorer': kind,
        'setup': str(world_dir(t)), 'world_set': WORLD_SET, 'process_patches': [patch]}
    try:
        load_controller(controller).run(io); result['error'] = None
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'; log.append({'event': 'exception', 'trace': traceback.format_exc()[-2500:]})
    history = []
    try:
        history, r = score(t, task, world, io, robot, monitor, result['error']); result.update(r); result['scoring_error'] = None
    except Exception as e:
        result.update(goal_success=False, partial=0., scoring_error=f'{type(e).__name__}: {e}')
    reasons = []
    if result['error']: reasons.append('timeout' if result['error'].startswith('Limit') else 'controller_exception')
    if id(io) in _TIMEOUT and 'timeout' not in reasons: reasons.append('timeout')
    if world.invalid: reasons.append('invalid_physics_or_integrity: ' + str(world.invalid_reason))
    if result['scoring_error']: reasons.append('scoring_incomplete')
    result['episode_valid'] = not reasons; result['invalid_reasons'] = reasons
    result['success'] = bool(result['goal_success'] and result['episode_valid'])
    result.update(sim_time=float(world.data.time), wall=time.time() - t0, evidence=world.evidence)
    if out:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise FileExistsError(f'episode directory not empty: {out}')
        result['hashes'] = hashes(t, controller)
        _xwrite(out / 'log.json', json.dumps(log, indent=1, default=_plain))
        _xwrite(out / 'history.json', json.dumps(history, indent=1, default=_plain))
        with open(out / 'trace.npz', 'xb') as f:
            np.savez_compressed(f, trace=np.array(io.trace), final_qpos=world.data.qpos, final_qvel=world.data.qvel)
        import cv2
        cv2.imwrite(str(out / 'first_obs_main.jpg'), cv2.cvtColor(first['rgb'], cv2.COLOR_RGB2BGR)[::2, ::2])
        if io.frames: cv2.imwrite(str(out / 'last_obs_main.jpg'), cv2.cvtColor(io.frames[-1], cv2.COLOR_RGB2BGR)[::2, ::2])
        _xwrite(out / 'result.json', json.dumps(result, indent=1, default=_plain))
    _MON.pop(id(io), None); _TRUSTED.pop(id(io), None); _TIMEOUT.discard(id(io))
    if keep_world:
        return result, world, io
    world.close()
    return result


def _xwrite(path, text):
    with open(path, 'x') as f:
        f.write(text)


def hashes(t, controller):
    s = specs.spec(t); wd = world_dir(t)
    out = {f: sha(HERE / f) for f in HASHED}
    out['reset_eval_runner.py'] = sha(ME); out['eval_layouts'] = sha(LAYOUTS)
    out.update({'scene.xml': sha(wd / 'scene.xml'), 'build.json': sha(wd / 'build.json'), 'home.json': sha(wd / 'home.json'), 'layouts.json': sha(wd / 'layouts.json'),
        'controller': sha(controller), 'source_scene.xml': sha(Path(s['world']) / 'scene.xml'), 'source_reference.npz': sha(Path(s['world']) / 'reference.npz'),
        'source_world_tree': _dep_hash(Path(s['world']))})
    for rel in DEP_FILES:
        out['dep:' + rel] = _dep_hash(ROOT / rel)
    if t == 'sweep_objects': out['dep:sweep_env_v3.py'] = sha(ROOT / 'rsi/xtrainer_sweep/sweep/env_v3.py')
    import xml.etree.ElementTree as ET
    for f in sorted({e.get('file') for e in ET.parse(wd / 'scene.xml').getroot().iter() if e.get('file')}):
        out['scene_file:' + f] = sha(f if os.path.isabs(f) else os.path.join(wd, f))
    return out


def check_device(dev):
    if isinstance(dev, bool) or not isinstance(dev, (int, np.integer)) or int(dev) not in EGL_VALID:
        raise ValueError(f'invalid EGL device {dev!r}: must be one of {EGL_VALID}')
    return int(dev)


def child_env(dev):
    dev = check_device(dev)
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', '__EGL_VENDOR_LIBRARY_DIRS', 'EGL_PLATFORM', 'MUJOCO_PLATFORM')}
    env.update(__EGL_VENDOR_LIBRARY_FILENAMES=str(EGL_VENDOR_JSON), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(dev), OMP_NUM_THREADS='2')
    env.setdefault('RSI_SAM3_REMOTE', '1'); env.setdefault('RSI_SAM3_ADDR', __import__('tempfile').gettempdir() + '/sam3.sock')
    return env


class slot_lock:
    '''Blocking exclusive flock on one free slot of the device pool; released on exit.'''
    def __init__(self, dev): self.dev = check_device(dev); self.f = None
    def __enter__(self):
        import fcntl
        while True:
            for k in range(SLOTS):
                f = open(__import__('tempfile').gettempdir() + f'/reset_eval_franka_egl_dev{self.dev}_s{k}.lock', 'a')
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); self.f = f; return self
                except BlockingIOError:
                    f.close()
            time.sleep(1.)
    def __exit__(self, *exc):
        import fcntl
        if self.f is not None:
            try: fcntl.flock(self.f, fcntl.LOCK_UN)
            finally: self.f.close(); self.f = None
        return False


def _job(args):
    t, controller, index, out, dev, timeout = args
    out = Path(out); env = child_env(dev)
    cmd = [sys.executable, str(ME), 'child', t, str(controller), str(index), str(out), '--layouts', str(LAYOUTS), '--split', SPLIT]
    status, rc, tail = 'ok', None, ''
    with slot_lock(dev):
        try:
            p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)
            rc = p.returncode; tail = (p.stderr or '')[-2000:]
            if rc != 0: status = 'crashed' if rc < 0 else 'error'
        except subprocess.TimeoutExpired as e:
            status = 'timedout'; err = e.stderr
            tail = (err.decode(errors='replace') if isinstance(err, bytes) else (err or ''))[-2000:]
    return _final_record(out, index, status, rc, tail)


def _stash_raw(out):
    rp = out / 'result.json'
    if not rp.exists(): return None
    dest = out / 'result_raw.json'; n = 1
    while dest.exists():
        dest = out / f'result_raw_{n}.json'; n += 1
    rp.rename(dest); return dest


def _final_record(out, index, status, rc, tail):
    rp = out / 'result.json'; prior = None
    if rp.exists():
        try:
            prior = json.loads(rp.read_text())
        except Exception:
            prior = None
        if isinstance(prior, dict) and status == 'ok':
            return prior
    raw = _stash_raw(out)
    r = {**(prior or {}), 'index': index, 'success': False, 'goal_success': False, 'episode_valid': False, 'child_status': status if status != 'ok' else 'no_result',
        'returncode': rc, 'infra_error': f'child {status} rc={rc}', 'stderr_tail': tail}
    if raw is not None: r['raw_result_file'] = raw.name
    _xwrite(rp, json.dumps(r, indent=1, default=_plain))
    return r


def _safe_job(a):
    try:
        return _job(a)
    except Exception as e:
        r = {'index': a[2], 'success': False, 'goal_success': False, 'episode_valid': False, 'child_status': 'supervisor_error', 'infra_error': f'{type(e).__name__}: {e}'}
        try:
            out = Path(a[3]); out.mkdir(parents=True, exist_ok=True); _stash_raw(out); _xwrite(out / 'result.json', json.dumps(r, indent=1))
        except Exception as e2:
            r['artifact_error'] = str(e2)
        return r


def batch(t, controller, out_root, indices, workers=1, devices=(1, 2), timeout=2400):
    devices = [check_device(d) for d in devices]
    from concurrent.futures import ThreadPoolExecutor
    indices = list(indices)
    nl = len(dev_layouts(t))
    if len(set(indices)) != len(indices) or any(i < 0 or i >= nl for i in indices):
        raise ValueError(f'indices must be unique {SPLIT} 0-{nl - 1}')
    validate_setup(t)
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=False); t0 = time.time()
    snap = out_root / 'snapshot'; snap.mkdir()
    for f in HASHED: shutil.copy2(HERE / f, snap / f)
    shutil.copy2(ME, snap / 'reset_eval_runner.py')
    ctrl = snap / 'controller.py'; shutil.copy2(controller, ctrl); ctrl.chmod(0o444)
    start = hashes(t, controller); start['original_controller_path'] = str(controller)
    snap_ok = all(sha(snap / f) == start[f] for f in HASHED) and sha(ctrl) == start['controller']
    _xwrite(out_root / 'START_HASHES.json', json.dumps({**start, 'snapshot_matches_live': snap_ok}, indent=1))
    if not snap_ok: raise RuntimeError('executed snapshot differs from live inputs')
    dirs = [out_root / f'{SPLIT}_{i:03d}' for i in indices]
    for d in dirs: d.mkdir(exist_ok=False)
    jobs = [(t, str(ctrl), i, str(d), devices[n % len(devices)], timeout) for n, (i, d) in enumerate(zip(indices, dirs))]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(_safe_job, jobs))
    try:
        end = hashes(t, ctrl); unchanged = all(end[k] == start[k] for k in end); end_err = None
    except Exception as e:
        end, unchanged, end_err = {}, False, f'{type(e).__name__}: {e}'
    _xwrite(out_root / 'END_HASHES.json', json.dumps({**end, 'unchanged_since_start': unchanged, **({'hash_read_error': end_err} if end_err else {})}, indent=1))
    n_success = sum(bool(r.get('success')) for r in res) if unchanged else 0
    summary = {'task': t, 'robot': 'franka1_tilted_umi', 'split': SPLIT, 'layouts_file': str(LAYOUTS), 'layouts_sha256': sha(LAYOUTS), 'timeout_s': timeout, 'world_set': WORLD_SET,
        'n_infra_error': sum(1 for r in res if r.get('child_status')), 'controller': str(controller), 'executed_snapshot': str(ctrl), 'setup': str(world_dir(t)), 'indices': indices,
        'inputs_unchanged': unchanged, 'end_hash_error': end_err, 'n': len(res), 'n_success': n_success, 'n_goal_success': sum(bool(r.get('goal_success')) for r in res),
        'n_child_failed': sum(1 for r in res if r.get('child_status')), 'wall_s': time.time() - t0, 'workers': workers, 'devices': list(devices),
        'per_seed': [{k: r.get(k) for k in ('index', 'success', 'goal_success', 'episode_valid', 'invalid_reasons', 'error', 'failed_checks', 'drift_m', 'sim_time', 'wall', 'infra_error', 'child_status')} for r in res]}
    _xwrite(out_root / 'summary.json', json.dumps(summary, indent=1, default=_plain))
    return summary


def parse(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-'); out += list(range(int(a), int(b or a) + 1))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('build').add_argument('task')
    r = sub.add_parser('run'); r.add_argument('task'); r.add_argument('controller'); r.add_argument('out_root'); r.add_argument('indices')
    r.add_argument('--workers', type=int, default=1); r.add_argument('--devices', default='1,2'); r.add_argument('--timeout', type=float, default=2400)
    c = sub.add_parser('child')
    for k in ('task', 'controller', 'index', 'out'): c.add_argument(k)
    for q in (r, c): q.add_argument('--layouts', required=True); q.add_argument('--split', required=True, choices=('dev', 'test'))
    a = ap.parse_args(); LAYOUTS = Path(a.layouts).resolve(); SPLIT = a.split
    if a.cmd == 'child':
        run(a.task, Path(a.controller), int(a.index), out=a.out); sys.exit(0)
    if a.cmd == 'build':
        info = build(a.task); print(a.task, 'built', info['scene_sha256'][:12], 'mechanics ok', info['mechanics']['ok'])
    else:
        s = batch(a.task, Path(a.controller).resolve(), a.out_root, parse(a.indices), a.workers, [int(x) for x in a.devices.split(',')], a.timeout)
        print(json.dumps({k: s[k] for k in ('n', 'n_success', 'n_goal_success', 'wall_s', 'inputs_unchanged')}))
        for p in s['per_seed']: print(p)
