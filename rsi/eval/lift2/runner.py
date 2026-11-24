# Changes (physics, IO, controllers, scorers, validity vetoes, child timeouts unchanged; frozen common files are still hashed/snapshotted from their original location):
#  1. layouts come from --layouts FILE with a --split LABEL (dev|test); indices 0..len-1 (DEV files: 0-19, TEST files: 0-49); result/summary carry the split.
#     DEV: SHA must equal the setup offset_manifest derived SHA (srunner rule) or the common layout pin (no manifest);
#     TEST: SHA must equal layouts/test/DERIVE_REPORT.json for this robot/task/setup, and the file's source SHA must match.
#  2. episode dirs are named <split>_<iii>.
#  3. EGL on the evaluation machine: devices 0/1 only (2/3 are broken Mesa nodes); per-device slot pool (RESET_EGL_SLOTS, default 8) of flocks
#     $TMPDIR/reset_eval_lift2_egl_dev{D}_s{K}.lock instead of one exclusive lock per device (lock wait still outside the child timeout).
#  4. this file's own SHA is added to the START/END hashes and the snapshot.
'''Trusted episode runner for the LIFT2 suite (all runnable tasks). Controllers never import this.

Usage:
  runner.py freeze TASK                       freeze source DEV0-19 layouts (identity map, no resampling for the target robot)
  runner.py home TASK                         solve the declared fixed ready pose for the task mount
  runner.py run TASK CONTROLLER|source OUT_ROOT INDICES [--workers N] [--devices 0]
result.json keeps goal_success (original scorer semantics) separate from episode_valid; success = goal_success AND episode_valid.
Validity vetoes: controller exception, sim time cap (Limit), invalid physics/integrity, scoring incomplete. sweep_objects runs the
privileged per-step Monitor on every controller step AND every dwell step. Inputs are snapshotted (immutable copies) before execution;
START/END hashes of live inputs must equal each other and the snapshot copies.
'''
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

ME = Path(__file__).resolve()
ROOT = ME.parents[3]
HERE = ROOT / 'rsi/lift2_suite/common'  # original frozen common dir (hashed + snapshotted files)
LAYOUTS = None  # set by CLI --layouts
SPLIT = None  # set by CLI --split
DERIVE_REPORT = ROOT / 'layouts/test/DERIVE_REPORT.json'
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')

from rsi.lift2_suite.common import specs, scorers as S, sweep_scorer_v3_lift2 as SW  # noqa: E402
from rsi.lift2_suite.common import build_world as BW  # noqa: E402
from rsi.lift2_suite.common.embodiment import lift2 as R  # noqa: E402
from rsi.lift2_suite.common.embodiment.world import TargetWorld, ARMS  # noqa: E402
from rsi.lift2_suite.common.embodiment.facade import IO, Limit, _plain, _TRUSTED, _TIMEOUT  # noqa: E402
from rsi.lift2_suite.common.embodiment.priors import canonical_priors  # noqa: E402  runtime demo/home priors are SOURCE CANONICAL; target_priors is a deprecated physical diagnostic
from rsi.lift2_suite.common.embodiment.kinematics import TaskKinematics  # noqa: E402
from twin.practice.motion import KinematicsError  # noqa: E402

WORLDS = ROOT / 'runs/lift2_suite_20261108/worlds'
HASHED = ['specs.py', 'scorers.py', 'sweep_scorer_v3_lift2.py', 'build_world.py', 'runner.py', 'embodiment/world.py', 'embodiment/facade.py',
    'embodiment/kinematics.py', 'embodiment/urdf_chain.py', 'embodiment/priors.py', 'embodiment/frames.py', 'embodiment/lift2.py', 'layout_pins.json']
# Frozen LIVE dependencies (executed from the live tree, NOT from the snapshot; only controller.py executes from the immutable snapshot).
# Hashed at START/END and per episode; a missing one fails closed.
DEP_FILES = ['rsi/perception.py', 'rsi/perception_geo.py', 'rsi/perception_sam3.py', 'rsi/env.py', 'rsi/layouts', 'twin/practice/motion.py',
    'twin/practice/layouts.py', 'twin/practice/scoring.py', 'twin/practice/runtime.py']
_MON = {}  # id(io) -> sweep Monitor (runner-private)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class SetupError(RuntimeError):
    pass


def _dep_hash(p):
    '''SHA-256 of a file, or of a directory tree (sorted relative path + file hash, __pycache__ ignored). Fails closed if absent.'''
    p = Path(p)
    if not p.exists() and p.with_suffix('.py').is_file(): p = p.with_suffix('.py')
    if p.is_file(): return sha(p)
    if not p.is_dir(): raise RuntimeError(f'frozen dependency missing: {p}')
    h = hashlib.sha256()
    for f in sorted(x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts):
        h.update(str(f.relative_to(p)).encode() + b'\0' + sha(f).encode() + b'\0')
    return h.hexdigest()


def world_dir(t, setup=None):
    '''Trusted per-task setup/world directory: default current v1; an explicit path selects a fixed task-owned variant (same for every seed).'''
    return Path(setup).resolve() if setup else WORLDS / f'{t}_lift2_v1'


def layouts_path(t):
    if LAYOUTS is None: raise SetupError('no --layouts file given')
    return Path(LAYOUTS).resolve()


def home_path(t, setup=None):
    return world_dir(t, setup) / 'home.json'


EGL_VENDOR_JSON = ROOT / 'configs/egl/10_nvidia.json'  # explicit NVIDIA ICD; the default Mesa llvmpipe is NOT accepted
EGL_VALID = (0, 1)  # evaluation machine: EGL 0/1 = H800s; 2/3 are broken Mesa nodes # measured MuJoCo EGL device indices; these are EGL indices, never CUDA indices
EGL_DEFAULT = 0  # this suite's worker device; 0/1 only when passed explicitly (rigid4)
EGL_LOCK = __import__('tempfile').gettempdir() + '/reset_eval_lift2_egl_dev{dev}_s{slot}.lock'
SLOTS = int(os.environ.get('RESET_EGL_SLOTS', '8'))


def check_device(dev):
    '''Return dev as an int EGL index in 0..2; anything else (bool, str, CUDA-style index, 3+) is rejected.'''
    if isinstance(dev, bool) or not isinstance(dev, (int, np.integer)) or int(dev) not in EGL_VALID:
        raise ValueError(f'invalid EGL device {dev!r}: must be one of {EGL_VALID}')
    return int(dev)


def default_devices():
    '''Default child EGL device: [2]. Explicit devices (0/1/2) are passed to batch(devices=...) / --devices; each is validated against the measured EGL indices.'''
    return [EGL_DEFAULT]


def child_env(dev):
    '''Trusted child environment, built BEFORE the child imports MuJoCo: explicit NVIDIA EGL vendor ICD, EGL platform, validated EGL device.'''
    dev = check_device(dev)
    if not EGL_VENDOR_JSON.is_file(): raise RuntimeError(f'EGL vendor json missing: {EGL_VENDOR_JSON}')
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', '__EGL_VENDOR_LIBRARY_DIRS', 'EGL_PLATFORM', 'MUJOCO_PLATFORM')}
    env.update(__EGL_VENDOR_LIBRARY_FILENAMES=str(EGL_VENDOR_JSON), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(dev), OMP_NUM_THREADS='2')
    return env


class egl_lock:
    '''Per-episode exclusive flock on the device slot, held only around the active child. Acquisition blocks here, OUTSIDE the child timeout; released in __exit__ (and by fd close).'''
    def __init__(self, dev): self.dev = check_device(dev); self.f = None
    def __enter__(self):
        import fcntl
        while True:
            for k in range(SLOTS):
                f = open(EGL_LOCK.format(dev=self.dev, slot=k), 'a')
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); self.f = f; return self
                except BlockingIOError:
                    f.close()
            time.sleep(.5)
    def __exit__(self, *exc):
        import fcntl
        if self.f is not None:
            try: fcntl.flock(self.f, fcntl.LOCK_UN)
            finally: self.f.close(); self.f = None
        return False


def _xwrite(path, text):
    with open(path, 'x') as f:
        f.write(text)


def freeze_layouts(t, n=20):
    specs.runnable(t)
    if t == 'sweep_objects':
        from rsi.xtrainer_sweep.sweep import env_v3
        gen = 'env_v3.sample(dev, i) on the SOURCE xTrainer sweep scene'; out = [env_v3.sample('dev', i)[0] for i in range(n)]
    else:
        from rsi.layouts import sample
        gen = 'rsi.layouts.sample(task, dev, i) on the SOURCE xTrainer scene'; out = [sample(t, 'dev', i)[0] for i in range(n)]
    _xwrite(layouts_path(t), json.dumps({'task': t, 'split': 'dev', 'indices': list(range(n)), 'transform': 'identity (task frame = source world)', 'generator': gen, 'layouts': out}, indent=1))
    return out


def dev_layouts(t):
    return json.loads(layouts_path(t).read_text())['layouts']


def layout_pins():
    '''Accepted original DEV0-19 layout-file SHA pins (common/layout_pins.json, derived from the independent initial-contact audit). Fails closed if unreadable.'''
    try:
        return json.loads((HERE / 'layout_pins.json').read_text())['pins']
    except Exception as e:
        raise SetupError(f'layout pin manifest unreadable: {e}')


def validate_setup(t, setup=None, layouts=None):
    '''Mandatory gate before any scored episode: build record task/source identity, source and built scene hashes, mechanics.ok with 0 diffs,
    frozen original DEV0-19 layout identity. Rejects an already-existing bad or changed build. No per-seed relocation exists.'''
    spec = specs.runnable(t); wd = world_dir(t, setup); errs = []
    try:
        b = json.loads((wd / 'build.json').read_text())
    except Exception as e:
        raise SetupError(f'{t}: cannot read build.json in {wd}: {e}')
    if b.get('task') != t: errs.append(f"build task {b.get('task')!r} != {t!r}")
    src = Path(spec['world']) / 'scene.xml'
    if b.get('source_scene_sha256') != sha(src): errs.append('source scene hash differs from build record')
    if not (wd / 'scene.xml').is_file() or b.get('scene_sha256') != sha(wd / 'scene.xml'): errs.append('built scene.xml missing or hash differs from build record')
    mech = b.get('mechanics') or {}
    if mech.get('ok') is not True or mech.get('n_diffs') != 0: errs.append(f"mechanics not ok: {mech.get('n_diffs')} diffs {mech.get('diffs', [])[:5]}")
    if not (wd / 'home.json').is_file(): errs.append('home.json missing')
    try:
        lp = Path(layouts) if layouts else layouts_path(t); L = json.loads(lp.read_text()); n = len(L.get('layouts', []))
        if L.get('task') != t or n < 1 or L.get('indices') != list(range(n)) or L.get('split') != SPLIT or not str(L.get('transform', '')).startswith('identity'):
            errs.append(f'layout file is not the identity {SPLIT} 0..n-1 map for this task')
    except Exception as e:
        errs.append(f'layouts unreadable: {e}')
    try:  # content pin: a changed pose with intact metadata must not pass
        errs += _layout_pin_errors(t, wd, lp)
    except Exception as e:
        errs.append(f'layout pin check failed: {e}')
    if errs:
        raise SetupError(f'{t}: invalid setup {wd}: ' + '; '.join(errs))
    return b


def _layout_pin_errors(t, wd, lp):
    '''DEV: srunner rule (offset_manifest derived SHA whose source SHA is the accepted pin) or the common pin. TEST: DERIVE_REPORT entry.'''
    if SPLIT == 'dev':
        if (wd / 'offset_manifest.json').is_file():
            man = json.loads((wd / 'offset_manifest.json').read_text()); orig = layout_pins().get(t) or sha(ROOT / man['source_layouts'])
            if man.get('task') != t or man.get('source_layouts_sha256') != orig: return [f'{t}: offset manifest does not derive from the accepted original layout pin']
            return [] if sha(lp) == man.get('derived_layouts_sha256') else ['DEV layout file SHA differs from the setup offset manifest']
        pin = layout_pins().get(t)
        return [] if pin and sha(lp) == pin else ['DEV layout file SHA != accepted original pin']
    if SPLIT == 'test':
        cells = [c for c in json.loads(DERIVE_REPORT.read_text())['cells'] if c['robot'] == 'lift2' and c['task'] == t and (ROOT / c['setup']).resolve() == wd.resolve()]
        if len(cells) != 1 or cells[0].get('status') != 'ok': return ['no derived TEST layout record for this setup']
        c = cells[0]; L = json.loads(lp.read_text()); e = []
        if (ROOT / c['test_layouts']).resolve() != lp.resolve() or sha(lp) != c['test_layouts_sha256']: e.append('TEST layout file differs from DERIVE_REPORT')
        if sha(ROOT / L['derived_from']) != L['derived_from_sha256'] or L['derived_from_sha256'] != c['test_source_sha256']: e.append('TEST source file SHA mismatch')
        return e
    return [f'unknown split {SPLIT!r}']


def mount(t, setup=None):
    return np.array(json.loads((world_dir(t, setup) / 'build.json').read_text())['mount_T_task_base'])


def solve_home(t):
    '''Declared ready pose (not a source angle): per arm TCP at base (0.30, +-0.22, 0.12) m, approach down, closing along base y.'''
    T = mount(t); kin = TaskKinematics(R.URDF, T, R.ARMS)
    Rb = np.stack([[0, 0, -1.], [0, 1., 0], [1., 0, 0]], 1); home = []
    for arm, side in (('left', 1), ('right', -1)):
        p = T[:3, :3] @ [.30, side * .22, .12] + T[:3, 3]
        for seed in ([0, 1., 1., 0, 0, 0], [0, .5, 1.5, 0, 1., 0], [0, 1.5, 2., 0, .5, 0], [0, 1., 1.5, 0, -1., 0]):
            try:
                home.append(kin.ik(arm, p, T[:3, :3] @ Rb, np.array(seed, float), 400)); break
            except KinematicsError:
                continue
        else:
            raise RuntimeError('home IK failed')
    _xwrite(home_path(t), json.dumps({'home_joints': np.array(home).tolist(), 'definition': solve_home.__doc__}, indent=1))


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
    '''IO plus the privileged sweep Monitor around every physics step (monitor lives in the runner, not on io).'''

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
    '''11 snapshots over 0.5 s. Per-task source dwell contract: generic_rsi and box_strict_v2 dwell via io.wait(.05) INSIDE the time cap;
    sweep_v3 dwells cap-free with held targets and the privileged Monitor on every step.'''
    kind = specs.spec(t)['scorer']; hist = []
    if monitor: monitor.pre_step(world.model, world.data)  # catches writes after the controller's last step
    for k in range(11):
        if k:
            if kind == 'sweep_v3': _dwell_step(world, io, monitor)  # source env_v3: cap-free dwell with held controls
            else: io.wait(.05)  # source rsi.env / strict box runner: dwell is INSIDE the task time cap (Limit -> scoring_incomplete)
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


def run(t, controller, index, out=None, frame_hook=None, keep_world=False, layout=None, setup=None):
    validate_setup(t, setup)
    before = (set(_MON), set(_TRUSTED), set(_TIMEOUT))
    world = TargetWorld(world_dir(t, setup) / 'scene.xml', R, mount(t, setup))
    try:
        return _run_world(t, world, controller, index, out, frame_hook, keep_world, layout, setup)
    except BaseException:
        for reg, keep in zip((_MON, _TRUSTED, _TIMEOUT), before):  # no stale id(io) entries after an exception
            for k in [k for k in reg if k not in keep]:
                reg.pop(k, None) if isinstance(reg, dict) else reg.discard(k)
        world.close(); raise


def _run_world(t, world, controller, index, out, frame_hook, keep_world, layout, setup=None):
    task = specs.source_task(t); t0 = time.time(); kind = specs.spec(t)['scorer']
    layout = dev_layouts(t)[index] if layout is None else layout  # override only for self-tests
    robot = S.robot_bodies_of(world.model, R.BASE_BODY); S.check_roles(world.model, robot)
    world.reset(layout, json.loads(home_path(t, setup).read_text())['home_joints'])
    monitor = SW.Monitor(world.model, world.data) if kind == 'sweep_v3' else None
    warm(world)
    log = []; cls = MonitoredIO if monitor else IO
    io = cls(world, task, task.cap_sec, log, canonical_priors(task)); io.frame_hook = frame_hook
    if monitor: _MON[id(io)] = monitor
    first = io.observe(keep=False)
    result = {'task': t, 'robot': 'lift2', 'split': SPLIT, 'layouts_file': str(layouts_path(t)), 'index': index, 'controller': str(controller), 'scorer': kind, 'setup': str(world_dir(t, setup))}
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
    if id(io) in _TIMEOUT and 'timeout' not in reasons: reasons.append('timeout')  # trusted latch: a controller that swallowed Limit still cannot be valid
    if world.invalid: reasons.append('invalid_physics_or_integrity: ' + str(world.invalid_reason))
    if result['scoring_error']: reasons.append('scoring_incomplete')
    result['episode_valid'] = not reasons; result['invalid_reasons'] = reasons
    result['success'] = bool(result['goal_success'] and result['episode_valid'])
    result.update(sim_time=float(world.data.time), wall=time.time() - t0, evidence=world.evidence)
    if out:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise FileExistsError(f'episode directory not empty: {out}')
        result['hashes'] = hashes(t, controller, setup)
        _xwrite(out / 'log.json', json.dumps(log, indent=1, default=_plain))
        _xwrite(out / 'history.json', json.dumps(history, indent=1, default=_plain))
        with open(out / 'trace.npz', 'xb') as f:
            np.savez_compressed(f, trace=np.array(io.trace), final_qpos=world.data.qpos, final_qvel=world.data.qvel)
        import cv2
        cv2.imwrite(str(out / 'first_obs_head.jpg'), cv2.cvtColor(first['rgb'], cv2.COLOR_RGB2BGR))
        if io.frames: cv2.imwrite(str(out / 'last_obs_head.jpg'), cv2.cvtColor(io.frames[-1], cv2.COLOR_RGB2BGR))
        _xwrite(out / 'result.json', json.dumps(result, indent=1, default=_plain))  # written last, exclusive create
    _MON.pop(id(io), None); _TRUSTED.pop(id(io), None); _TIMEOUT.discard(id(io))
    if keep_world:
        return result, world, io
    world.close()
    return result


def hashes(t, controller, setup=None):
    s = specs.spec(t); wd = world_dir(t, setup)
    out = {f: sha(HERE / f) for f in HASHED}
    out['reset_eval_runner.py'] = sha(ME)
    out.update({'scene.xml': sha(wd / 'scene.xml'), 'build.json': sha(wd / 'build.json'), 'home.json': sha(home_path(t, setup)), 'layouts': sha(layouts_path(t)),
        'controller': sha(controller), 'source_scene.xml': sha(Path(s['world']) / 'scene.xml'), 'source_reference.npz': sha(Path(s['world']) / 'reference.npz'),
        'rsi/scorer.py': sha(ROOT / 'rsi/scorer.py'), 'rsi/tasks.py': sha(ROOT / 'rsi/tasks.py')})
    for rel in DEP_FILES:
        out['dep:' + rel] = _dep_hash(ROOT / rel)
    out['model:lift2'] = _dep_hash(R.MODEL)  # patched.urdf, scene.xml, all robot meshes
    out['source_reconstruction.json'] = sha(Path(s['world']) / 'reconstruction.json')
    out['source_world_tree'] = _dep_hash(Path(s['world']))  # source scene, reference.npz, task meshes/assets
    out['built_world_tree'] = _dep_hash(wd)
    out['pins:source_controllers_director.json'] = sha(ROOT / 'rsi/lift2_suite/source_controllers.json')
    for rel in ('rsi/__init__.py', 'twin/__init__.py', 'twin/practice/__init__.py'):
        if (ROOT / rel).is_file(): out['dep:' + rel] = sha(ROOT / rel)
    if t == 'sweep_objects': out['dep:sweep_env_v3.py'] = sha(ROOT / 'rsi/xtrainer_sweep/sweep/env_v3.py')
    import xml.etree.ElementTree as ET
    for f in sorted({e.get('file') for e in ET.parse(wd / 'scene.xml').getroot().iter() if e.get('file')}):
        out['scene_file:' + f] = sha(f if os.path.isabs(f) else os.path.join(wd, f))  # every mesh/texture the built scene loads (fails closed if missing)
    if t == 'lift_box':  # source b8 imports this xTrainer-only static helper: literal baseline preserved and disclosed, adapted branch needs its own LIFT checker
        out['dep:xtrainer_robot_collision.py'] = sha(ROOT / 'rsi/xtrainer_lift_box/robot_collision.py')
    return out


def snapshot_inputs(t, controller, out_root, setup=None):
    snap = out_root / 'snapshot'; snap.mkdir(); wd = world_dir(t, setup)
    for f in HASHED:
        (snap / f).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(HERE / f, snap / f)
    for src, name in ((wd / 'scene.xml', 'scene.xml'), (wd / 'build.json', 'build.json'), (home_path(t, setup), 'home.json'), (layouts_path(t), 'layouts.json')):
        shutil.copy2(src, snap / name)
    shutil.copy2(ME, snap / 'reset_eval_runner.py')
    ctrl = snap / 'controller.py'; shutil.copy2(controller, ctrl); ctrl.chmod(0o444)
    return ctrl


def _job(args):
    # One fresh subprocess per seed. Trusted env (NVIDIA EGL vendor, MUJOCO_GL/PYOPENGL_PLATFORM=egl, validated device) is built BEFORE the child imports MuJoCo.
    # The device flock is taken first, so lock wait is never charged to the child timeout; it is held only while the child runs.
    t, controller, index, out, dev, setup, timeout = args
    dev = check_device(dev); out = Path(out); env = child_env(dev)
    cmd = [sys.executable, str(ME), 'child', t, str(controller), str(index), str(out), str(setup) if setup else '-', '--layouts', str(layouts_path(t)), '--split', SPLIT]
    status, rc, tail = 'ok', None, ''
    with egl_lock(dev):
        try:
            p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)
            rc = p.returncode; tail = (p.stderr or '')[-2000:]
            if rc != 0:
                status = 'crashed' if rc < 0 else 'error'
        except subprocess.TimeoutExpired as e:
            status = 'timedout'; err = e.stderr
            tail = (err.decode(errors='replace') if isinstance(err, bytes) else (err or ''))[-2000:]
    return _final_record(out, index, status, rc, tail)


def _stash_raw(out):
    # Move whatever the child left as result.json aside, byte-for-byte, so the canonical record can replace it. Never overwrites an earlier raw file.
    rp = out / 'result.json'
    if not rp.exists():
        return None
    try:
        json.loads(rp.read_bytes()); stem, suf = 'result_raw', '.json'
    except Exception:
        stem, suf = 'result_raw', '.txt'
    dest = out / (stem + suf); n = 1
    while dest.exists():
        dest = out / f'{stem}_{n}{suf}'; n += 1
    rp.rename(dest)
    return dest


def _final_record(out, index, status, rc, tail):
    # Canonical per-seed record: the returned dict and result.json on disk are the same record. A child result that cannot count (child failed after writing it, or it is unreadable) is kept in a raw file and replaced by a failing record.
    rp = out / 'result.json'; prior, unreadable = None, None
    if rp.exists():
        try:
            prior = json.loads(rp.read_text())
            if not isinstance(prior, dict):
                unreadable, prior = 'NotAnObject', None
        except Exception as e:
            unreadable = type(e).__name__
        if prior is not None and status == 'ok':
            return prior
    raw = _stash_raw(out)
    if prior is not None:
        r = {**prior, 'raw_result': dict(prior), 'raw_result_file': raw.name, 'success': False, 'goal_success': False, 'episode_valid': False, 'child_status': status, 'returncode': rc,
            'invalid_reasons': list(prior.get('invalid_reasons') or []) + ['child_' + status]}
    else:
        if unreadable:
            if status == 'ok': status = 'bad_result_json'
            tail = (tail + f' unreadable result.json: {unreadable}')[-2000:]
        r = {'index': index, 'success': False, 'goal_success': False, 'episode_valid': False, 'child_status': status if status != 'ok' else 'no_result',
            'returncode': rc, 'infra_error': f'child {status} rc={rc}', 'stderr_tail': tail}
        if raw is not None:
            r['raw_result_file'] = raw.name
    _xwrite(rp, json.dumps(r, indent=1, default=_plain))
    return r


def _safe_job(a):
    try:
        return _job(a)
    except Exception as e:  # one seed can never abort the batch
        r = {'index': a[2], 'success': False, 'goal_success': False, 'episode_valid': False, 'child_status': 'supervisor_error', 'infra_error': f'{type(e).__name__}: {e}'}
        try:  # per-seed artifact: same record on disk, any earlier result.json kept raw
            out = Path(a[3]); out.mkdir(parents=True, exist_ok=True)
            raw = _stash_raw(out)
            if raw is not None:
                r['raw_result_file'] = raw.name
            _xwrite(out / 'result.json', json.dumps(r, indent=1))
        except Exception as e2:
            r['artifact_error'] = f'{type(e2).__name__}: {e2}'
        return r


def _invalidate_final(out, index, rec, reason, detail=None):
    # END dependency mismatch or END hash read failure: the seed cannot count. The prior record is kept raw (result_raw*.json, nested as raw_result); result.json and the returned record are the same failing record.
    out = Path(out); rp = out / 'result.json'; base = None
    try:
        base = json.loads(rp.read_text()) if rp.exists() else None
    except Exception:
        base = None
    if not isinstance(base, dict):
        base = dict(rec) if isinstance(rec, dict) else {}
    raw, err = None, None
    try:
        raw = _stash_raw(out)
    except Exception as e:
        err = f'{type(e).__name__}: {e}'
    orig = base['raw_result'] if isinstance(base.get('raw_result'), dict) else dict(base)  # an earlier child-failure record already nests the original child result
    r = {**base, 'index': index, 'raw_result': orig, 'raw_success': bool(orig.get('success')), 'success': False, 'goal_success': False, 'episode_valid': False,
        'invalid_reasons': list(base.get('invalid_reasons') or []) + [reason]}
    if base.get('raw_result_file'):  # original child result stays the raw reference; the stashed canonical record is kept beside it
        r['earlier_raw_result_file'] = base['raw_result_file']
        if raw is not None:
            r['prior_record_file'] = raw.name
    elif raw is not None:
        r['raw_result_file'] = raw.name
    if reason == 'inputs_changed':
        r['inputs_changed'] = True
    else:
        r['hash_read_error'] = detail
    if err:
        r['artifact_error'] = err
    try:
        _xwrite(rp, json.dumps(r, indent=1, default=_plain))
    except Exception as e:
        r['artifact_error'] = (r.get('artifact_error') or '') + f' write: {type(e).__name__}: {e}'
    return r


def batch(t, controller, out_root, indices, workers=1, devices=None, setup=None, timeout=1800):
    devices = [check_device(d) for d in (devices if devices else default_devices())]  # reject invalid EGL devices before any output
    from concurrent.futures import ThreadPoolExecutor
    indices = list(indices)
    nl = len(json.loads(layouts_path(t).read_text())['layouts'])
    if len(set(indices)) != len(indices) or any(i < 0 or i >= nl for i in indices):
        raise ValueError(f'indices must be unique {SPLIT} 0-{nl - 1}')
    validate_setup(t, setup)  # mandatory gate: no scored episode on an invalid or changed world
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=False); t0 = time.time()
    ctrl = snapshot_inputs(t, controller, out_root, setup)
    start = hashes(t, controller, setup); start['original_controller_path'] = str(controller)
    snap_ok = all(sha(out_root / 'snapshot' / f) == start[f] for f in HASHED) and sha(ctrl) == start['controller']
    _xwrite(out_root / 'START_HASHES.json', json.dumps({**start, 'snapshot_matches_live': snap_ok}, indent=1))
    if not snap_ok:
        raise RuntimeError('executed snapshot differs from live inputs')
    dirs = [out_root / f'{SPLIT}_{i:03d}' for i in indices]
    for d in dirs: d.mkdir(exist_ok=False)
    jobs = [(t, str(ctrl), i, str(d), devices[n % len(devices)], setup, timeout) for n, (i, d) in enumerate(zip(indices, dirs))]
    with ThreadPoolExecutor(workers) as ex:  # threads only supervise child processes
        res = list(ex.map(_safe_job, jobs))
    end_err = None
    try:
        end = hashes(t, ctrl, setup); unchanged = all(end[k] == start[k] for k in end)
    except Exception as e:  # END hash read failure is an invalid batch, never a pass; END_HASHES and summary are still saved
        end, unchanged, end_err = {}, False, f'{type(e).__name__}: {e}'
    _xwrite(out_root / 'END_HASHES.json', json.dumps({**end, 'unchanged_since_start': unchanged, **({'hash_read_error': end_err} if end_err else {})}, indent=1))
    reason = None if unchanged else ('hash_read_error' if end_err else 'inputs_changed')
    if reason:  # every intended seed is finalized invalid on disk and in the summary; prior records stay raw
        res = [_invalidate_final(d, i, r, reason, end_err) for d, i, r in zip(dirs, indices, res)]
    summary = {'task': t, 'robot': 'lift2', 'split': SPLIT, 'layouts_file': str(layouts_path(t)), 'layouts_sha256': sha(layouts_path(t)), 'timeout_s': timeout,
        'n_infra_error': sum(1 for r in res if r.get('child_status')), 'controller': str(controller), 'executed_snapshot': str(ctrl), 'setup': str(world_dir(t, setup)), 'indices': indices, 'inputs_unchanged': unchanged,
        'n': len(res), 'n_intended': len(indices), 'invalidated_reason': reason, 'end_hash_error': end_err, 'n_success': sum(bool(r.get('success')) for r in res), 'n_goal_success': sum(bool(r.get('goal_success')) for r in res),
        'n_child_failed': sum(1 for r in res if r.get('child_status')), 'wall_s': time.time() - t0, 'workers': workers, 'devices': list(devices),
        'per_seed': [{k: r.get(k) for k in ('index', 'success', 'goal_success', 'episode_valid', 'invalid_reasons', 'error', 'failed_checks', 'drift_m', 'sim_time', 'wall', 'infra_error', 'child_status', 'returncode', 'raw_result_file', 'inputs_changed', 'hash_read_error')} for r in res]}
    _xwrite(out_root / 'summary.json', json.dumps(summary, indent=1, default=_plain))
    return summary


def parse(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-'); out += list(range(int(a), int(b or a) + 1))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    for c in ('freeze', 'home'): sub.add_parser(c).add_argument('task')
    r = sub.add_parser('run'); r.add_argument('task'); r.add_argument('controller'); r.add_argument('out_root'); r.add_argument('indices')
    r.add_argument('--workers', type=int, default=1); r.add_argument('--devices', default=None); r.add_argument('--setup', default=None); r.add_argument('--timeout', type=float, default=1800)
    c = sub.add_parser('child')
    for k in ('task', 'controller', 'index', 'out', 'setup'): c.add_argument(k)
    for q in (r, c): q.add_argument('--layouts', required=True); q.add_argument('--split', required=True, choices=('dev', 'test'))
    a = ap.parse_args(); LAYOUTS = Path(a.layouts).resolve(); SPLIT = a.split
    if a.cmd == 'child':
        run(a.task, Path(a.controller), int(a.index), out=a.out, setup=None if a.setup == '-' else a.setup); sys.exit(0)
    if a.cmd == 'freeze': print(len(freeze_layouts(a.task)), 'layouts frozen ->', layouts_path(a.task))
    elif a.cmd == 'home': solve_home(a.task); print('home ->', home_path(a.task))
    else:
        ctrl = specs.resolve_source_controller(a.task)[0] if a.controller == 'source' else Path(a.controller).resolve()
        s = batch(a.task, ctrl, a.out_root, parse(a.indices), a.workers, [int(x) for x in a.devices.split(',')] if a.devices else None, a.setup, a.timeout)
        print(json.dumps({k: s[k] for k in ('n', 'n_success', 'n_goal_success', 'wall_s')}))
        for p in s['per_seed']: print(p)
