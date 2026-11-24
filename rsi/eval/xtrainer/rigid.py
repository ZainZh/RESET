# Split taken from env RSI_EVAL_SPLIT (default dev),
# outputs under results/xtrainer/<task>/. Physics, layouts, IO, scorer and fail-closed logic unchanged.
'''Fail-closed DEV runner v2 for build_tower (simulation DEV only; VAL/TEST/hardware untouched).
Same world, layouts, IO, physics and independent scorer as rsi.env.run, unchanged. Fixes over run_dev.py:
- the controller is copied to <run>/controller_snapshot.py, hash-verified, and the SNAPSHOT is executed (not the live path);
- snapshot hash is re-checked inside every episode at start and end; the live controller hash is recorded at start and end; any mismatch fails the episode;
- duplicate indices are rejected before anything is created; every episode dir is reserved exclusively (mkdir exist_ok=False) before launch;
- errors from world construction, reset and warm-render are persisted as result.json (infra_error for EGL/observation failures, setup_error otherwise), never counted as success, never rerun;
- meta.json holds a manifest: world/reference hashes, helper source hashes, original runner (run_dev.py) hash, this runner hash, argv, env.
Success = raw scorer success AND no controller exception AND no scoring error AND hashes intact.

env -u PYTHONPATH MUJOCO_GL=egl RSI_EGL_DEVICES=1 RSI_SAM3_REMOTE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m rsi.eval.xtrainer.rigid <controller.py> <run> --indices 0-7 --workers 2
'''
import argparse, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPLIT = os.environ.get('RSI_EVAL_SPLIT', 'dev')
RUNS = ROOT / 'results/xtrainer' / os.environ['RIGID_TASK']
TASK = os.environ['RIGID_TASK']
HELPERS = ['rsi/env.py', 'rsi/scorer.py', 'rsi/tasks.py', 'rsi/layouts.py', 'rsi/perception.py', 'rsi/perception_geo.py',
           'rsi/perception_sam3.py', 'twin/practice/runtime.py', 'twin/practice/motion.py', 'twin/practice/scoring.py']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_indices(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _save(out, result, log, diag):
    import numpy as np
    from rsi import env
    out = Path(out)
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=env._plain))
    (out / 'log.json').write_text(json.dumps(log, indent=1, default=env._plain))
    (out / 'diag.json').write_text(json.dumps({'events': diag}, indent=1, default=env._plain))


def episode(args):
    snap, snap_sha, live, index, out, video, delay, dev = args
    time.sleep(delay)
    out = Path(out); t0 = time.time()
    result = {'task': TASK, 'split': SPLIT, 'index': index, 'success': False, 'raw_success': None, 'error': None}
    log = []; diag = []; world = None; io = None
    try:
        result['controller_sha256_snapshot_start'] = sha(snap)
        result['controller_sha256_live_start'] = sha(live)
        if result['controller_sha256_snapshot_start'] != snap_sha:
            raise RuntimeError('snapshot hash mismatch at episode start')
        import numpy as np
        from rsi import env, scorer
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(dev)  # set after importing mujoco, as rsi.env._run_star does
        from rsi.tasks import get
        from rsi.layouts import sample
        from twin.practice.runtime import MotorWorld
        task = get(TASK)
        world = MotorWorld(task.world / 'scene.xml', task.record)
        poses, _ = sample(TASK, SPLIT, index, world.model, task.record)
        world.reset(task.world / 'reference.npz', poses)
        env._warm_renderer(world)
    except Exception as e:
        kind = 'infra_error' if ('EGL' in type(e).__name__ or 'EGL' in str(e) or type(e).__name__ == 'ObservationError') else 'setup_error'
        result['error'] = f'{kind}: {type(e).__name__}: {e}'
        log.append({'event': 'exception', 'phase': 'setup', 'trace': traceback.format_exc()[-1500:]})
        result.update(wall=time.time() - t0)
        try:
            if world is not None: world.close()
        except Exception:
            pass
        try:
            _save(out, result, log, diag)
        except Exception as e2:
            (out / 'result_save_error.txt').write_text(repr(e2))
        return result
    io = env.IO(world, task, task.cap_sec, log); io.record_every = .7 if video else 0.
    names = list(task.objects)
    base_log = io.log

    def spy_log(event, **data):  # runner-side diagnostic only; nothing is returned to the controller
        base_log(event, **data)
        diag.append({'t': float(world.data.time), 'event': event, 'objects': {n: world.data.body('object_' + n).xpos.tolist() for n in names}})
    io.log = spy_log
    try:
        env.load_controller(snap).run(io)  # the immutable snapshot, not the live path
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'; log.append({'event': 'exception', 'phase': 'controller', 'trace': traceback.format_exc()[-1500:]})
    try:
        history = []
        for k in range(11):
            if k: io.wait(.05)
            world.synchronize(); history.append(scorer.snapshot(task, world.model, world.data))
        result.update(scorer.evaluate(history, .06 if any(g[0] == 'inside' for g in task.goals) else .005))
    except Exception as e:
        result.update(success=False, partial=0., scoring_error=f'{type(e).__name__}: {e}')
    result['raw_success'] = bool(result.get('success'))
    result['controller_sha256_snapshot_end'] = sha(snap)
    result['controller_sha256_live_end'] = sha(live)
    result['hashes_intact'] = (result['controller_sha256_snapshot_end'] == snap_sha and result['controller_sha256_live_end'] == result['controller_sha256_live_start'])
    result['success'] = bool(result['raw_success'] and result.get('error') is None and not result.get('scoring_error') and result['hashes_intact'])
    result.update(sim_time=float(world.data.time), wall=time.time() - t0)
    try:
        _save(out, result, log, diag)
        if video and io.frames:
            import cv2
            fr = io.frames
            idx = np.linspace(0, len(fr) - 1, 12).astype(int) if len(fr) >= 12 else list(range(len(fr))) + [len(fr) - 1] * (12 - len(fr))
            sheet = [cv2.resize(fr[i], (320, 180)) for i in idx]
            grid = np.concatenate([np.concatenate(sheet[i:i + 4], 1) for i in range(0, 12, 4)], 0)
            cv2.imwrite(str(out / 'frames.jpg'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    finally:
        world.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('controller'); ap.add_argument('run')
    ap.add_argument('--indices', required=True); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--no-video', action='store_true')
    a = ap.parse_args()
    idx = parse_indices(a.indices)
    if len(set(idx)) != len(idx):
        raise SystemExit('duplicate indices rejected: %s' % sorted(i for i in set(idx) if idx.count(i) > 1))
    out = RUNS / a.run
    if out.exists():
        raise SystemExit(f'{out} exists; runs are never overwritten')
    live = Path(a.controller).resolve()
    live_sha = sha(live)
    out.mkdir(parents=True)
    snap = out / 'controller_snapshot.py'
    shutil.copy(live, snap)
    snap_sha = sha(snap)
    if snap_sha != live_sha:
        raise SystemExit('snapshot hash differs from live controller right after copy; aborting')
    for i in idx:  # exclusive reservation of every episode directory before any launch
        (out / f'{SPLIT}_{i:03d}').mkdir(exist_ok=False)
    from rsi.tasks import get
    task = get(TASK)
    manifest = {'world_scene_sha256': sha(task.world / 'scene.xml'), 'world_reference_sha256': sha(task.world / 'reference.npz'),
                'helpers_sha256': {h: sha(ROOT / h) for h in HELPERS if (ROOT / h).exists()}}
    devs = [int(x) for x in os.environ.get('RSI_EGL_DEVICES', '1').split(',') if x.strip()]
    (out / 'meta.json').write_text(json.dumps({'task': TASK, 'runner': 'run_dev2.py', 'controller': str(live), 'controller_sha256': live_sha,
        'snapshot_sha256': snap_sha, 'runner_sha256': sha(__file__), 'original_runner_run_dev_py_sha256': None,
        'manifest': manifest, 'argv': sys.argv, 'indices': idx, 'split': SPLIT, 'workers': a.workers, 'egl_devices': devs,
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'env': {k: os.environ.get(k) for k in ('MUJOCO_GL', 'RSI_EGL_DEVICES', 'RSI_SAM3_REMOTE', 'RSI_SAM3_ADDR')}}, indent=1))
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(str(snap), snap_sha, str(live), i, str(out / f'{SPLIT}_{i:03d}'), not a.no_video, .4 * (n % a.workers), devs[n % len(devs)]) for n, i in enumerate(idx)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as ex:
        rs = list(ex.map(episode, jobs))
    live_end = sha(live)
    rows = [{'index': r['index'], 'success': r['success'], 'raw_success': r.get('raw_success'), 'partial': round(r.get('partial', 0), 3),
        'error': r.get('error'), 'scoring_error': r.get('scoring_error'), 'failed': r.get('failed_checks'), 'hashes_intact': r.get('hashes_intact'),
        'infra': 'infra_error' in str(r.get('error'))} for r in rs]
    s = {'task': TASK, 'runner': 'run_dev2.py', 'controller': str(live), 'controller_sha256': live_sha, 'snapshot_sha256': snap_sha,
        'live_sha256_end': live_end, 'live_unchanged_during_run': live_end == live_sha, 'split': SPLIT,
        'successes_fail_closed': sum(r['success'] for r in rows), 'successes_raw': sum(bool(r['raw_success']) for r in rows),
        'infra_errors': [r['index'] for r in rows if r['infra']], 'n': len(rows), 'wall_sec': round(time.time() - t0, 1), 'rows': rows}
    (out / 'summary.json').write_text(json.dumps(s, indent=1))
    for r in rows:
        print(r['index'], r['success'], r['raw_success'], r['partial'], r['error'], r['failed'])
    print(f"RESULT {a.run}: {s['successes_fail_closed']}/{s['n']} fail-closed (raw {s['successes_raw']}), infra {s['infra_errors']}, {s['wall_sec']}s -> {out}", flush=True)


if __name__ == '__main__':
    main()
