# Split taken from env RSI_EVAL_SPLIT (default dev),
# outputs under results/xtrainer/<task>/. Physics, layouts, IO, scorer and fail-closed logic unchanged.
'''Fail-closed DEV runner for open_drawer_and_place_object (simulation DEV only).

Same world, layouts (rsi.layouts.sample, split dev), IO, physics and source scorer (rsi.scorer, unchanged) as rsi.env.run. Differences:
- the controller is copied into the run directory first and that snapshot is executed for every episode;
- every episode writes result.json even on reset/infra/scoring failure; a controller exception or a numerically invalid world
  vetoes success (the source-scorer result is kept as raw_original_success);
- refuses to overwrite a run directory, rejects duplicate indices, reserves every episode directory before running,
  and never overwrites an episode result.json.

env -u PYTHONPATH MUJOCO_GL=egl RSI_EGL_DEVICES=1 RSI_SAM3_REMOTE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m rsi.eval.xtrainer.drawer <controller.py> <run-name> --indices 0,1,15,16 --workers 2
'''
import argparse, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPLIT = os.environ.get('RSI_EVAL_SPLIT', 'dev')
RUNS = ROOT / 'results/xtrainer' / 'open_drawer_and_place_object'
TASK = 'open_drawer_and_place_object'


def parse_indices(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _write(out, result, log=()):
    out = Path(out)  # reserved (created empty) by main before any episode runs
    from rsi.env import _plain
    if (out / 'result.json').exists():
        raise RuntimeError(f'{out}/result.json exists; scored episodes are never overwritten')
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=_plain))
    (out / 'log.json').write_text(json.dumps(list(log), indent=1, default=_plain))


def episode(args):
    ctrl, index, out, video, delay, dev = args
    time.sleep(delay)
    result = {'task': TASK, 'split': SPLIT, 'index': index, 'success': False, 'error': None}
    log = []
    world = None
    try:
        import numpy as np
        from rsi import env, scorer
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(dev)  # after importing mujoco, as rsi.env._run_star
        from rsi.tasks import get
        from rsi.layouts import sample
        from twin.practice.runtime import MotorWorld
        task = get(TASK); t0 = time.time()
        world = MotorWorld(task.world / 'scene.xml', task.record)
        poses, _ = sample(TASK, SPLIT, index, world.model, task.record)
        world.reset(task.world / 'reference.npz', poses)
        env._warm_renderer(world)
    except Exception as e:
        result.update(error='infra_error: ' + f'{type(e).__name__}: {e}', infra=True)
        _write(out, result, [{'event': 'exception', 'trace': traceback.format_exc()[-1500:]}])
        if world is not None: world.close()
        return result
    io = env.IO(world, task, task.cap_sec, log); io.record_every = .7 if video else 0.
    try:
        env.load_controller(ctrl).run(io)
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'; log.append({'event': 'exception', 'trace': traceback.format_exc()[-1500:]})
    try:
        history = []
        for k in range(11):
            if k: io.wait(.05)
            world.synchronize(); history.append(scorer.snapshot(task, world.model, world.data))
        result.update(scorer.evaluate(history, .06 if any(g[0] == 'inside' for g in task.goals) else .005))
        result['raw_original_success'] = bool(result.get('success'))
        failed = list(result.get('failed_checks') or [])
        if result['error']:
            failed.append('controller_exception')
        if world.numerical_invalid:
            failed.append('numerical_invalid')
        if failed != list(result.get('failed_checks') or []):
            result['failed_checks'] = failed; result['success'] = False
    except Exception as e:
        result.update(success=False, raw_original_success=False, partial=0., scoring_error=f'{type(e).__name__}: {e}',
                      failed_checks=(['controller_exception'] if result['error'] else []) + ['scoring_exception'])
    result['numerical_invalid'] = bool(world.numerical_invalid)
    result['numerical_evidence'] = world.numerical_evidence
    result.update(sim_time=float(world.data.time), wall=time.time() - t0)
    frames = io.frames
    _write(out, result, log)
    if video and frames:
        import cv2
        idx = np.linspace(0, len(frames) - 1, 12).astype(int) if len(frames) >= 12 else list(range(len(frames))) + [len(frames) - 1] * (12 - len(frames))
        sheet = [cv2.resize(frames[i], (320, 180)) for i in idx]
        grid = np.concatenate([np.concatenate(sheet[i:i + 4], 1) for i in range(0, 12, 4)], 0)
        cv2.imwrite(str(Path(out) / 'frames.jpg'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    world.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('controller'); ap.add_argument('run')
    ap.add_argument('--indices', required=True); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--no-video', action='store_true')
    a = ap.parse_args()
    out = RUNS / a.run
    if out.exists():
        raise SystemExit(f'{out} exists; runs are never overwritten')
    idx = parse_indices(a.indices)
    if len(set(idx)) != len(idx):
        raise SystemExit(f'duplicate DEV indices in {a.indices}; each episode directory is used once')
    out.mkdir(parents=True)
    src = Path(a.controller).resolve()
    snap = out / 'controller_snapshot.py'
    shutil.copy(src, snap)
    sha = hashlib.sha256(snap.read_bytes()).hexdigest()
    for i in idx:
        (out / f'{SPLIT}_{i:03d}').mkdir()  # reserve; fails if it already exists
    devs = [int(x) for x in os.environ.get('RSI_EGL_DEVICES', '1').split(',') if x.strip()]
    srcdir = ROOT / 'rsi'
    (out / 'meta.json').write_text(json.dumps({'task': TASK, 'controller_source': str(src), 'controller_snapshot': str(snap),
        'controller_sha256': sha, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_scorer_sha256': hashlib.sha256((srcdir / 'scorer.py').read_bytes()).hexdigest(),
        'source_tasks_sha256': hashlib.sha256((srcdir / 'tasks.py').read_bytes()).hexdigest(),
        'source_layouts_sha256': hashlib.sha256((srcdir / 'layouts.py').read_bytes()).hexdigest(),
        'argv': sys.argv, 'indices': idx, 'split': SPLIT, 'workers': a.workers, 'egl_devices': devs,
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'env': {k: os.environ.get(k) for k in ('MUJOCO_GL', 'RSI_EGL_DEVICES', 'RSI_SAM3_REMOTE', 'RSI_SAM3_ADDR')}}, indent=1))
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(str(snap), i, out / f'{SPLIT}_{i:03d}', not a.no_video, .4 * (n % a.workers), devs[n % len(devs)]) for n, i in enumerate(idx)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as ex:
        rs = list(ex.map(episode, jobs))
    rows = [{'index': r['index'], 'success': bool(r.get('success')), 'raw_original_success': r.get('raw_original_success'),
        'partial': round(r.get('partial', 0.), 3), 'error': r.get('error'), 'scoring_error': r.get('scoring_error'),
        'failed': r.get('failed_checks'), 'infra': bool(r.get('infra'))} for r in rs]
    s = {'task': TASK, 'scorer': 'rsi.scorer (source, unchanged) + controller_exception/numerical_invalid veto', 'controller_sha256': sha,
        'split': SPLIT, 'successes': sum(r['success'] for r in rows), 'successes_raw_original': sum(bool(r['raw_original_success']) for r in rows),
        'infra_errors': [r['index'] for r in rows if r['infra']], 'n': len(rows), 'wall_sec': round(time.time() - t0, 1), 'rows': rows}
    (out / 'summary.json').write_text(json.dumps(s, indent=1))
    for r in rows:
        print(r['index'], r['success'], r['raw_original_success'], r['partial'], r['error'], r['failed'])
    print(f"RESULT {a.run}: {s['successes']}/{s['n']} (raw original {s['successes_raw_original']}), infra {s['infra_errors']}, {s['wall_sec']}s -> {out}", flush=True)


if __name__ == '__main__':
    main()
