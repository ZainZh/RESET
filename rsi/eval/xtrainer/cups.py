# Split taken from env RSI_EVAL_SPLIT (default dev),
# outputs under results/xtrainer/<task>/. Physics, layouts, IO, scorer and fail-closed logic unchanged.
'''Fail-closed DEV runner for the stack_cups improvement worker (simulation DEV only), revision rev2.

Same episode procedure as rsi.env.run (same world, layouts, IO, independent scorer, all unchanged), plus:
- the controller is copied into the run directory first, hash-verified, and that SNAPSHOT is what every episode executes;
- duplicate indices are rejected before anything is created; every episode directory is reserved exclusively;
- result.json is written for every episode, including world-construction/warm-render failures and worker crashes,
  and is never overwritten;
- success = raw scorer success AND no controller exception AND no scoring error AND no numerically invalid world;
- meta.json holds a manifest of controller/runner/scorer/world/layout/helper hashes;
- post-episode diagnostics (scorer margins, object positions at log events) are read by the runner only.

env -u PYTHONPATH MUJOCO_GL=egl RSI_EGL_DEVICES=1 RSI_SAM3_REMOTE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m rsi.eval.xtrainer.cups <controller.py> <run> --indices 0-95 --workers 2
'''
import argparse, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPLIT = os.environ.get('RSI_EVAL_SPLIT', 'dev')
RUNS = ROOT / 'results/xtrainer' / 'stack_cups'
TASK = 'stack_cups'
REVISION = 'rev2_snapshot_exec_manifest'
HASHED = ['rsi/scorer.py', 'rsi/env.py', 'rsi/layouts.py', 'rsi/tasks.py', 'rsi/perception.py', 'rsi/perception_geo.py',
          'rsi/perception_sam3.py', 'twin/practice/scoring.py', 'twin/practice/runtime.py', 'twin/practice/layouts.py',
          'worlds/xtrainer/stack_cups/scene.xml', 'worlds/xtrainer/stack_cups/reference.npz',
          'worlds/xtrainer/stack_cups/reconstruction.json']


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_indices(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _write(out, result, log=(), diag=None):
    from rsi.env import _plain
    out = Path(out)  # reserved (created empty) by main before any episode runs
    if (out / 'result.json').exists():
        raise RuntimeError(f'{out}/result.json exists; scored episodes are never overwritten')
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=_plain))
    (out / 'log.json').write_text(json.dumps(list(log), indent=1, default=_plain))
    if diag is not None:
        (out / 'diag.json').write_text(json.dumps(diag, indent=1, default=_plain))


def episode(args):
    ctrl, want, index, out, video, delay, dev = args
    time.sleep(delay)
    result = {'task': TASK, 'split': SPLIT, 'index': index, 'success': False, 'raw_success': None, 'error': None,
              'runner_revision': REVISION}
    log = []; diag = []; world = None
    try:
        if sha256(ctrl) != want:
            raise RuntimeError('controller snapshot hash mismatch')
        import numpy as np
        from rsi import env, scorer
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(dev)  # set after importing mujoco, as rsi.env._run_star does
        from rsi.tasks import get
        from rsi.layouts import sample
        from twin.practice.runtime import MotorWorld
        from twin.practice.layouts import world_vertices
        from twin.practice.scoring import _inside_profile
        task = get(TASK); t0 = time.time()
        world = MotorWorld(task.world / 'scene.xml', task.record)
        poses, _ = sample(TASK, SPLIT, index, world.model, task.record)
        world.reset(task.world / 'reference.npz', poses)
        env._warm_renderer(world)
    except Exception as e:
        result.update(error='infra_error: setup ' + f'{type(e).__name__}: {e}', infra=True)
        _write(out, result, [{'event': 'exception', 'trace': traceback.format_exc()[-1500:]}])
        if world is not None:
            try: world.close()
            except Exception: pass
        return result
    io = env.IO(world, task, task.cap_sec, log); io.record_every = .7 if video else 0.
    names = list(task.objects)
    base_log = io.log

    def spy_log(event, **data):  # runner-side diagnostic only; the controller gets nothing back
        base_log(event, **data)
        diag.append({'t': float(world.data.time), 'event': event,
                     'objects': {n: world.data.body('object_' + n).xpos.tolist() for n in names}})
    io.log = spy_log
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
    except Exception as e:
        result.update(success=False, partial=0., scoring_error=f'{type(e).__name__}: {e}')
    result['raw_success'] = bool(result.get('success'))
    result['numerical_invalid'] = bool(getattr(world, 'numerical_invalid', False))
    result['success'] = bool(result['raw_success'] and result.get('error') is None and not result.get('scoring_error')
                             and not result['numerical_invalid'])
    margins = []
    try:
        model, data = world.model, world.data
        for kind, child, parent in task.goals:
            if kind not in ('inside', 'nested'):
                continue
            sup = data.body('object_' + parent); cb = data.body('object_' + child)
            local = (world_vertices(model, data, 'object_' + child) - sup.xpos) @ sup.xmat.reshape(3, 3)
            ok, m = _inside_profile(local, task.record['container_collision_geometry'][parent], scorer.TOL, partial=kind == 'nested')
            margins.append({'goal': f'{kind}:{child}:{parent}', 'ok': bool(ok), 'metrics': m,
                            'xy_offset_m': float(np.linalg.norm(cb.xpos[:2] - sup.xpos[:2])),
                            'child_z': float(cb.xpos[2]),
                            'parent_tilt_deg': float(np.degrees(np.arccos(np.clip(sup.xmat.reshape(3, 3)[2, 2], -1, 1))))})
    except Exception as e:
        margins.append({'margin_error': f'{type(e).__name__}: {e}'})
    result.update(sim_time=float(world.data.time), wall=time.time() - t0)
    frames = io.frames
    _write(out, result, log, {'margins': margins, 'events': diag})
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
    idx = parse_indices(a.indices)
    if len(set(idx)) != len(idx):
        raise SystemExit(f'duplicate DEV indices in {a.indices}; each episode directory is used once')
    out = RUNS / a.run
    if out.exists():
        raise SystemExit(f'{out} exists; runs are never overwritten')
    out.mkdir(parents=True)
    src = Path(a.controller).resolve()
    snap = out / 'controller_snapshot.py'
    shutil.copy(src, snap)
    sha = sha256(snap)
    if sha != sha256(src):
        raise SystemExit('snapshot hash differs from source right after copy; aborting')
    for i in idx:
        (out / f'{SPLIT}_{i:03d}').mkdir()  # reserve; fails if it already exists
    devs = [int(x) for x in os.environ.get('RSI_EGL_DEVICES', '1').split(',') if x.strip()]
    manifest = {p: sha256(ROOT / p) for p in HASHED if (ROOT / p).exists()}
    (out / 'meta.json').write_text(json.dumps({'task': TASK, 'runner_revision': REVISION, 'controller_source': str(src),
        'controller_snapshot': str(snap), 'controller_sha256': sha, 'runner_sha256': sha256(__file__),
        'manifest_sha256': manifest, 'argv': sys.argv, 'indices': idx, 'split': SPLIT, 'workers': a.workers,
        'egl_devices': devs, 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'env': {k: os.environ.get(k) for k in ('MUJOCO_GL', 'RSI_EGL_DEVICES', 'RSI_SAM3_REMOTE', 'RSI_SAM3_ADDR')}}, indent=1))
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(str(snap), sha, i, out / f'{SPLIT}_{i:03d}', not a.no_video, .4 * (n % a.workers), devs[n % len(devs)]) for n, i in enumerate(idx)]
    t0 = time.time()
    rs = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as ex:
        futs = [ex.submit(episode, j) for j in jobs]
        for j, f in zip(jobs, futs):
            try:
                rs.append(f.result())
            except Exception as e:  # worker crash: one seed must not destroy the others; it stays in the denominator
                r = {'task': TASK, 'split': SPLIT, 'index': j[2], 'success': False, 'raw_success': None, 'infra': True,
                     'runner_revision': REVISION, 'error': f'worker_crash: {type(e).__name__}: {e}'}
                if not (j[3] / 'result.json').exists():
                    (j[3] / 'result.json').write_text(json.dumps(r, indent=1))
                rs.append(r)
    rows = [{'index': r['index'], 'success': bool(r.get('success')), 'raw_success': r.get('raw_success'),
             'partial': round(r.get('partial', 0), 3), 'error': r.get('error'), 'scoring_error': r.get('scoring_error'),
             'numerical_invalid': r.get('numerical_invalid'), 'failed': r.get('failed_checks'),
             'infra': bool(r.get('infra'))} for r in rs]
    s = {'task': TASK, 'runner_revision': REVISION, 'controller_sha256': sha, 'split': SPLIT,
         'successes_fail_closed': sum(r['success'] for r in rows), 'successes_raw': sum(bool(r['raw_success']) for r in rows),
         'infra_errors': [r['index'] for r in rows if r['infra']], 'n': len(rows), 'wall_sec': round(time.time() - t0, 1), 'rows': rows}
    (out / 'summary.json').write_text(json.dumps(s, indent=1))
    for r in rows:
        print(r['index'], r['success'], r['raw_success'], r['partial'], r['error'], r['failed'])
    print(f"RESULT {a.run}: {s['successes_fail_closed']}/{s['n']} fail-closed (raw {s['successes_raw']}), infra {s['infra_errors']}, {s['wall_sec']}s -> {out}", flush=True)


if __name__ == '__main__':
    main()
