# Split taken from env RSI_EVAL_SPLIT (default dev),
# outputs under results/xtrainer/<task>/. Physics, layouts, IO, scorer and fail-closed logic unchanged.
"""DEV runner for this subtask. Writes results/xtrainer/<task>/<run>/ and never overwrites.

Each run: the controller file is copied to <run>/controller_snapshot/controller.py, made read-only, hashed,
and the batch executes the snapshot (not the live branch file). meta.json records the snapshot sha256, the
start time and the freeze identity: sha256 of every file in --freeze manifest (sha256sum format), verified
against the manifest before starting (mismatch aborts unless --no-freeze-check).

env -u PYTHONPATH MUJOCO_GL=egl RSI_SAM3_REMOTE=1 python -m rsi.eval.xtrainer.pending \
    sweep_objects rsi/branches/sweep_objects/<id>/controller.py <run_name> --indices 0-7 \
    --env rsi.xtrainer_sweep.sweep.env --freeze rsi/xtrainer_sweep/sweep/FREEZE.sha256
"""
import argparse, hashlib, json, os, shutil, stat, time, importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPLIT = os.environ.get('RSI_EVAL_SPLIT', 'dev')
RUNS = ROOT / 'results/xtrainer'


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


def freeze_identity(manifest):
    rows, bad = [], []
    for line in Path(manifest).read_text().splitlines():
        if not line.strip(): continue
        want, path = line.split(None, 1); path = path.strip()
        have = sha(ROOT / path)
        rows.append({'path': path, 'sha256': have}); bad += [path] if have != want else []
    return {'manifest': str(manifest), 'manifest_sha256': sha(manifest), 'files': rows, 'mismatches': bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task'); ap.add_argument('controller'); ap.add_argument('run')
    ap.add_argument('--indices', required=True); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--env', default='rsi.env', help='episode runner module (rsi.env or a subtask runner)')
    ap.add_argument('--freeze', default=None); ap.add_argument('--no-freeze-check', action='store_true')
    ap.add_argument('--video', action='store_true', help='contact-sheet frames (off during tuning)')
    ap.add_argument('--note', default='')
    a = ap.parse_args()
    out = RUNS / a.task / a.run
    if out.exists():
        raise SystemExit(f'{out} exists; runs are never overwritten')
    ident = freeze_identity(a.freeze) if a.freeze else None
    if ident and ident['mismatches'] and not a.no_freeze_check:
        raise SystemExit(f'freeze mismatch: {ident["mismatches"]}')
    out.mkdir(parents=True)
    src = Path(a.controller).resolve()
    snap_dir = out / 'controller_snapshot'; snap_dir.mkdir()
    snap = snap_dir / 'controller.py'; shutil.copy2(src, snap)
    for extra in ('NOTES.md',):
        if (src.parent / extra).exists(): shutil.copy2(src.parent / extra, snap_dir / extra)
    os.chmod(snap, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    idx = parse_indices(a.indices)
    meta = {'task': a.task, 'controller_source': str(src), 'controller_snapshot': str(snap), 'controller_sha256': sha(snap),
        'indices': idx, 'split': SPLIT, 'env': a.env, 'workers': a.workers, 'video': a.video, 'note': a.note,
        'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'freeze': ident}
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    env = importlib.import_module(a.env)
    t0 = time.time()
    rs = env.batch(a.task, snap, SPLIT, idx, out, workers=a.workers, video=a.video)
    if sha(snap) != meta['controller_sha256']:
        raise SystemExit('controller snapshot changed during the run')
    rows = [{'index': r['index'], 'success': r['success'], 'raw_goal_success': r.get('raw_goal_success'),
        'partial': round(r.get('partial', 0), 3), 'drift_mm': round(r.get('drift_m', 0) * 1000, 1), 'error': r.get('error'),
        'failed': r.get('failed_checks'), 'fail_reasons': r.get('fail_reasons'), 'used_attempt': r.get('used_attempt'),
        'attempts': r.get('attempts'), 'rerun_after_infra_error': r.get('rerun_after_infra_error', False)} for r in rs]
    s = {'task': a.task, 'controller_sha256': meta['controller_sha256'], 'split': SPLIT, 'successes': sum(r['success'] for r in rs),
        'n': len(rs), 'failed_indices': [r['index'] for r in rs if not r['success']], 'wall_sec': round(time.time() - t0, 1),
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'rows': rows}
    (out / 'summary.json').write_text(json.dumps(s, indent=1))
    for r in rows:
        print(r['index'], r['success'], r['partial'], r['drift_mm'], r['error'], r['failed'])
    print(f"RESULT {a.task} {a.run}: {s['successes']}/{s['n']} ({s['wall_sec']}s) -> {out}", flush=True)


if __name__ == '__main__':
    main()
