# tdbs_parent_r2 (r2 = r1 calling tdbs_seed_r2.py): runs DEV seeds of one task/controller in isolated per-seed child processes with a hard wall timeout; keeps raw stdout/stderr and a
# record for every seed (crash/timeout -> failed record). Each child: flock $TMPDIR/egl_d2_g1dbs.lock, G1 venv, NVIDIA EGL device 2. Only a timed-out child's
# own process group is killed. Never overwrites. DEV seeds 0-19 only.
RESET_ROOT = __import__('os').environ.get('RESET_ROOT') or next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'eval.py').is_file()).as_posix()
import sys, os, json, time, argparse, subprocess, signal
ap = argparse.ArgumentParser()
ap.add_argument('--task', required=True)
ap.add_argument('--ctl', required=True)
ap.add_argument('--profiles', required=True)
ap.add_argument('--outdir', required=True)
ap.add_argument('--seeds', default='0-19')
ap.add_argument('--world', default='b2')
ap.add_argument('--timeout', type=int, default=900)
ap.add_argument('--video-seeds', default='')
ap.add_argument('--diag', action='store_true')
ap.add_argument('--layout-file', required=True)
ap.add_argument('--layout-key', required=True, choices=['dev', 'test'])
ap.add_argument('--lock', required=True)  # own flock file per parent
ap.add_argument('--egl-device', required=True, choices=[str(i) for i in range(16)])
a = ap.parse_args()
HERE = RESET_ROOT + '/programs/g1/drawer_box_sweep'  # original dir (cwd / imports unchanged)
RESET_HERE = os.path.dirname(os.path.abspath(__file__))
PY = __import__('sys').executable
LOCK = a.lock
ENV = dict(os.environ)
ENV.pop('PYTHONPATH', None)
ENV.update(__EGL_VENDOR_LIBRARY_FILENAMES=RESET_ROOT + '/configs/egl/10_nvidia.json', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
           MUJOCO_EGL_DEVICE_ID=a.egl_device, RSI_SAM3_ADDR=ENV.get('RSI_SAM3_ADDR', __import__('tempfile').gettempdir() + '/sam3_g1_r4.sock'))


def parse(s):
    out = []
    for part in s.split(','):
        part = part.strip()
        if part:
            if '-' in part:
                x, y = part.split('-'); out += list(range(int(x), int(y) + 1))
            else:
                out.append(int(part))
    return out


seeds, video = parse(a.seeds), set(parse(a.video_seeds))
if any(s < 0 or s > 49 for s in seeds):  # DEV 0-19 or TEST 0-49 via --layout-key
    raise SystemExit('only seeds 0-49 are allowed')
os.makedirs(a.outdir, exist_ok=True)
rows = []
for s in seeds:
    out = '%s/seed_%02d.json' % (a.outdir, s)
    if os.path.exists(out):
        raise SystemExit('refusing to overwrite ' + out)
    cmd = ['flock', LOCK, PY, '-I', RESET_HERE + '/tdbs_seed_eval.py', '--task', a.task, '--ctl', a.ctl, '--profiles', a.profiles, '--seed', str(s), '--outdir', a.outdir,
           '--world', a.world] + (['--video'] if s in video else []) + (['--diag'] if a.diag else []) + ['--layout-file', os.path.abspath(a.layout_file), '--layout-key', a.layout_key]
    t0 = time.time()
    so, se = open('%s/seed_%02d.stdout.log' % (a.outdir, s), 'w'), open('%s/seed_%02d.stderr.log' % (a.outdir, s), 'w')
    p = subprocess.Popen(cmd, env=ENV, stdout=so, stderr=se, start_new_session=True, cwd=HERE)
    try:
        status = 'exit_%d' % p.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        status = 'hard_timeout_%ds' % a.timeout
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
    so.close(); se.close()
    wall = round(time.time() - t0, 1)
    if os.path.exists(out):
        r = json.load(open(out))
        sr = r.get('scorer_result') or {}
        row = dict(seed=s, status=status, success=bool(r.get('success')), error=r.get('error'), vetoes=r.get('vetoes'), controller_error=r.get('controller_error'),
                   failed_checks=sr.get('failed_checks') if isinstance(sr, dict) else None, wall_sec=wall)
    else:
        row = dict(seed=s, status=status, success=False, error='no_child_record', wall_sec=wall)
        json.dump(dict(seed=s, task=a.task, ctl=a.ctl, success=False, error='child_crash_or_timeout:' + status, wall_sec=wall), open(out, 'w'), indent=1)
    rows.append(row)
    print('PARENT seed', s, json.dumps(row), flush=True)
n = sum(r['success'] for r in rows)
res = dict(eval_layout_file=os.path.abspath(a.layout_file), eval_layout_key=a.layout_key, eval_layout_sha256=__import__('hashlib').sha256(open(a.layout_file, 'rb').read()).hexdigest(), eval_egl_device=a.egl_device, eval_lock=LOCK, task=a.task, ctl=a.ctl, world=a.world, profiles=json.loads(a.profiles), seeds=seeds, n_success=n, n_total=len(rows), rows=rows, timeout_sec=a.timeout, diag=a.diag,
           disclosure='fixed pelvis, legs, waist; robot-body gravcomp; not walking or balance; DEV only; trusted-side grasp profiles; finger encoder clamp logged')
sp = '%s/summary_%d_%d.json' % (a.outdir, seeds[0], seeds[-1])
if os.path.exists(sp):
    raise SystemExit('refusing to overwrite ' + sp)
json.dump(res, open(sp, 'w'), indent=1)
print('PARENT_DONE %s %s success %d/%d' % (a.task, a.ctl, n, len(rows)), flush=True)
