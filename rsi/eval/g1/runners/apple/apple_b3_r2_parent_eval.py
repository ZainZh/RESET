# apple_b3_r2_parent (copy of cubes_r5_parent): runs b3 collect_apple DEV seeds in isolated per-seed child processes with a hard wall timeout. Raw child stdout/stderr are kept per seed. Failures, crashes and timeouts are all kept as per-seed records.
# Child = apple_b3_r2_seed.py under the G1 venv with NVIDIA EGL, serialized by flock on the EGL slot lock of the chosen device (default device 1 / slot1: EGL device 0 readback is broken). Only a timed-out child's own process group is killed; no other process is touched.
# SAM3: children talk to the private rsi SAM3 service on $TMPDIR/sam3_g1_r4.sock.
# Disclosure: G1 pelvis, legs and waist are fixed (no freejoint); robot-body gravcomp is in the world. This is not walking or balance. DEV seeds only. No VAL, TEST or hardware.
RESET_ROOT = __import__('os').environ.get('RESET_ROOT') or next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'eval.py').is_file()).as_posix()
import sys, os, json, time, argparse, subprocess, signal

ap = argparse.ArgumentParser()
ap.add_argument('--outdir', required=True)
ap.add_argument('--seeds', default='0-19')
ap.add_argument('--timeout', type=int, default=1800)
ap.add_argument('--video-seeds', default='')
ap.add_argument('--tag', default='apple_b3_r2')
ap.add_argument('--egl-device', required=True, choices=[str(i) for i in range(16)])
ap.add_argument('--sam-addr', default=__import__('tempfile').gettempdir() + '/sam3_g1_cubes_r2.sock')
ap.add_argument('--layout-file', required=True)
ap.add_argument('--layout-key', required=True, choices=['dev', 'test'])
ap.add_argument('--lock', required=True)  # own flock file per parent
args = ap.parse_args()
HERE = RESET_ROOT + '/programs/g1/collect_apple'  # original dir (cwd / imports unchanged)
RESET_HERE = os.path.dirname(os.path.abspath(__file__))
PY = __import__('sys').executable
LOCK = args.lock
ENV = dict(os.environ)
ENV.pop('PYTHONPATH', None)
ENV.update(__EGL_VENDOR_LIBRARY_FILENAMES=RESET_ROOT + '/configs/egl/10_nvidia.json', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=args.egl_device, RSI_SAM3_ADDR=args.sam_addr)


def parse(s):
    out = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


seeds = parse(args.seeds)
video = set(parse(args.video_seeds))
if any(s < 0 or s > 49 for s in seeds):  # DEV 0-19 or TEST 0-49 via --layout-key
    raise SystemExit('only seeds 0-49 are allowed')
os.makedirs(args.outdir, exist_ok=True)
summary = []
for s in seeds:
    out = '%s/seed_%02d.json' % (args.outdir, s)
    if os.path.exists(out):
        raise SystemExit('refusing to overwrite ' + out)
    cmd = ['flock', LOCK, PY, '-I', RESET_HERE + '/apple_b3_r2_seed_eval.py', '--seed', str(s), '--outdir', args.outdir] + (['--video'] if s in video else []) + ['--layout-file', os.path.abspath(args.layout_file), '--layout-key', args.layout_key]
    t0 = time.time()
    so, se = open('%s/seed_%02d.stdout.log' % (args.outdir, s), 'w'), open('%s/seed_%02d.stderr.log' % (args.outdir, s), 'w')
    p = subprocess.Popen(cmd, env=ENV, stdout=so, stderr=se, start_new_session=True)
    status = None
    try:
        rc = p.wait(timeout=args.timeout)
        status = 'exit_%d' % rc
    except subprocess.TimeoutExpired:
        status = 'hard_timeout_%ds' % args.timeout
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
    so.close()
    se.close()
    wall = round(time.time() - t0, 1)
    if os.path.exists(out):
        rec = json.load(open(out))
        row = dict(seed=s, status=status, success=bool(rec.get('success')), error=rec.get('error'), vetoes=rec.get('vetoes'), controller_error=rec.get('controller_error'), wall_sec=wall)
    else:
        row = dict(seed=s, status=status, success=False, error='no_child_record', vetoes=None, controller_error=None, wall_sec=wall)
        json.dump(dict(seed=s, task='collect_apple', tag=args.tag, success=False, error='child_crash_or_timeout:' + status, wall_sec=wall), open(out, 'w'), indent=1)
    summary.append(row)
    print('PARENT seed', s, json.dumps(row), flush=True)
n_ok = sum(1 for r in summary if r['success'])
res = dict(eval_layout_file=os.path.abspath(args.layout_file), eval_layout_key=args.layout_key, eval_layout_sha256=__import__('hashlib').sha256(open(args.layout_file, 'rb').read()).hexdigest(), eval_egl_device=args.egl_device, eval_lock=LOCK, egl_device=args.egl_device, tag=args.tag, task='collect_apple', build='b3', seeds=seeds, n_success=n_ok, n_total=len(summary), rows=summary, timeout_sec=args.timeout, sam_addr=args.sam_addr,
           disclosure='fixed pelvis, legs, waist; robot-body gravcomp; not walking or balance; DEV only; park pose and orientation candidates are code constants; finger encoder reads clamped on the trusted side and logged')
sp = args.outdir + '/summary_%s.json' % ('_'.join(str(x) for x in (seeds[0], seeds[-1])))
if os.path.exists(sp):
    raise SystemExit('refusing to overwrite ' + sp)
json.dump(res, open(sp, 'w'), indent=1)
print('PARENT_DONE success %d/%d' % (n_ok, len(summary)), flush=True)
