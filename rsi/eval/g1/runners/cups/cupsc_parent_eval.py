# cupsc_parent_r3 = cupsc_parent_r2 with default child script cupsc_seed_r3.py (plane-following layout shift).
# cupsc_parent_r2 = cupsc_parent_r1 + --shift-x passed to the child (default script cupsc_seed_r2.py) and default SAM3 address $TMPDIR/sam3.sock.
# cupsc_parent_r1 = copy of tasks_mug_clean/mugc_parent_r1.py; changes: script resolved in tasks_cups_clean/, lock $TMPDIR/egl_d2_g1cupsclean.lock, NVIDIA EGL device 2, default script cupsc_seed_r1.py, default timeout 240 s.
# cups_r1_parent: runs G1 DEV seeds of one task in isolated per-seed child processes with a hard wall timeout. Raw child stdout/stderr are kept per seed. Failures, crashes and timeouts are all kept as per-seed records.
# Child = --script (default cupsc_seed_r1.py) under the G1 venv with NVIDIA EGL device 2, serialized by flock on $TMPDIR/egl_d2_g1cupsclean.lock. Only a timed-out child's own process group is killed; no other process is touched.
# SAM3: children talk to the private rsi SAM3 service socket (default $TMPDIR/sam3_g1_r4.sock); the service is not started, changed or stopped here.
# Disclosure: G1 pelvis, legs and waist are fixed (no freejoint); robot-body gravcomp is in the world. This is not walking or balance. DEV seeds only. No VAL, TEST or hardware.
RESET_ROOT = __import__('os').environ.get('RESET_ROOT') or next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'eval.py').is_file()).as_posix()
import os, json, time, argparse, subprocess, signal

ap = argparse.ArgumentParser()
ap.add_argument('--outdir', required=True)
ap.add_argument('--script', default='cupsc_seed_eval.py')
ap.add_argument('--ctl', required=True)
ap.add_argument('--world', default='b2')
ap.add_argument('--seeds', default='0-19')
ap.add_argument('--timeout', type=int, default=240)
ap.add_argument('--video-seeds', default='')
ap.add_argument('--tag', required=True)
ap.add_argument('--sam-addr', default=__import__('tempfile').gettempdir() + '/sam3.sock')
ap.add_argument('--shift-x', default='0')
ap.add_argument('--layout-file', required=True)
ap.add_argument('--layout-key', required=True, choices=['dev', 'test'])
ap.add_argument('--lock', required=True)  # own flock file per parent
ap.add_argument('--egl-device', required=True, choices=[str(i) for i in range(16)])
args = ap.parse_args()
HERE = RESET_ROOT + '/programs/g1/stack_cups'  # original dir (cwd / imports unchanged)
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
    cmd = ['flock', LOCK, PY, '-I', RESET_HERE + '/' + args.script, '--seed', str(s), '--outdir', args.outdir, '--world', args.world, '--ctl', args.ctl, '--tag', args.tag, '--shift-x', args.shift_x] + (['--video'] if s in video else []) + ['--layout-file', os.path.abspath(args.layout_file), '--layout-key', args.layout_key]
    t0 = time.time()
    so, se = open('%s/seed_%02d.stdout.log' % (args.outdir, s), 'w'), open('%s/seed_%02d.stderr.log' % (args.outdir, s), 'w')
    p = subprocess.Popen(cmd, env=ENV, stdout=so, stderr=se, start_new_session=True)
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
        sr = rec.get('scorer_result') or {}
        row = dict(seed=s, status=status, success=bool(rec.get('success')), error=rec.get('error'), vetoes=rec.get('vetoes'), controller_error=rec.get('controller_error'), failed_checks=sr.get('failed_checks'), wall_sec=wall)
    else:
        row = dict(seed=s, status=status, success=False, error='no_child_record', vetoes=None, controller_error=None, failed_checks=None, wall_sec=wall)
        json.dump(dict(seed=s, tag=args.tag, success=False, error='child_crash_or_timeout:' + status, wall_sec=wall), open(out, 'w'), indent=1)
    summary.append(row)
    print('PARENT seed', s, json.dumps(row, default=str)[:400], flush=True)
n_ok = sum(1 for r in summary if r['success'])
res = dict(eval_layout_file=os.path.abspath(args.layout_file), eval_layout_key=args.layout_key, eval_layout_sha256=__import__('hashlib').sha256(open(args.layout_file, 'rb').read()).hexdigest(), eval_egl_device=args.egl_device, eval_lock=LOCK, layout_shift_x=float(args.shift_x), tag=args.tag, script=args.script, ctl=args.ctl, world=args.world, seeds=seeds, n_success=n_ok, n_total=len(summary), rows=summary, timeout_sec=args.timeout, sam_addr=args.sam_addr,
           disclosure='fixed pelvis, legs, waist; robot-body gravcomp; not walking or balance; DEV only; park pose, orientation candidates and grasp profile are code constants / disclosed contact priors; finger encoder reads clamped on the trusted side and logged')
sp = args.outdir + '/summary_%d_%d.json' % (seeds[0], seeds[-1])
if os.path.exists(sp):
    raise SystemExit('refusing to overwrite ' + sp)
json.dump(res, open(sp, 'w'), indent=1, default=str)
print('PARENT_DONE success %d/%d' % (n_ok, len(summary)), flush=True)
