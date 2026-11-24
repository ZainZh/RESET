# G1 launcher: runs one cell (delivered version) on DEV or TEST seeds with the evaluation runner copies, several parent processes in parallel
# (one seed range each, own flock file, EGL device alternating 0/1), then writes OUT/summary.json from the per-seed records. Never overwrites.
# SAM3 socket: env RESET_SAM_ADDR (default $TMPDIR/sam3.sock). Usage (cwd = repo root): python -I rsi/eval/g1/launch.py CELL dev|test OUTNAME "0-4;5-9[@TIMEOUT];..."
RESET_ROOT = __import__('os').environ.get('RESET_ROOT') or next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'eval.py').is_file()).as_posix()
import sys, os, json, subprocess, time, hashlib

REPO = RESET_ROOT
PY = __import__('sys').executable
G = REPO + '/programs/g1'
E = REPO + '/results/g1'
RUN = REPO + '/rsi/eval/g1/runners'
LAY = REPO + '/layouts/test/g1'
TDBS_PARK = '0,0.6,0,0.3,0,-0.4,0'
PROBE_DEV, PROBE_TEST = G + '/probe01_dev_layouts/layouts.json', LAY + '/probe01_test_layouts.json'
CELLS = {
    'tower': dict(task='build_tower', version='tc27', dev=PROBE_DEV, test=PROBE_TEST, timeout=300, orig='17/20', orig_fail=[1, 8, 18],
                  cmd=[RUN + '/tower/towc_parent_eval.py', '--task', 'build_tower', '--ver', 'tc27', '--world', 'b3', '--tag', 'tower_tc27_cleanpark', '--sam-addr', __import__('tempfile').gettempdir() + '/sam3.sock']),
    'apple': dict(task='collect_apple', version='apple_b3_r2', dev=PROBE_DEV, test=PROBE_TEST, timeout=900, orig='20/20', orig_fail=[],
                  cmd=[RUN + '/apple/apple_b3_r2_parent_eval.py', '--sam-addr', __import__('tempfile').gettempdir() + '/sam3.sock']),
    'cubes': dict(task='collect_cubes', version='cubes_v2_r1d_x2', dev=G + '/worlds/b3_cubes_v2_x2/layouts.json', test=LAY + '/b3_cubes_v2_x2_test_layouts.json', timeout=900,
                  orig='14/20', orig_fail=[2, 3, 5, 9, 10, 19], cmd=[REPO + '/rsi/eval/g1/runners/cubes/cubes_v2_r1d_x2_parent_eval.py', '--sam-addr', __import__('tempfile').gettempdir() + '/sam3.sock']),
    'cups': dict(task='stack_cups', version='cups_r28_planeshift045', dev=PROBE_DEV, test=PROBE_TEST, timeout=300, orig='19/20', orig_fail=[11],
                 cmd=[RUN + '/cups/cupsc_parent_eval.py', '--ctl', 'cups_r28_ctl', '--world', 'b3', '--tag', 'cups_r28_planeshift045', '--shift-x', '-0.045', '--sam-addr', __import__('tempfile').gettempdir() + '/sam3.sock']),
    'mug': dict(task='put_mug_on_box', version='mug_r4_cleanpark', dev=PROBE_DEV, test=PROBE_TEST, timeout=240, orig='19/20', orig_fail=[17],
                cmd=[RUN + '/mug/mugc_parent_eval.py', '--ctl', 'mug_r4_ctl', '--world', 'b3', '--tag', 'mug_r4_cleanpark', '--sam-addr', __import__('tempfile').gettempdir() + '/sam3.sock']),
    'drawer': dict(task='open_drawer_and_place_object', version='drawer_r9_cleanpark', dev=PROBE_DEV, test=PROBE_TEST, timeout=100, orig='19/20', orig_fail=[7],
                   cmd=[RUN + '/tdbs/tdbs_parent_eval.py', '--task', 'open_drawer_and_place_object', '--ctl', 'drawer_r9_ctl', '--profiles',
                        '{"left": ["cube", 0.032, 0.4], "right": ["cube", 0.032, 0.15]}', '--world', 'b3']),
    'box': dict(task='lift_box', version='box_r19', dev=PROBE_DEV, test=PROBE_TEST, timeout=100, orig='18/20', orig_fail=[5, 16],
                cmd=[RUN + '/tdbs/tdbs_parent_eval.py', '--task', 'lift_box', '--ctl', 'box_r19_ctl', '--profiles',
                     '{"left": ["sphere", 0.068, 0.0], "right": ["sphere", 0.068, 0.0]}', '--world', 'b3']),
    'sweep': dict(task='sweep_objects', version='sweep_r33', dev=G + '/worlds/b3_sweep_v1/layouts.json', test=LAY + '/b3_sweep_v1_test_layouts.json', timeout=75,
                  orig='9/20 (best, not frozen)', orig_fail=[0, 1, 2, 5, 8, 9, 10, 11, 13, 16, 19],
                  cmd=[RUN + '/tdbs/tdbs_parent_eval.py', '--task', 'sweep_objects', '--ctl', 'sweep_r33_ctl', '--profiles',
                       '{"left": ["cube", 0.032, 0.4], "right": ["cube", 0.032, 0.0]}', '--world', 'b3_sweep_v1']),
}


def parse(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


if __name__ == '__main__':
    cell, split, name, chunks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    C = CELLS[cell]
    lf = C[split]
    out = '%s/%s/%s' % (E, C['task'], name)
    if os.path.exists(out):
        raise SystemExit('refusing to reuse ' + out)
    os.makedirs(out)
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    sam = os.environ.get('RESET_SAM_ADDR', __import__('tempfile').gettempdir() + '/sam3.sock')  # same rsi.sam3_service protocol/checkpoint; $TMPDIR/sam3.sock (GPU0) or $TMPDIR/sam3_gpu1.sock (GPU1)
    C['cmd'] = [sam if x == __import__('tempfile').gettempdir() + '/sam3.sock' else x for x in C['cmd']]
    env.update(MUJOCO_GL='egl', RSI_SAM3_REMOTE='1', RSI_SAM3_ADDR=sam, TDBS_PARK=TDBS_PARK, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    _EGL = os.environ.get('RESET_EGL_DEVICES', '0,1').split(',')  # EGL devices to alternate over
    procs = []
    for k, ch in enumerate(chunks.split(';')):
        seeds, _, to = ch.partition('@')
        to = int(to) if to else C['timeout']
        cmd = [PY, '-I'] + C['cmd'] + ['--outdir', out, '--seeds', seeds, '--timeout', str(to), '--layout-file', lf, '--layout-key', split,
                                       '--lock', __import__('tempfile').gettempdir() + '/reset_g1_%s_%s_%d.lock' % (cell, name, k), '--egl-device', _EGL[k % len(_EGL)]]
        lg = open('%s/parent_%s.log' % (out, seeds), 'w')
        lg.write('CMD ' + json.dumps(cmd) + '\n'); lg.flush()
        procs.append((seeds, to, subprocess.Popen(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)))
    t0 = time.time()
    rc = {s: p.wait() for s, _, p in procs}
    rows = []
    allseeds = sorted(x for s, _, _ in procs for x in parse(s))
    tos = {x: to for s, to, _ in procs for x in parse(s)}
    for s in allseeds:
        f = '%s/seed_%02d.json' % (out, s)
        r = json.load(open(f)) if os.path.exists(f) else {'success': False, 'error': 'missing_record'}
        sr = r.get('scorer_result') if isinstance(r.get('scorer_result'), dict) else {}
        rows.append(dict(seed=s, success=bool(r.get('success')), error=r.get('error'), controller_error=r.get('controller_error'), vetoes=r.get('vetoes'),
                         failed_checks=sr.get('failed_checks'), sim_time=r.get('sim_time'), wall_sec=r.get('wall_sec'), timeout_sec=tos[s]))
    n = sum(r['success'] for r in rows)
    infra = [r['seed'] for r in rows if r['error'] and not str(r['error']).startswith('child_crash_or_timeout:hard_timeout')]
    timeouts = [r['seed'] for r in rows if r['error'] and 'hard_timeout' in str(r['error'])]
    res = dict(cell=cell, task=C['task'], version=C['version'], split=split, layout_file=lf, layout_sha256=hashlib.sha256(open(lf, 'rb').read()).hexdigest(),
               seeds=allseeds, n_success=n, n_total=len(rows), success_seeds=[r['seed'] for r in rows if r['success']], fail_seeds=[r['seed'] for r in rows if not r['success']],
               hard_timeout_seeds=timeouts, error_seeds=infra, parent_exit=rc, chunks=chunks, wall_total_sec=round(time.time() - t0, 1), rows=rows,
               original_dev=C['orig'], original_dev_fail_seeds=C['orig_fail'], interpreter=PY,
               env={k: env.get(k) for k in ('MUJOCO_GL', 'RSI_SAM3_REMOTE', 'RSI_SAM3_ADDR', 'TDBS_PARK', 'OMP_NUM_THREADS')},
               disclosure='G1 pelvis/legs/waist fixed (no freejoint) + robot-body gravity compensation; not walking or balance. Simulation only.')
    json.dump(res, open(out + '/summary.json', 'w'), indent=1)
    print('CELL_DONE', cell, split, name, '%d/%d' % (n, len(rows)), 'timeouts', timeouts, 'errors', infra, flush=True)
