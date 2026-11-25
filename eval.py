#!/usr/bin/env python3
"""Evaluate the frozen RESET programs in the rebuilt simulation worlds.

    python eval.py --robot franka --task stack_cups --layouts test
    python eval.py --robot g1 --task lift_box --layouts test --seeds 0-9
    python eval.py --all --layouts test          # all 32 robot-task cells, one after another
    python eval.py --table                       # print the success table from results/

Every cell is run by the robot's own trusted episode runner (rsi/eval/<robot>/); the runner snapshots the
program, builds the world, plays the TEST layout, and scores with the frozen task scorer. Results go to
results/<robot>/<task>/<run>/ (one directory per episode plus summary.json); existing runs are never overwritten.

Environment:
  RESET_PYTHON      interpreter for xtrainer / lift2 / franka (default: this interpreter)
  RESET_G1_PYTHON   interpreter for g1 (default: .venv-g1/bin/python)
  RSI_SAM3_ADDR     unix socket of the running SAM3 service (see docs/SAM3_SERVICE.md)
  RESET_EGL_DEVICES comma-separated EGL device indices for rendering (default: 0)
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROBOTS = ('xtrainer', 'franka', 'lift2', 'g1')
TASKS = ('collect_apple', 'collect_cubes', 'stack_cups', 'build_tower', 'put_mug_on_box', 'open_drawer_and_place_object', 'sweep_objects', 'lift_box')
PAPER = {  # TEST 0-49 successes reported in the paper
    'xtrainer': dict(collect_apple=50, collect_cubes=49, stack_cups=48, build_tower=50, put_mug_on_box=50, open_drawer_and_place_object=50, sweep_objects=44, lift_box=47),
    'franka': dict(collect_apple=47, collect_cubes=49, stack_cups=50, build_tower=50, put_mug_on_box=47, open_drawer_and_place_object=50, sweep_objects=39, lift_box=50),
    'lift2': dict(collect_apple=43, collect_cubes=38, stack_cups=45, build_tower=46, put_mug_on_box=42, open_drawer_and_place_object=35, sweep_objects=29, lift_box=40),
    'g1': dict(collect_apple=43, collect_cubes=35, stack_cups=50, build_tower=35, put_mug_on_box=40, open_drawer_and_place_object=50, sweep_objects=16, lift_box=40),
}
# xTrainer: runner module per task (TEST layouts are sampled by rsi.layouts with the sealed TEST seed stream)
XT_MOD = dict(collect_apple='rigid', build_tower='rigid', put_mug_on_box='rigid', collect_cubes='cubes', stack_cups='cups',
              open_drawer_and_place_object='drawer', lift_box='box', sweep_objects='pending')
# LIFT2: fixed task setup (world variant) and per-seed wall timeout used for the paper runs
L2_SETUP = dict(collect_apple='worlds/lift2/collect_apple', collect_cubes='worlds/lift2/collect_cubes', stack_cups='worlds/lift2/stack_cups',
                build_tower='worlds/lift2/build_tower', put_mug_on_box='worlds/lift2/put_mug_on_box', open_drawer_and_place_object='worlds/lift2/open_drawer_and_place_object',
                sweep_objects='worlds/lift2/sweep_objects', lift_box='worlds/lift2/lift_box')
L2_LAYOUT = dict(collect_apple='collect_apple__collect_apple_o3', collect_cubes='collect_cubes__collect_cubes_o2', stack_cups='stack_cups__stack_cups_o1',
                 build_tower='build_tower__build_tower', put_mug_on_box='put_mug_on_box__put_mug_on_box_o13',
                 open_drawer_and_place_object='open_drawer_and_place_object__open_drawer_and_place_object', sweep_objects='sweep_objects__sweep_g4', lift_box='lift_box__lift_box_o4')
L2_TIMEOUT = dict(collect_apple=900, put_mug_on_box=2400)
# G1: launcher cell names
G1_CELL = dict(build_tower='tower', collect_apple='apple', collect_cubes='cubes', stack_cups='cups', put_mug_on_box='mug',
               open_drawer_and_place_object='drawer', lift_box='box', sweep_objects='sweep')


def parse_seeds(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-'); out += list(range(int(a), int(b or a) + 1))
    return out


def base_env():
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env.update(MUJOCO_GL='egl', RSI_SAM3_REMOTE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RESET_ROOT=str(ROOT), RSI_UNSEAL_TEST='1')
    env.setdefault('RSI_SAM3_ADDR', os.path.join(__import__('tempfile').gettempdir(), 'sam3.sock'))
    return env


def run_cell(robot, task, split, seeds, workers, name):
    env = base_env(); devs = os.environ.get('RESET_EGL_DEVICES', '0')
    py = os.environ.get('RESET_PYTHON', sys.executable)
    ctrl = f'programs/{robot}/{task}/controller.py'
    idx = ','.join(map(str, seeds))
    if robot == 'xtrainer':
        env.update(RSI_EVAL_SPLIT=split, RSI_EGL_DEVICES=devs)
        if XT_MOD[task] == 'rigid': env['RIGID_TASK'] = task
        cmd = [py, '-m', f'rsi.eval.xtrainer.{XT_MOD[task]}']
        cmd += [task] if task == 'sweep_objects' else []
        cmd += [ctrl, name, '--indices', idx, '--workers', str(workers)]
        if task == 'sweep_objects': cmd += ['--env', 'rsi.xtrainer_sweep.sweep.env_v3', '--no-freeze-check', '--note', 'reset_eval']
        out = ROOT / 'results/xtrainer' / task / name
    elif robot in ('lift2', 'franka'):
        if split != 'test': raise SystemExit('lift2/franka: only --layouts test is shipped')
        lay = f'layouts/test/lift2/{L2_LAYOUT[task]}__test0_49.json' if robot == 'lift2' else f'layouts/test/franka/{task}__{task}__test0_49.json'
        out = ROOT / 'results' / robot / task / name
        cmd = [py, f'rsi/eval/{robot}/runner.py', 'run', task, ctrl, str(out.relative_to(ROOT)), idx]
        if robot == 'lift2': cmd += ['--setup', L2_SETUP[task]]
        cmd += ['--devices', devs, '--workers', str(workers), '--timeout', str(L2_TIMEOUT.get(task, 1800) if robot == 'lift2' else 2400),
                '--layouts', lay, '--split', split]
        env.setdefault('RESET_EGL_SLOTS', '16')  # concurrent episodes per EGL device (shared by all cells)
        if robot == 'franka': env['FRANKA_WORLD_SET'] = 'worlds_v2_parallel'
    else:
        if split != 'test': raise SystemExit('g1: only --layouts test is shipped')
        gpy = os.environ.get('RESET_G1_PYTHON', str(ROOT / '.venv-g1/bin/python'))
        env['RESET_SAM_ADDR'] = env['RSI_SAM3_ADDR']; env['RESET_G1_PYTHON'] = gpy; env['RESET_EGL_DEVICES'] = devs
        n = max(1, min(workers, len(seeds))); chunks = [seeds[i::n] for i in range(n)]
        spec = ';'.join(','.join(map(str, c)) + '@3600' for c in chunks if c)  # 3600 s wall-clock hang guard per seed (as in the paper runs)
        cmd = [gpy, '-I', 'rsi/eval/g1/launch.py', G1_CELL[task], split, name, spec]
        out = ROOT / 'results/g1' / task / name
    print('$', ' '.join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    res = collect(out)
    res.update(robot=robot, task=task, split=split, seeds=seeds, run_dir=str(out.relative_to(ROOT)), returncode=rc, wall_s=round(time.time() - t0, 1))
    (ROOT / 'results').mkdir(exist_ok=True)
    with open(ROOT / 'results/RUNS.jsonl', 'a') as f: f.write(json.dumps(res) + '\n')
    p = PAPER[robot][task]
    print(f'== {robot}/{task} {split}: {res["n_success"]}/{res["n"]} successes  (paper TEST 0-49: {p}/50)', flush=True)
    return res


def collect(out):
    '''Per-seed success from the runner's own records (result.json per episode; G1: seed_XX.json).'''
    succ, n = [], 0
    for f in sorted(out.glob('*_[0-9][0-9][0-9]/result.json')) + sorted(out.glob('seed_[0-9][0-9].json')):
        r = json.loads(f.read_text()); n += 1
        i = r.get('index', r.get('seed'))
        if r.get('success'): succ.append(int(i))
    if not n and (out / 'summary.json').is_file():  # xtrainer sweep runner keeps per-seed verdicts in summary.json rows
        rows = json.loads((out / 'summary.json').read_text()).get('rows', [])
        n = len(rows); succ = [int(r['index']) for r in rows if r.get('success')]
    return {'n': n, 'n_success': len(succ), 'success_seeds': sorted(succ)}


def table():
    rows = [json.loads(l) for l in open(ROOT / 'results/RUNS.jsonl')] if (ROOT / 'results/RUNS.jsonl').exists() else []
    last = {}
    for r in rows:
        if r['split'] == 'test' and (ROOT / r['run_dir']).is_dir():
            r.update(collect(ROOT / r['run_dir'])); last[(r['robot'], r['task'])] = r
    print('| robot | ' + ' | '.join(TASKS) + ' |'); print('|---' * (len(TASKS) + 1) + '|')
    for rb in ROBOTS:
        cells = []
        for t in TASKS:
            r = last.get((rb, t))
            cells.append(f"{100 * r['n_success'] / max(r['n'], 1):.0f}% ({r['n_success']}/{r['n']}; paper {2 * PAPER[rb][t]}%)" if r else f'- (paper {2 * PAPER[rb][t]}%)')
        print(f'| {rb} | ' + ' | '.join(cells) + ' |')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--robot', choices=ROBOTS); ap.add_argument('--task', choices=TASKS)
    ap.add_argument('--all', action='store_true', help='run every robot-task cell')
    ap.add_argument('--layouts', default='test', choices=('test', 'dev'), help='layout split (paper numbers: test)')
    ap.add_argument('--seeds', default='0-49', help='layout indices, e.g. 0-49 or 0,5,7')
    ap.add_argument('--workers', type=int, default=4, help='parallel episodes per cell')
    ap.add_argument('--name', default=None, help='run directory name (default: <split><first>_<last>_<time>)')
    ap.add_argument('--table', action='store_true', help='print the results table and exit')
    a = ap.parse_args()
    if a.table: return table()
    cells = [(r, t) for r in ROBOTS for t in TASKS] if a.all else [(a.robot, a.task)]
    if None in cells[0]: ap.error('give --robot and --task, or --all')
    seeds = parse_seeds(a.seeds)
    for r, t in cells:
        name = a.name or f'{a.layouts}{seeds[0]}_{seeds[-1]}_{time.strftime("%Y%m%d_%H%M%S")}'
        run_cell(r, t, a.layouts, seeds, a.workers, name)
    table()


if __name__ == '__main__':
    main()
