# Exact source pointers for the EIGHT rigid tasks of the G1 suite. Pure metadata plus file hashes; no MuJoCo import.
# Sweep is NOT in rsi.tasks.TASKS: it has its own world, env_v3 sampler and scorer_v3. Controllers are the frozen source xTrainer controllers (transfer baselines, never edited).
RESET_ROOT = __import__('os').environ.get('RESET_ROOT') or next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'eval.py').is_file()).as_posix()
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORLDS = ROOT / 'worlds/xtrainer'
SWEEP_WORLD = ROOT / 'worlds/xtrainer/sweep_objects'
BRANCHES = ROOT / 'rsi/branches'
SCORER_GENERIC = ROOT / 'rsi/scorer.py'
SCORER_BOX = ROOT / 'rsi/xtrainer_lift_box/lift_box_scorer.py'
SCORER_SWEEP = ROOT / 'rsi/xtrainer_sweep/sweep/scorer_v3.py'
SAMPLER_GENERIC = ROOT / 'rsi/layouts.py'
SAMPLER_SWEEP = ROOT / 'rsi/xtrainer_sweep/sweep/env_v3.py'
SOURCE_PYTHON = __import__('os').environ.get('RESET_PYTHON', 'python')
G1_PYTHON = __import__('sys').executable
G1_SCENE = Path(RESET_ROOT + '/robots/g1/g1_scene.xml')
G1_MESHDIR = Path(RESET_ROOT + '/robots/g1')
DEV_SEEDS = tuple(range(20))
TARGET_SUCCESSES = 17  # of 20 per task

# Nominal fixed G1 mount (tabletop centre in the G1 model-root frame). Same for every task and seed; not reach-optimized.
NOMINAL_MOUNT = (0.40, 0.0, 0.78)

_CONTROLLERS = {
    'collect_apple': 'collect_apple/a5',
    'collect_cubes': 'collect_cubes/xtrainer_20261107_c30_drop_memory',
    'stack_cups': 'stack_cups/xtrainer_20261107_c6',
    'put_mug_on_box': 'put_mug_on_box/m1',
    'build_tower': 'build_tower/xtrainer_20261107_t1',
    'open_drawer_and_place_object': 'open_drawer_and_place_object/xtrainer_20261107_s1',
    'sweep_objects': 'sweep_objects/xtrainer_20261107_w9',
    'lift_box': 'lift_box/xtrainer_20261107_b8',
}

TASK_NAMES = tuple(_CONTROLLERS)


def _spec(name):
    sweep = name == 'sweep_objects'
    return {
        'name': name,
        'world': SWEEP_WORLD if sweep else WORLDS / name,
        'controller': BRANCHES / _CONTROLLERS[name] / 'controller.py',
        'scorer': SCORER_SWEEP if sweep else (SCORER_BOX if name == 'lift_box' else SCORER_GENERIC),
        'sampler': SAMPLER_SWEEP if sweep else SAMPLER_GENERIC,
        'task_module': 'rsi.xtrainer_sweep.sweep.env_v3' if sweep else 'rsi.tasks',
        'sampler_call': 'env_v3.sample(split, index)' if sweep else 'rsi.layouts.sample(name, split, index, model, record)',
    }


SPECS = {n: _spec(n) for n in TASK_NAMES}


def get(name):
    if name not in SPECS:
        raise ValueError('unknown task %s; choose from %s' % (name, sorted(SPECS)))
    return SPECS[name]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def hashes(name):
    s = get(name)
    w = s['world']
    return {
        'scene_sha256': sha256(w / 'scene.xml'),
        'record_sha256': sha256(w / 'reconstruction.json'),
        'reference_sha256': sha256(w / 'reference.npz'),
        'controller_sha256': sha256(s['controller']),
        'scorer_sha256': sha256(s['scorer']),
        'sampler_sha256': sha256(s['sampler']),
    }
