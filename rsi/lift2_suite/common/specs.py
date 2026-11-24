'''LIFT2 suite task specs (privileged: controllers never import this).

One entry per task. Everything not listed here (objects, masses, friction, joints, equalities, contacts, solver options,
scorer thresholds, caps, source controller) comes unchanged from the source task world / scorer / controller.
status: ready | done_do_not_repeat | blocked_prerequisite.
scorer: generic_rsi (rsi.scorer semantics, role-mapped) | box_strict_v2 | sweep_v3 (per-step privileged Monitor).
Mount entries are DECLARED fixed table mounts (same for every seed); they must pass the static checks in build_world.py.
'''
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RSI_WORLDS = ROOT / 'worlds/xtrainer'
SWEEP_WORLD = ROOT / 'worlds/xtrainer/sweep_objects'
DEFAULT_MOUNT = {'edge': '+X', 'lateral_v_m': 0., 'overhang_m': .06, 'platform_back_x_m': .01}  # apple v1 mount, task-frame relative to the source table


def _t(world, scorer, branch, cap, goals, sha=None, status='ready', articulated=False, note=''):
    return dict(world=world, scorer=scorer, source_branch=branch, source_sha_prefix=sha, cap_sec=cap, goals=goals, status=status,
        articulated=articulated, mount=dict(DEFAULT_MOUNT), note=note)


TASKS = {
    'collect_apple': _t(RSI_WORLDS / 'collect_apple', 'generic_rsi', 'a5', 30, ['inside:apple:bowl'], status='done_do_not_repeat', note='l7b 19/20 already delivered'),
    'collect_cubes': _t(RSI_WORLDS / 'collect_cubes', 'generic_rsi', 'c30_drop_memory', 90, ['inside:yellow_cube:bowl', 'inside:yellow_bar:bowl', 'inside:green_cube:bowl', 'inside:purple_cube:bowl', 'inside:blue_cube:bowl'], sha='954a6312'),
    'build_tower': _t(RSI_WORLDS / 'build_tower', 'generic_rsi', 't1', 60, ['on:purple_cube:purple_bar', 'on:yellow_cube:purple_cube'], sha='6d37f79d'),
    'stack_cups': _t(RSI_WORLDS / 'stack_cups', 'generic_rsi', 'c6', 30, ['nested:purple_cup:blue_cup'], sha='df85a1b9'),
    'put_mug_on_box': _t(RSI_WORLDS / 'put_mug_on_box', 'generic_rsi', 'm1', 45, ['on:mug:box'], sha='7cd36e90'),
    'open_drawer_and_place_object': _t(RSI_WORLDS / 'open_drawer_and_place_object', 'generic_rsi', 's1', 60, ['in_drawer:orange_cube:drawer'], articulated=True,
        note='object_drawer_shell static + slide-jointed object_drawer child; drawer_slide must stay passive'),
    'lift_box': _t(RSI_WORLDS / 'lift_box', 'box_strict_v2', 'b8', 40, ['on:carton:box', 'long_axis_vertical:carton', 'upright:box']),
    'sweep_objects': _t(SWEEP_WORLD, 'sweep_v3', 'w9', 40, ['sweep_v3'], note='scorer_v3 independently accepted; per-step Monitor stays privileged and active in every controller and dwell step'),
    'fold_tshirt': _t(None, None, None, None, [], status='blocked_prerequisite', note='no valid physical grasp on the source cloth world yet; not run, no failure runs spent'),
}


def spec(task_id):
    if task_id not in TASKS:
        raise ValueError(f'unknown task {task_id}; choose from {sorted(TASKS)}')
    return TASKS[task_id]


def runnable(task_id):
    s = spec(task_id)
    if s['status'] == 'blocked_prerequisite':
        raise RuntimeError(f'{task_id} is blocked: {s["note"]}')
    return s


def world_dir(task_id):
    return Path(runnable(task_id)['world'])


def record(task_id):
    return json.loads((world_dir(task_id) / 'reconstruction.json').read_text())


def source_task(task_id):
    '''Source Task object (instruction, goals, upright, objects, cap) for the original xTrainer task.'''
    runnable(task_id)
    if task_id == 'sweep_objects':
        from rsi.xtrainer_sweep.sweep import env_v3
        return env_v3.TASK
    from rsi.tasks import get
    return get(task_id)


def source_pins():
    '''Full SHA-256 pins of the source controllers.'''
    return json.loads((ROOT / 'rsi/lift2_suite/source_controllers.json').read_text())


def resolve_source_controller(task_id):
    '''Exact source controller file for the task; fails closed on ambiguity or a pinned-SHA mismatch.'''
    import hashlib
    s = runnable(task_id); base = ROOT / 'rsi/branches' / task_id
    exact = base / s['source_branch']
    cands = [exact] if exact.is_dir() else sorted(base.glob('*' + s['source_branch']))
    cands = [c for c in cands if (c / 'controller.py').is_file()]
    if len(cands) != 1:
        raise RuntimeError(f'{task_id}: source branch {s["source_branch"]} resolves to {[str(c) for c in cands]}')
    p = cands[0] / 'controller.py'; h = hashlib.sha256(p.read_bytes()).hexdigest()
    pin = source_pins().get(task_id)
    if not pin or pin['sha256'] != h or Path(pin['controller']).resolve() != p.resolve():
        raise RuntimeError(f'{task_id}: controller {p} sha256 {h} does not match the full pin {pin}')
    return p, h
