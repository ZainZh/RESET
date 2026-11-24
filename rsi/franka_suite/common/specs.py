'''Franka suite task specs (privileged: controllers never import this).

Task semantics (source world, scorer kind, cap, goals) come unchanged from the LIFT2 suite registry
(rsi/lift2_suite/common/specs.py, read-only). DEV0-19 layouts are the SOURCE layouts already frozen for the LIFT2 suite
(identity map, generator rsi.layouts.sample / env_v3.sample on the source xTrainer scene); their SHA-256 is pinned here.
The only per-task Franka choice is the robot station: board_xy = board (source task table) top centre in the robot frame.
'''
from pathlib import Path
from rsi.lift2_suite.common import specs as L

ROOT = Path(__file__).resolve().parents[3]
LAYOUT_SRC = {t: ROOT / f'runs/lift2_suite_20261108/worlds/dev_layouts_{t}_0_19.json' for t in L.TASKS if t not in ('collect_apple', 'fold_tshirt')}
LAYOUT_SRC['collect_apple'] = ROOT / 'runs/lift2_suite_20261108/smooth_table_v1_rigid/dev_layouts_collect_apple_0_19.json'
LAYOUT_PINS = {
    'collect_apple': 'a25ac444ac4702dfb2560ec1ca0bb6b891715acbf9db8059f33a9cccc74404e9',
    'build_tower': '935fbb9cd1e9b3d618a9810301b2674432338f636f6981c17931b9068075b17b',
    'collect_cubes': 'ff2347a38ccebd89d8c2963af83c1036edcd0a3af275cb40f64ff2d2f3304d28',
    'lift_box': '515421111a81cfa254058e01aa90ef026b6d83411308d777505abf0e4aaa63ea',
    'open_drawer_and_place_object': '1e751aaa339f51773c11a821933dbec46d699a0a95c8935d2655759ca3b264f0',
    'put_mug_on_box': '33783538cdb4705906a5b02d8fa99c00cd09399e5069eedf6c866b38fa8e6bda',
    'stack_cups': 'ef32c1bd7a5c2404c3018e933f6c1382ec98bd01d51727dda4ec93e8a8b5e03c',
    'sweep_objects': 'ae17336462e3ade38f0477b958db7b393c97147f54479b0273a34f08765fa34d',
}
TASK_IDS = ('collect_apple', 'collect_cubes', 'stack_cups', 'put_mug_on_box', 'build_tower', 'open_drawer_and_place_object', 'lift_box', 'sweep_objects')
STATION = {t: (.70, 0.) for t in TASK_IDS}  # board centre 0.70 m ahead = where the main_cam optical axis meets the desk


def spec(t):
    if t not in TASK_IDS:
        raise ValueError(f'unknown task {t}')
    s = dict(L.runnable(t)); s['layouts'] = LAYOUT_SRC[t]; s['layout_pin'] = LAYOUT_PINS[t]; s['board_xy'] = STATION[t]
    return s


def source_task(t):
    return L.source_task(t)
