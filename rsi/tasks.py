"""Task specs for the current-setup worlds (privileged: never imported by controllers).

`movable` gives layout randomization half-ranges per object: (dx_m, dy_m, dyaw_deg)
about the reconstructed demo pose. `goals` are scorer relations. `cap_sec` bounds
simulated time. `info` is what a controller may read (instruction + demo priors).
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLDS = ROOT / 'worlds/xtrainer'


@dataclass(frozen=True)
class Task:
    name: str
    instruction: str
    movable: dict
    goals: list
    cap_sec: float
    upright: tuple = ()
    objects: dict = field(default_factory=dict)  # perception spec per object (public)

    @property
    def world(self):
        return WORLDS / self.name

    @property
    def record(self):
        return json.loads((self.world / 'reconstruction.json').read_text())


CUBE = {'kind': 'block', 'size': [.032, .032, .032]}
TASKS = {t.name: t for t in [
    Task('collect_apple', 'Put the apple into the bowl.',
         movable={'apple': (.06, .06, 180), 'bowl': (.04, .04, 0)},
         goals=[('inside', 'apple', 'bowl')], cap_sec=30, upright=('bowl',),
         objects={'apple': {'kind': 'sphere', 'color': 'red', 'size': [.068, .068, .07]},
                  'bowl': {'kind': 'bowl', 'color': 'beige', 'size': [.2, .2, .09]}}),
    Task('collect_cubes', 'Put all five blocks into the bowl.',
         movable={'bowl': (.03, .03, 0), 'yellow_cube': (.04, .04, 45), 'yellow_bar': (.03, .03, 30),
                  'green_cube': (.03, .03, 45), 'purple_cube': (.04, .04, 45), 'blue_cube': (.04, .04, 45)},
         goals=[('inside', n, 'bowl') for n in ('yellow_cube', 'yellow_bar', 'green_cube', 'purple_cube', 'blue_cube')],
         cap_sec=90, upright=('bowl',),
         objects={'bowl': {'kind': 'bowl', 'color': 'beige', 'size': [.2, .2, .09]},
                  'yellow_cube': {**CUBE, 'color': 'yellow'}, 'yellow_bar': {**CUBE, 'color': 'yellow'},
                  'green_cube': {**CUBE, 'color': 'green'}, 'purple_cube': {**CUBE, 'color': 'purple'},
                  'blue_cube': {**CUBE, 'color': 'blue'}}),
    Task('put_mug_on_box', 'Put the mug on top of the wooden box.',
         movable={'mug': (.05, .05, 60), 'box': (.03, .03, 10)},
         goals=[('on', 'mug', 'box')], cap_sec=45, upright=('mug',),
         objects={'mug': {'kind': 'mug', 'color': 'grey_blue', 'size': [.12, .09, .1]},
                  'box': {'kind': 'box', 'color': 'wood', 'size': [.216, .17, .1]}}),
    Task('open_drawer_and_place_object', 'Open the drawer and put the orange block inside.',
         movable={'orange_cube': (.05, .05, 45)},
         goals=[('in_drawer', 'orange_cube', 'drawer')], cap_sec=60,
         objects={'drawer_shell': {'kind': 'box', 'color': 'wood', 'size': [.2, .17, .1]},
                  'orange_cube': {**CUBE, 'color': 'orange'}}),
    Task('stack_cups', 'Put the purple cup into the blue cup.',
         movable={'blue_cup': (.04, .04, 0), 'purple_cup': (.05, .05, 0)},
         goals=[('nested', 'purple_cup', 'blue_cup')], cap_sec=30, upright=('blue_cup', 'purple_cup'),
         objects={'blue_cup': {'kind': 'cup', 'color': 'blue'}, 'purple_cup': {'kind': 'cup', 'color': 'purple'}}),
    Task('lift_box', 'Put the juice carton on top of the wooden box.',
         movable={'carton': (.05, .05, 30), 'box': (.03, .03, 10)},
         goals=[('on', 'carton', 'box')], cap_sec=40,
         objects={'carton': {'kind': 'block', 'color': 'carton', 'size': [.065, .1, .065]},
                  'box': {'kind': 'box', 'color': 'wood', 'size': [.216, .17, .1]}}),
    Task('build_tower', 'Stack the blocks: purple cube on the purple bar, then yellow cube on top.',
         movable={'purple_bar': (.04, .04, 30), 'purple_cube': (.04, .04, 45), 'yellow_cube': (.04, .04, 45)},
         goals=[('on', 'purple_cube', 'purple_bar'), ('on', 'yellow_cube', 'purple_cube')], cap_sec=60,
         objects={'purple_bar': {'kind': 'block', 'color': 'purple', 'size': [.064, .032, .032]},
                  'purple_cube': {**CUBE, 'color': 'purple'}, 'yellow_cube': {**CUBE, 'color': 'yellow'}}),
]}


def get(name):
    if name not in TASKS:
        raise ValueError(f'unknown task {name}; choose from {sorted(TASKS)}')
    return TASKS[name]
