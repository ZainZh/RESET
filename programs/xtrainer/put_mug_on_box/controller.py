# put_mug_on_box m1 (m0 with SAM3 text-grounded mug and box): left grasps mug -> hand-over at table centre -> right regrasps -> box top -> release
import numpy as np
from rsi import perception as P
from rsi import perception_sam3 as S
from rsi import perception_geo as G


def tcps_of(io):
    return [io.tcp(a)[:3, 3] for a in ('left', 'right')]


def find_mug(io, obs):
    m = S.locate_object(S.get_proc(), obs, "mug", plane=G.fit_plane(obs), max_height=.14)
    if m is None:
        raise RuntimeError("mug not found")
    rim = m["points"][m["points"][:, 2] > m["top"] - .02]
    xy, _ = P.fit_upright_cylinder(rim, .042)
    return xy


def find_box(io, obs):
    b = S.locate_object(S.get_proc(), obs, "wooden box", plane=G.fit_plane(obs), max_height=.14)
    if b is None:
        raise RuntimeError("box not found")
    top = b["points"][b["points"][:, 2] > b["top"] - .012]
    return top[:, :2].mean(0), b["top"]


def run(io):
    ev = io.demo['events']
    lc = next(e for e in ev if e['arm'] == 'left' and e['kind'] == 'close')
    lo = next(e for e in ev if e['arm'] == 'left' and e['kind'] == 'open' and e['time'] > lc['time'])
    rc = next(e for e in ev if e['arm'] == 'right' and e['kind'] == 'close')
    ro = next(e for e in ev if e['arm'] == 'right' and e['kind'] == 'open' and e['time'] > rc['time'])
    obs = io.observe()
    mug = find_mug(io, obs)
    box_xy, box_top = find_box(io, obs)
    io.log('perceived', mug=mug, box=box_xy, box_top=box_top)
    up = np.array([0, 0, 1.])
    # left arm: grasp the mug with the demo grasp offset
    off = lc['position'] - lc['objects']['mug']
    t = np.r_[mug + off[:2], lc['position'][2]]
    io.grip('left', 1., .3)
    io.move('left', t + .10 * up, lc['rotation'], 1.5)
    io.move('left', t, lc['rotation'], 1.0)
    io.grip('left', 0., .6)
    io.move('left', t + .12 * up, lc['rotation'], 1.0)
    # hand-over spot (demo prior), release, retreat to home
    h = lo['position']
    io.move('left', h + .06 * up, lo['rotation'], 2.0)
    io.move('left', h, lo['rotation'], .8)
    io.grip('left', 1., .6)
    io.move('left', h + .12 * up, lo['rotation'], .8)
    home = io.demo['home']['left']
    try:
        io.move('left', home[:3, 3], home[:3, :3], 2.0)
    except Exception as e:
        io.log('left_home_failed', err=str(e)[:80])
    # right arm: re-perceive the mug, regrasp, carry to the box
    obs = io.observe()
    mug2 = find_mug(io, obs)
    off2 = rc['position'] - rc['objects']['mug']
    t2 = np.r_[mug2 + off2[:2], rc['position'][2]]
    io.log('regrasp', mug=mug2, target=t2)
    io.grip('right', 1., .3)
    io.move('right', t2 + .10 * up, rc['rotation'], 1.5)
    io.move('right', t2, rc['rotation'], 1.0)
    io.grip('right', 0., .6)
    io.move('right', t2 + .15 * up, rc['rotation'], 1.0)
    boff = ro['position'] - ro['objects']['box']
    drop = np.r_[box_xy + boff[:2], box_top + ro['position'][2] - (ro['objects']['box'][2] + .05)]
    io.move('right', drop + .08 * up, ro['rotation'], 2.0)
    io.move('right', drop, ro['rotation'], 1.0)
    io.grip('right', 1., .6)
    io.move('right', drop + .12 * up, ro['rotation'], .8)
    io.wait(2.5)
    io.observe()
