"""collect_apple a5 (a4 + apple-centred release; a3 with SAM3 text-grounded apple and bowl, colour-free): grasp apple (demo grasp offset) -> above bowl -> release."""
import numpy as np
from rsi import perception as P
from rsi import perception_sam3 as S
from rsi import perception_geo as G


def run(io):
    obs = io.observe()
    tcps = [io.tcp(a)[:3, 3] for a in ('left', 'right')]
    proc = S.get_proc(); plane = G.fit_plane(obs)
    apple = S.locate_object(proc, obs, 'apple', plane=plane, max_height=.12); bowl = S.locate_object(proc, obs, 'bowl', plane=plane, max_height=.12)
    if apple is None or bowl is None:
        raise RuntimeError(f'perception failed apple={apple is not None} bowl={bowl is not None}')
    apple_c = P.fit_sphere(apple['points'], .034)
    bowl_xy, bowl_r = P.fit_upright_cylinder(bowl['points'][bowl['points'][:, 2] > bowl['top'] - .015])
    io.log('perceived', apple=apple_c, bowl=bowl_xy, bowl_r=bowl_r)
    close = next(e for e in io.demo['events'] if e['kind'] == 'close')
    opened = next(e for e in io.demo['events'] if e['kind'] == 'open' and e['time'] > close['time'])
    arm = close['arm']; R = close['rotation']
    grasp_off = close['position'] - close['objects']['apple']
    target = apple_c + grasp_off
    above = target + [0, 0, .10]
    io.grip(arm, 1., .3)
    io.move(arm, above, R, 1.5)
    io.move(arm, target, R, 1.0)
    io.grip(arm, 0., .6)
    io.move(arm, target + [0, 0, .15], R, 1.0)
    release_off = opened['position'] - opened['objects']['bowl']
    # a5: centre the APPLE (not the hand) over the bowl axis; a4 used the demo hand offset, leaving the apple ~35 mm off axis.
    drop = np.r_[bowl_xy + grasp_off[:2], bowl['top'] + release_off[2]]
    drop[2] = max(drop[2], bowl['top'] + .03)
    # IK can fail for far bowl positions with the demo release orientation; fall back to the grasp orientation, then a higher drop.
    for rot, lift in ((opened['rotation'], .0), (R, .0), (R, .04)):
        try:
            d2 = drop + [0, 0, lift]
            io.move(arm, d2 + [0, 0, .03], rot, 1.5)
            io.move(arm, d2, rot, .8)
            drop = d2
            break
        except Exception as e:
            io.log('drop_retry', err=str(e)[:80])
    else:
        raise RuntimeError('drop unreachable')
    io.grip(arm, 1., .6)
    for rot, h in ((opened['rotation'], .1), (R, .1), (R, .06), (R, .03)):
        try:
            io.move(arm, drop + [0, 0, h], rot, .8)
            break
        except Exception as e:
            io.log('retreat_retry', err=str(e)[:80])
    io.wait(2.5)
    io.observe()
