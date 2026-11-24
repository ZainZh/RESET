'''stack_cups c6 (c5 + observation-only retries of the in-hand measurement): s5 perception and roles (generic SAM3 prompt 'cup', roles from demo geometry). The drop centres the held cup over the observed target axis using the in-hand offset measured from the lifted cup (SAM3 mask + depth + encoder TCP) instead of the commanded grasp target. Falls back to the commanded offset if the measurement fails sanity checks. Stage markers only call io.log.'''
import numpy as np
from itertools import permutations
from PIL import Image
from rsi import perception as P
from rsi import perception_sam3 as S
from rsi import perception_geo as G

UP = np.array([0, 0, 1.])


def find_cups(obs, prompts=('cup', 'paper cup'), min_score=.4, min_pts=100, max_height=.14):
    plane = G.fit_plane(obs)
    p, ys, xs = G.safe_points(obs)
    h = G.height(p, plane)
    proc = S.get_proc()
    state = proc.set_image(Image.fromarray(obs['rgb']))
    pairs = []
    for pr in prompts:
        mk, sc = S._prompt(proc, state, pr)
        pairs += list(zip(mk, sc))
    res = []
    for m, s in sorted(pairs, key=lambda x: -x[1]):
        if s < min_score:
            continue
        sel = m[ys, xs] & (h > .005) & (h < max_height)
        if sel.sum() < min_pts:
            continue
        q = p[sel]
        c = np.median(q[:, :2], 0)
        if any(np.linalg.norm(c - r['c']) < .03 for r in res):
            continue
        res.append({'c': c, 'points': q, 'top': float(np.percentile(q[:, 2], 95)), 'score': float(s)})
    return res


def axis(c):
    rim = c['points'][c['points'][:, 2] > c['top'] - .012]
    xy, r = P.fit_upright_cylinder(rim)
    return np.asarray(xy, float), c['top'], r


def held_offset(io, arm, obs, held0, prompts=('cup', 'paper cup'), min_score=.4):
    '''TCP xy minus the rim-axis xy of the cup seen in the hand (observation + encoders only).'''
    tcp = io.tcp(arm)[:3, 3]
    cups = find_cups(obs, prompts=prompts, min_score=min_score, max_height=.32, min_pts=60)
    exp = tcp[:2] - held0
    cand = [c for c in cups if np.linalg.norm(c['c'] - exp) < .03 and c['top'] > .16]
    if not cand:
        return None, 'no held cup among %d detections (dist_to_expected, top): %s' % (len(cups), [(round(float(np.linalg.norm(c['c'] - exp)), 3), round(c['top'], 3)) for c in cups])
    c = min(cand, key=lambda c: np.linalg.norm(c['c'] - exp))
    xy, top, r = axis(c)
    if not .025 < r < .045:
        return None, 'implausible rim radius %.4f' % r
    return tcp[:2] - xy, 'ok top=%.3f r=%.4f n=%d score=%.2f' % (top, r, len(c['points']), c['score'])


def measure_held(io, arm, held0, t):
    '''In-hand offset with observation-only retries: default SAM3 pass, then a low-score pass with an extra
    generic prompt (same geometric gates), then the same two passes after lifting 3 cm higher.'''
    tries = []
    for lift in (0., .03):
        if lift:
            io.move(arm, t + (.16 + lift) * UP, io.tcp(arm)[:3, :3], .5)
        obs = io.observe()
        for kw in (dict(), dict(prompts=('cup', 'paper cup', 'plastic cup'), min_score=.15)):
            h, info = held_offset(io, arm, obs, held0, **kw)
            tries.append(info)
            if h is not None:
                return h, 'pass %d: %s' % (len(tries), info)
    return None, ' | '.join(tries)

def run(io):
    ev = io.demo['events']
    close = next(e for e in ev if e['kind'] == 'close')
    opened = next(e for e in ev if e['kind'] == 'open' and e['time'] > close['time'])
    arm, R = close['arm'], close['rotation']
    names = list(close['objects'])
    gname = min(names, key=lambda n: np.linalg.norm(close['objects'][n][:2] - close['position'][:2]))
    tname = next(n for n in names if n != gname)
    obs = io.observe()
    cups = find_cups(obs)
    if len(cups) < 2:
        raise RuntimeError('found %d cups' % len(cups))
    cups = cups[:3]
    o0 = io.demo['objects0']
    best = min(permutations(cups, 2), key=lambda pr: np.linalg.norm(pr[0]['c'] - o0[gname][:2]) + np.linalg.norm(pr[1]['c'] - o0[tname][:2]))
    pxy, ptop, pr_ = axis(best[0])
    bxy, btop, br = axis(best[1])
    io.log('perceived', grasp=pxy, grasp_top=ptop, grasp_r=pr_, target=bxy, target_top=btop, target_r=br, n_cups=len(cups), scores=[c['score'] for c in cups])
    off = close['position'] - close['objects'][gname]
    t = np.r_[pxy + off[:2], close['position'][2]]
    io.grip(arm, 1., .3)
    io.move(arm, t + .10 * UP, R, 1.5)
    io.move(arm, t, R, 1.0)
    io.grip(arm, 0., .6)
    io.log('stage_grasped')
    io.move(arm, t + .16 * UP, R, 1.0)
    held = t[:2] - pxy
    commanded = held.copy()
    hobs, info = None, ''
    try:
        hobs, info = measure_held(io, arm, held, t)
    except Exception as e:
        info = '%s: %s' % (type(e).__name__, e)
    used = 'commanded'
    if hobs is not None and np.linalg.norm(hobs - held) < .03:
        held, used = hobs, 'measured'
    io.log('held_offset', commanded=commanded, measured=hobs, used=used, info=info)
    drop = np.r_[bxy + held, opened['position'][2] + .01]
    io.move(arm, drop + .06 * UP, opened['rotation'], 1.5)
    io.move(arm, drop, opened['rotation'], 1.0)
    io.log('stage_pre_release')
    io.grip(arm, 1., .6)
    io.log('stage_post_open')
    for h in (.10, .06, .03):
        try:
            io.move(arm, drop + h * UP, opened['rotation'], .8)
            break
        except Exception as e:
            if type(e).__name__ in ('NumericalFailure', 'Limit'):
                raise
            io.log('lift_retry', h=h)
    io.log('stage_post_lift')
    io.wait(2.0)
    io.observe()
