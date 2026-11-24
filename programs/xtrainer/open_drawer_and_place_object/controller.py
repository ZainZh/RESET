# s1: d9 + SAM-first block search: all SAM3 instances for learned semantic prompts (orange cube first), rejected by the
#     opened cabinet's own mask footprint and by block-like size/height, instead of a fixed 0.21 m disc around (0.10, face y).
# d9: d8 + table height from P.table_height (depth) instead of G.fit_plane
# d8: d7 + colour-free perception (SAM3 text prompts for cabinet and block)
# d7: d6 + higher/deeper drop point with reachable fallbacks
"""drawer s1: left arm grasps the knob with a 30 deg tilted gripper, pulls 13 cm, lets go; right arm picks the block found by
SAM3 (text prompts, no HSV thresholds) and drops it at the centre of the opened drawer.
Semantic cues used: 'wooden cabinet' (material), 'orange cube' / 'small cube' / 'orange block' / 'toy block' (learned colour words).
Demonstration contact priors: left close pose rotation (knob grasp), right close/open pose rotations."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from PIL import Image
from rsi import perception as P
from rsi import perception_sam3 as S
from rsi import perception_geo as G

UP = np.array([0, 0, 1.])
PULL = .13
PROMPTS = ('orange cube', 'small cube', 'orange block', 'toy block')
CAB_MARGIN = .02


def front_face(io, prompt='wooden cabinet'):
    """Drawer front face from depth: SAM3 text-grounded cabinet extent facing the robot (min x); y = extent centre."""
    obs = io.observe()
    w = S.locate_object(S.get_proc(), obs, prompt, plane=G.fit_plane(obs), max_height=.13)
    if w is None:
        raise RuntimeError('drawer not found')
    p = w['points']; z0 = P.table_height(obs)
    x = np.percentile(p[:, 0], 1)
    y = .5 * (np.percentile(p[:, 1], 2) + np.percentile(p[:, 1], 98))
    return np.array([x, y, z0 + .066]), w


def find_block(io, obs, cab_points):
    """Best block-like SAM3 instance over all prompts. Returns (candidate, diagnostics list)."""
    proc = S.get_proc(); plane = G.fit_plane(obs)
    p, ys, xs = G.safe_points(obs); h = G.height(p, plane)
    state = proc.set_image(Image.fromarray(obs['rgb']))
    lo = cab_points[:, :2].min(0) - CAB_MARGIN; hi = cab_points[:, :2].max(0) + CAB_MARGIN
    diag = []; best = None
    for prompt in PROMPTS:
        masks, scores = S._prompt(proc, state, prompt)
        for m, s in zip(masks, scores):
            if s < .3:
                continue
            sel = m[ys, xs] & (h > .005) & (h < .06)
            n = int(sel.sum())
            if n < 40:
                diag.append({'prompt': prompt, 'score': float(s), 'reject': 'few_points', 'n': n}); continue
            q = p[sel]; hq = h[sel]
            ext = q[:, :2].max(0) - q[:, :2].min(0)
            top = float(np.percentile(hq, 95))
            c = np.r_[np.median(q[:, :2], 0), np.median(q[:, 2])]
            if np.all(c[:2] > lo) and np.all(c[:2] < hi):
                diag.append({'prompt': prompt, 'score': float(s), 'reject': 'inside_cabinet_footprint', 'center': c}); continue
            if ext.max() > .09 or not (.015 <= top <= .05):
                diag.append({'prompt': prompt, 'score': float(s), 'reject': 'not_block_like', 'extent': ext, 'top': top}); continue
            if s < .5 and n < 100:
                diag.append({'prompt': prompt, 'score': float(s), 'reject': 'low_score_few_points', 'n': n}); continue
            cand = {'points': q, 'top': float(np.percentile(q[:, 2], 95)), 'center': c, 'score': float(s), 'n': n, 'prompt': prompt}
            diag.append({'prompt': prompt, 'score': float(s), 'accept': True, 'center': c, 'n': n, 'extent': ext})
            if best is None or cand['score'] > best['score']:
                best = cand
        if best is not None and best['score'] >= .7:
            break
    return best, diag


def lift(io, t, R):
    err = None
    for dz in (.16, .12, .08, .05):
        try:
            io.move('right', t + dz * UP, R, 1.0)
            return
        except Exception as e:
            err = e
            io.log('lift_unreachable', dz=dz, error=str(e)[:80])
    raise err


def retreat(io, d, R):
    for off in ([0, 0, .10], [0, 0, .06], [0, 0, .03], [-.08, 0, .03], [-.08, 0, 0]):
        try:
            io.move('right', d + np.array(off), R, .8)
            return
        except Exception as e:
            io.log('retreat_unreachable', off=off, error=str(e)[:60])


def run(io):
    ev = io.demo['events']
    lc = next(e for e in ev if e['arm'] == 'left' and e['kind'] == 'close')
    rc = next(e for e in ev if e['arm'] == 'right' and e['kind'] == 'close')
    ro = next(e for e in ev if e['arm'] == 'right' and e['kind'] == 'open')
    knob, _ = front_face(io)
    io.log('knob', knob=knob)
    R = Rot.from_euler('y', -30, degrees=True).as_matrix() @ lc['rotation']
    g = knob + [-.002, 0, 0]
    io.grip('left', 1., .3)
    io.move('left', g - R[:, 2] * .06 + [0, 0, .03], R, 1.5)
    io.move('left', g, R, 1.0)
    io.grip('left', 0., .6)
    io.move('left', g + [-PULL, 0, 0], R, 2.5)
    io.grip('left', 1., .5)
    io.move('left', g + [-PULL - .05, 0, .08], R, 1.0)
    io.home('left', 2.0)
    face, cab = front_face(io, 'wooden cabinet with open drawer')
    io.log('opened_face', face=face, opened_m=float(knob[0] - face[0]))
    obs = io.observe()
    c, diag = find_block(io, obs, cab['points'])
    io.log('block_candidates', candidates=diag)
    if c is None:
        raise RuntimeError('block not found by SAM3 prompts')
    c['table'] = P.table_height(obs)
    io.log('block_perceived', center=c['center'], n=c['n'], prompt=c['prompt'], score=c['score'])
    top = c['points'][c['points'][:, 2] > c['top'] - .008]
    t = np.r_[top[:, :2].mean(0), c['table'] + .021]
    io.grip('right', 1., .3)
    io.move('right', t + .10 * UP, rc['rotation'], 1.5)
    io.move('right', t, rc['rotation'], 1.0)
    io.grip('right', 0., .6)
    lift(io, t, rc['rotation'])
    d = None
    for dx, dz in ((.07, .14), (.06, .12), (.05, .11)):
        cand = np.r_[face[0] + dx, face[1], c['table'] + dz]
        try:
            io.move('right', cand + .05 * UP, ro['rotation'], 1.5)
            io.move('right', cand, ro['rotation'], .8)
            d = cand
            break
        except Exception as e:
            if type(e).__name__ == 'NumericalFailure':
                raise
            io.log('drop_unreachable', dx=dx, dz=dz, error=str(e)[:60])
    if d is None:
        raise RuntimeError('no reachable drop point')
    io.grip('right', 1., .6)
    retreat(io, d, ro['rotation'])
    io.home('right', 1.5)
    io.wait(1.5)
    io.observe()
