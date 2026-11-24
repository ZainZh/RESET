# sweep_r33_ctl (r33 = r32 + (A1) wrap point chosen among head-axis offsets {+16, +10, 0} mm from the head centre (all carry the brush in brush_wrap_probe) as the one farthest from the perceived cube that has a feasible chain; same rule for every seed; (A2) when every repush variant is unreachable, SHORT pushes: the sweep end is limited to 3 cm past the current cube along the push (up to 3 short pushes, each re-perceived); r32 = r31 + a repush whose sweep is unreachable (IK error) is retried with the other head sense and pan-mouth target offsets (0, +-2 cm), each from a lifted safe pose (r31 smoke: seeds 2 and 4 stopped after one IK-failed repush with the cube 0.6-2.2 cm past the mouth); r31 = r22 + pan wrap pinch offset dx -10 mm, dz -18 mm (pan_grid_probe s7: both left groups on the pan by 1 s and yaw -9..-10 deg; r22's dx 0 dz -14 gave thumb only until the pan turned -15 deg) + 3.5 s settle before the mouth measurement (sweep_r22_pandiag: yaw still moving at r22's measurement) + brush pre-shape fully open (r28); r22 = r21 + repush threshold INSIDE_MIN 3.5 cm past the perceived mouth (r21 cube_in_pan-only fails 8, 13 stopped at 1.5-1.9 cm, final local y -0.107..-0.118 (region needs |y|<= .120 with all vertices); successes ended at local y -0.055..-0.080); r21 = r20 + brush closure back to r15's simultaneous .8 s wrap ('fast'; same TCP in r15 s7 lifted the brush into the hand, r20 'fingers' staged closure left it on the table in 7/8); r20 = r19 + brush wrap TCP 17 mm below the perceived head top: perceiving before the pan grasp (r17+) gives the true top (.0767 vs .0702 behind the arm), so r19 at -10 mm sat 7 mm too high and lost the brush in 8/8 (brush_wrap_probe: -14..-20 mm carried 8.0-8.3/8.2 cm); r19 = r18 + (A) after the left wrap the pan mouth is re-perceived: SAM3 pan mask points, low-y boundary per x bin, line fit -> mouth centre and inward normal; push target = mouth centre + FLOOR_IN along the normal (the wrap turns the pan ~-18 deg, r17 still aimed at the unturned pan); (B) brush closure default 'fingers' (jump 2-3 cm vs 4-9 cm); (C) sweep step .7 s and no repush after REPUSH_T s (r18 hit the 40 s cap); r18 = r17 + staged brush wrap selected by env BRUSH_STAGE (batch constant): 'slow' = 2.0 s simultaneous, 'thumb' = thumb to its target first (0.8 s) then fingers (1.5 s), 'fingers' = fingers first then thumb; r17 s9/s14: the .8 s simultaneous closure threw the brush 9 cm / turned it 21 deg before any carry; r17 = r16 + cube and brush are perceived BEFORE the left hand moves to the pan (r16 s7: with the wrapped left arm in view the brush mask/axis was wrong, near end read (-.194,-.504) vs ~(-.156,-.618), and the hand missed the brush); r16 = r15 + left hand closes on the pan handle with a WRAP posture (env PAN_WRAP 'f,tm,tp', batch constant; same disclosed contact-prior family as the brush wrap) after descending PAN_WRAP_DZ, instead of the hand-module pinch: r15 pan contact was '-' right after closing in 8/8 seeds; r15 = r13 + head yawed so its tip leads the push and the hand trails: the unit vector hand->contact is e = cos(A) p + sin(A) d (d push direction, p perpendicular, side = flip), A from LEAD_ANGLES chosen by the IK/margin chain search; r10-r14 kept the head long axis perpendicular to d, so the wrapping fingers/thumb on the leading side touched the cube (no_robot_cube in 4-5 of 7 seeds, index/thumb proximal); contact point 8 mm from the tip; r13 = r12 + wrap 10 mm below the perceived head top (was 20) so the palm/index proximal ride above the cube top (r12: robot-cube contact in 4/6 executed seeds, palm and index proximal); r12 = r10 + sweep plan search over the target point along the pan mouth (TOX 0, -3, +3, -5, +5 cm) and both head-axis senses (FLIP), as in the reach scan that gave 20/20; the chosen target/sense is used for the sweep and every repush (r10 broad smoke: 7/13 'no reachable brush grasp + sweep chain'); r10 = r9 + right-hand WRAP posture on the brush head (disclosed contact prior, approved by the): grasp at the head centre, TCP 20 mm below the perceived head top, then public joint targets fingers proximal 1.40 rad, thumb metacarpal 1.06, thumb proximal 1.00 (all inside actuator/joint ranges; hand model, force and limits untouched) instead of the hand-module pinch closure (brush_wrap_probe: brush carried 7.9/8.2 cm with thumb + digits, vs <= 2.9 cm for every pinch); push with the head end farther from the hand; r9 = r8 + brush TCP 15 mm below the PERCEIVED head top: the perceived top (95th pct of masked depth points, camera-side faces) reads ~11 mm below the true top (offline comparison seeds 7, 12), so r8 at -26 mm sat ~37 mm under the true top and only the thumb touched; r8 = r7 + the pan is located by perception, not by the demo TCP: SAM3 'dustpan'/'grey dustpan'/'black dustpan' mask, handle end = 99.5th percentile y, pan centre x = median x of the body points; the pinch target keeps the demo-derived offset of the handle plate from those features (P1' = demo left TCP minus demo pan features, b3 demo frame: x = feature_x + .025, y = ymax - .095); mouth and floor-in follow the same shift (r7 b3_sweep_v1: the demo TCP pointed 10 cm beside the moved pan, pan never grasped); r7 = r6 + for world b3_sweep_v1 and the clean park: brush pinch at the head centre region (s = 35/80 mm from the near end, cube-far first), TCP 26 mm below the perceived head top, pad offset 0, top-down then tilt 30 along the head, both pad signs; the head end NEAR the grasp pushes (push point = the other end); full finger closure (right squeeze .5, trusted-side profile) from brush_static_r3 (seed 7: 20 clean two-group head pinches, all at closure 1.0, dz -14..-26 mm); r6 = r5 + brush: approach from 7 cm straight above only after the hand is moved over the grasp point at 7 cm (no diagonal descent next to the cube), grasp point = head end farther from the cube first, slower carry (1.5x) after closing; r5 = r4 + pinch offsets from the offline static contact probes (hand-to-tool geometry only, no per-seed value): pan TCP +5 mm along the pad axis, no slide (pan_contact_probe_r1: both held groups at dz -16 mm, dx >= +4 mm); brush TCP 22 mm below the perceived head top and +4 mm along the pad axis, pad-axis sign fixed by the static result (brush_contact_probe_r1: both groups only at dz <= -22 mm, sg +1); r4 = r3 + pan: descend with the fixed thumb pad 2.5 mm outside the handle's -x face, then slide +x 6 mm so it presses, then close (r3 shifted the wrong way); brush: pre-shape opening .5 and pad-axis sign with the fingers pointing away from the cube (r3: open index tip hit the cube, hand stopped 2.8 cm high); r3 = r2 + pan pinch 4 mm toward -x so the fixed thumb pad presses the handle side (r2: only the index touched), brush approach/close logged; r2 = r1 + pan grasp logged at approach/close, PAN_Z_STEPS tries lower pinch heights until the left hand closes on the handle (logged, encoder-only check: index proximal stops short of the commanded closure)): G1 sweep_objects controller (runtime side: uses only io = rendered RGB-D, calibration, encoders, robot-only model joint ranges, demo priors, SAM3).
# Shared skill (from xTrainer w9, semantics unchanged): hold the dustpan with the left hand, perceive cube and brush with SAM3 ('orange cube', 'orange stick'),
# grasp the brush head with the right hand at one end, sweep the cube with the other end of the head toward the pan (head long axis perpendicular to the push),
# lift and back off, re-observe the cube and repush (max 2) while it is not past the mouth; keep both tools held to the end.
# G1-specific plan (this file): left pan grasp = top-down pinch across the flat handle plate at a demo-derived point; right brush grasp orientation chosen from a fixed
# candidate list (approach tilt along the head axis, pad axis across the head) by robot-only IK joint margin over the whole grasp+sweep chain.
# Demo-derived priors (listed in NOTES.md): P1 pan grasp point from the demo left TCP at t0 (+12 mm along the handle, -16 mm height); P2 mouth 0.138 m from that TCP
# along -y (w9); P3 brush head is the -x end (fallback only, w9); P4 sweep floor-in 55 mm (w9).
import math
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from PIL import Image
from rsi import perception as P, perception_geo as G, perception_sam3 as S
from drawer_r8_ctl import Rax, safe_move

UP = np.array([0., 0., 1.])
MOUTH_DY = -.138          # P2
FLOOR_IN = .055           # P4
PAN_OFF = np.array([-.005, .012, -.016])
PAN_SLIDE = 0.
BRUSH_DZ, BRUSH_PAD = -.017, 0.
WRAP = (1.40, 1.06, 1.00)  # fingers proximal, thumb metacarpal, thumb proximal (rad)
BRUSH_PRE_OPEN = 1.0   # P1 (G1 pinch frame on the flat handle plate)
SWEEP_STEP_SEC = .7
REPUSH_T = 27.
OVERSHOOT = .010
HEAD_LEN = .115
GRIP_NEAR, GRIP_FAR = .0575, .0575        # pinch point along the head from its near (tip) end
PUSH_NEAR, PUSH_FAR = .008, .107
BRUSH_PROMPT, CUBE_PROMPT = 'orange stick', 'orange cube'
RETRACT_UP, RETRACT_BACK = .040, .060
INSIDE_MIN = .035
REOBS_SCORE, REOBS_N, REOBS_SPAN = .10, 100, .075
MAX_REPUSH = 2
MIN_MARGIN = .04
TOX = (0., .03, -.03)
LEAD_ANGLES = (60., 45., 75., 30.)
SW = {'tox': 0., 'flip': False, 'A': 0.}
MN = {'n': np.array([0., 1.])}


def axis_for(d, flip, A, sgn):
    # brush axis (near->handle sense) so that the hand->contact vector leads the push by A degrees; sgn = sign(contact_s - grasp_s)
    p = np.array([d[1], -d[0]]) * (-1. if flip else 1.)
    e = math.cos(math.radians(A)) * p + math.sin(math.radians(A)) * d
    return sgn * e
PAN_Z_STEPS = (0., -.008, -.014)


def yaw_of(v):
    return float(np.arctan2(v[1], v[0]))


def rotz(a):
    return Rot.from_euler('z', a).as_matrix()


def fit_plane(obs, iters=4, band=.008):
    p, _, _ = G.safe_points(obs)
    z0 = P.table_height(obs)
    w = p[(p[:, 0] > G.FIT_X[0]) & (p[:, 0] < G.FIT_X[1]) & (p[:, 1] > G.FIT_Y[0]) & (p[:, 1] < G.FIT_Y[1])]
    c = np.array([0., -.5, z0]); n = np.array([0., 0., 1.])
    sel = np.abs(w[:, 2] - z0) < .015
    for _ in range(iters):
        near = w[sel]
        if len(near) < 200:
            break
        c = near.mean(0)
        _, _, vt = np.linalg.svd(near[::5] - c, full_matrices=False)
        n = vt[-1] * np.sign(vt[-1][2])
        sel = np.abs((w - c) @ n) < band
    return c, n


def masked_points(obs, mask, plane, lo=.003, hi=.08):
    p, ys, xs = G.safe_points(obs)
    h = G.height(p, plane)
    k = mask[ys, xs] & (h > lo) & (h < hi)
    return p[k]


def instances(proc, obs, prompt):
    state = proc.set_image(Image.fromarray(obs['rgb']))
    masks, scores = S._prompt(proc, state, prompt)
    order = np.argsort(-np.asarray(scores)) if len(scores) else []
    return [(masks[i], float(scores[i])) for i in order]


def find_cube(obs, plane, proc, min_score=.4, min_n=20, max_span=.06):
    for m, s in instances(proc, obs, CUBE_PROMPT):
        q = masked_points(obs, m, plane, hi=.06)
        if s >= min_score and len(q) >= min_n and np.ptp(q[:, :2], 0).max() < max_span:
            return {'center': np.r_[np.median(q[:, :2], 0), np.median(q[:, 2])], 'score': s, 'n': len(q)}
    return None


def perceive_cube(io):
    obs = io.observe()
    return find_cube(obs, fit_plane(obs), S.get_proc(), REOBS_SCORE, REOBS_N, REOBS_SPAN)


def perceive(io):
    obs = io.observe()
    plane = fit_plane(obs)
    proc = S.get_proc()
    cube = find_cube(obs, plane, proc)
    if cube is None:
        raise RuntimeError('perception: SAM3 cube not found')
    brush = None
    for m, s in instances(proc, obs, BRUSH_PROMPT):
        q = masked_points(obs, m, plane)
        if len(q) < 80:
            continue
        xy = q[:, :2] - q[:, :2].mean(0); _, v = np.linalg.eigh(np.cov(xy.T)); ax = v[:, -1]
        pr = xy @ ax
        if np.ptp(np.percentile(pr, [1, 99])) > .15 and s >= .2:
            brush = (q, ax, s); break
    if brush is None:
        raise RuntimeError('perception: SAM3 brush not found')
    q, ax, bscore = brush
    c = q[:, :2].mean(0); pr = (q[:, :2] - c) @ ax; perp = (q[:, :2] - c) @ np.array([-ax[1], ax[0]])
    lo, hi = np.percentile(pr, [1, 99]); third = (hi - lo) / 3

    def width(sel):
        return float(np.ptp(np.percentile(perp[sel], [5, 95]))) if sel.sum() > 15 else 0.
    w_lo, w_hi = width(pr < lo + third), width(pr > hi - third)
    if abs(w_lo - w_hi) > .004:
        head_at_lo = w_lo > w_hi; source = 'mask_width_profile'
    else:
        head_at_lo = ax[0] > 0; source = 'prior_P3_head_is_minus_x'
    if not head_at_lo:
        ax = -ax; pr = -pr; lo, hi = -hi, -lo
    near = c + ax * lo
    head_pts = q[((q[:, :2] - near) @ ax >= 0) & ((q[:, :2] - near) @ ax <= HEAD_LEN)]
    top = float(np.percentile(head_pts[:, 2], 95)) if len(head_pts) > 20 else float(np.percentile(q[:, 2], 95))
    io.log('perceived', cube=cube['center'], cube_score=cube['score'], brush_near=near, brush_axis=ax, brush_len=hi - lo, head_top=top,
           brush_score=bscore, n_brush=len(q), head_end_source=source)
    return cube['center'], near, ax, top, plane


def margin_chain(io, side, legs):
    # IK through [(p, R)] from the current encoders (3 interpolation points per leg); worst arm-joint margin (robot-only model), first leg interior feasibility only
    m = io.adapter.hand.m
    enc = io._enc()
    T = io.adapter.tcp(enc, side)
    p0 = T[:3, 3].copy()
    worst = 9.
    for i, (p, R) in enumerate(legs):
        p = np.asarray(p, float)
        for f in ((.5, 1.) if i else (1.,)):
            try:
                q = io.adapter.plan(enc, side, p0 + (p - p0) * f, R)
            except Exception:
                return None
            enc.update(q)
            for n, v in q.items():
                lo, hi = m.jnt_range[m.joint(n).id]
                worst = min(worst, v - lo, hi - v)
        p0 = p
    return worst


def pan_features(io):
    obs = io.observe()
    plane = fit_plane(obs)
    proc = S.get_proc()
    best = None
    for prompt in ('dustpan', 'grey dustpan', 'black dustpan'):
        for m, s in instances(proc, obs, prompt):
            q = masked_points(obs, m, plane, lo=.002, hi=.06)
            if s >= .3 and len(q) > 300 and (best is None or s > best[1]):
                best = (q, s, prompt)
        if best is not None:
            break
    if best is None:
        raise RuntimeError('perception: SAM3 dustpan not found')
    q, s, prompt = best
    ymax = float(np.percentile(q[:, 1], 99.5))
    xb = float(np.median(q[q[:, 1] < ymax - .12][:, 0]))
    io.log('pan_perceived', ymax=ymax, x_body=xb, score=s, prompt=prompt, n=len(q))
    return xb, ymax


# demo-derived (P1'): demo left TCP relative to the pan features measured on the b3 demo layout (pan_percept_probe_b3: x_body -.0245, ymax -.2499 vs demo TCP)
def hold_pan(io):
    L = io.demo['home']['left']
    xb, ymax = pan_features(io)
    shift = np.array([xb - (-.0245), ymax - (-.2499), 0.])
    io.log('pan_shift', shift=shift)
    L = L.copy(); L[:3, 3] = L[:3, 3] + shift
    g = L[:3, 3] + PAN_OFF
    cands = []
    for tilt in (0., 20., -20., 35.):
        a = np.array([math.sin(math.radians(tilt)), 0., -math.cos(math.radians(tilt))])
        for sg in (1., -1.):
            cands.append(('t%+d_x%+d' % (tilt, sg), Rax(a, (sg, 0., 0.))))
    best = None
    for nm, R in cands:
        w = margin_chain(io, 'left', [(g - R[:, 2] * .06, R), (g, R)])
        if w is not None and w >= MIN_MARGIN:
            best = (nm, R, w); break
    if best is None:
        raise RuntimeError('pan handle unreachable')
    nm, R, w = best
    io.log('pan_grasp', cand=nm, margin=w, target=g)
    io.grip('left', 1., .3)
    safe_move(io, 'left', g - R[:, 2] * .06, R, 1.5)
    safe_move(io, 'left', g, R, .8)
    gs = g + R[:, 1] * PAN_SLIDE  # along the pad axis (+x): thumb pad moves into the handle's -x face
    safe_move(io, 'left', gs, R, .6)
    io.log('pan_at', tcp=io.tcp('left')[:3, 3])
    pw = [float(v) for v in os.environ.get('PAN_WRAP', '1.40,0.80,1.00').split(',')]
    safe_move(io, 'left', gs + [0, 0, float(os.environ.get('PAN_WRAP_DZ', '-.018'))], R, .5)
    wrap(io, 'left', *pw, dur=float(os.environ.get('PAN_WRAP_DUR', '.8')))
    io.wait(.2)
    io.log('pan_closed', index_prox=io._enc()['left_index_proximal_joint'])
    return L[:3, 3]
    io.log('pan_grasp_unconfirmed')
    io.grip('left', 0., .7)
    return L[:3, 3]


def wrap(io, side, f, tm, tp, dur=.8):
    tgt = {'%s_%s_proximal_joint' % (side, k): f for k in ('index', 'middle', 'ring', 'pinky')}
    tgt['%s_thumb_metacarpal_joint' % side] = tm
    tgt['%s_thumb_proximal_joint' % side] = tp
    q0 = {k: io.targets[k] for k in tgt}
    n = max(1, int(dur / io.dt))
    for i in range(n):
        for k, v in tgt.items():
            io.targets[k] = q0[k] + (v - q0[k]) * (i + 1) / n
        io._step()


def pan_mouth(io):
    # mouth edge from the pan mask: lowest y per x bin, robust line fit; returns (mouth centre xy, inward unit normal) or None
    obs = io.observe()
    plane = fit_plane(obs)
    proc = S.get_proc()
    best = None
    for prompt in ('dustpan', 'grey dustpan', 'black dustpan'):
        for m, s_ in instances(proc, obs, prompt):
            q = masked_points(obs, m, plane, lo=.002, hi=.06)
            if s_ >= .3 and len(q) > 300 and (best is None or s_ > best[1]):
                best = (q, s_)
        if best is not None:
            break
    if best is None:
        return None
    q = best[0]
    xs = np.percentile(q[:, 0], [3, 97])
    bins = np.arange(xs[0], xs[1], .008)
    pts = []
    for b in bins:
        k = (q[:, 0] >= b) & (q[:, 0] < b + .008)
        if k.sum() >= 5:
            pts.append((b + .004, float(np.percentile(q[k, 1], 2))))
    if len(pts) < 6:
        return None
    pts = np.array(pts)
    a, c = np.polyfit(pts[:, 0], pts[:, 1], 1)
    res = np.abs(pts[:, 1] - (a * pts[:, 0] + c))
    keep = res < max(.006, np.percentile(res, 60))
    if keep.sum() >= 4:
        a, c = np.polyfit(pts[keep, 0], pts[keep, 1], 1)
    xm = float(np.median(pts[keep, 0])) if keep.sum() >= 4 else float(np.median(pts[:, 0]))
    n = np.array([-a, 1.]); n /= np.linalg.norm(n)
    return np.array([xm, a * xm + c]), n, float(np.degrees(np.arctan(a)))


def brush_rots(ax):
    perp = np.array([-ax[1], ax[0], 0.])
    out = []
    for tilt in (0., 20., 40., -20.):
        a = -UP * math.cos(math.radians(tilt)) + np.r_[ax[:2], 0.] * math.sin(math.radians(tilt))
        for sg in (1., -1.):
            out.append(('t%+d_p%+d' % (tilt, sg), Rax(a, sg * perp)))
    return out


def run(io):
    cube, near, ax, top, plane = perceive(io)
    pl = hold_pan(io)
    mouth = pl + np.array([0, MOUTH_DY, 0])
    target = np.array([pl[0], mouth[1] + FLOOR_IN])
    io.wait(3.5)
    pm = pan_mouth(io)
    if pm is not None and abs(pm[2]) < 45.:
        target = pm[0] + pm[1] * FLOOR_IN
        mouth = np.r_[pm[0], mouth[2]]
        MN['n'] = pm[1]
        io.log('pan_mouth', mouth=pm[0], normal=pm[1], yaw=pm[2], target=target)
    else:
        io.log('pan_mouth_failed', got=None if pm is None else pm[2])
    along = float((cube[:2] - near[:2]) @ ax[:2])
    order = [(GRIP_FAR, PUSH_NEAR), (GRIP_NEAR, PUSH_FAR)] if along < HEAD_LEN / 2 else [(GRIP_NEAR, PUSH_FAR), (GRIP_FAR, PUSH_NEAR)]  # grasp the end farther from the cube
    gz = top + BRUSH_DZ
    chosen = None
    offs = sorted((.016, .010, 0.), key=lambda o: -float(np.linalg.norm(near[:2] + ax[:2] * (.0575 + o) - cube[:2])))
    order = [(.0575 + o, c_) for o in offs for _, c_ in order[:1]]
    io.log('wrap_offsets', order=[round(o[0], 4) for o in order])
    for grasp_s, contact_s in order:
        g0 = np.r_[near[:2] + ax[:2] * grasp_s, gz]
        head_c = near[:2] + ax[:2] * contact_s
        cands = sorted(brush_rots(ax), key=lambda c: float(c[1][:, 1] @ np.r_[cube[:2] - near[:2] - ax[:2] * grasp_s, 0.]) > 0)
        for nm, Rg in cands:
            g = g0 + Rg[:, 1] * BRUSH_PAD
            for tox in TOX:
                for flip, A in [(f_, a_) for a_ in LEAD_ANGLES for f_ in (False, True)]:
                    tg = target + np.array([tox, 0.])
                    d = tg - cube[:2]; dist = float(np.linalg.norm(d)); d = d / dist
                    axis_new = axis_for(d, flip, A, np.sign(contact_s - grasp_s))
                    turn = (yaw_of(axis_new) - yaw_of(ax[:2]) + math.pi) % (2 * math.pi) - math.pi
                    Rs = rotz(turn) @ Rg
                    hin = Rg.T @ (np.r_[head_c, g[2]] - g)

                    def tcp_for(xy, z):
                        return np.r_[xy - (Rs @ hin)[:2], z]
                    start = cube[:2] - d * (.016 + .013 + .012)
                    end = cube[:2] + d * (dist - .029 + OVERSHOOT)
                    legs = [(g + .07 * UP, Rg), (g, Rg), (g + .02 * UP, Rg), (tcp_for(start, g[2] + .02), Rs), (tcp_for(start, g[2] + .002), Rs),
                            (tcp_for(0.5 * (start + end), g[2] + .002), Rs), (tcp_for(end, g[2] + .002), Rs)]
                    w = margin_chain(io, 'right', legs)
                    if w is not None and w >= MIN_MARGIN:
                        chosen = (grasp_s, contact_s, nm, Rg, g, w); SW['tox'], SW['flip'], SW['A'] = tox, flip, A
                        break
                if chosen:
                    break
            if chosen:
                break
        if chosen:
            break
    if chosen is None:
        raise RuntimeError('no reachable brush grasp + sweep chain')
    grasp_s, contact_s, nm, Rg, g, w = chosen
    io.log('brush_grasp', cand=nm, grasp_s=grasp_s, contact_s=contact_s, margin=w, target=g, tox=SW['tox'], flip=SW['flip'], lead=SW['A'])
    io.grip('right', BRUSH_PRE_OPEN, .3)
    safe_move(io, 'right', g + .12 * UP, Rg, 1.5)
    safe_move(io, 'right', g + .07 * UP, Rg, .8)
    safe_move(io, 'right', g, Rg, 1.0)
    io.log('brush_at', tcp=io.tcp('right')[:3, 3])
    st = os.environ.get('BRUSH_STAGE', 'fast')
    fq = io.targets['right_index_proximal_joint']
    tmq, tpq = io.targets['right_thumb_metacarpal_joint'], io.targets['right_thumb_proximal_joint']
    if st == 'fast':
        wrap(io, 'right', *WRAP)
    elif st == 'slow':
        wrap(io, 'right', *WRAP, dur=2.0)
    elif st == 'thumb':
        wrap(io, 'right', fq, WRAP[1], WRAP[2], dur=.8)
        wrap(io, 'right', *WRAP, dur=1.5)
    else:
        wrap(io, 'right', WRAP[0], tmq, tpq, dur=1.5)
        wrap(io, 'right', *WRAP, dur=.8)
    io.wait(.3)
    io.log('brush_closed', tcp=io.tcp('right')[:3, 3])
    T = io.tcp('right'); tcp0 = T[:3, 3].copy(); Rt = T[:3, :3].copy()
    head_c = near[:2] + ax[:2] * contact_s
    head_in_tcp = Rt.T @ (np.r_[head_c, tcp0[2]] - tcp0)
    z_sweep = tcp0[2] + .002
    safe_move(io, 'right', tcp0 + .02 * UP, Rt, .6)

    def sweep(cube_xy, label, max_len=None):
        tg = target + np.array([SW['tox'], 0.])
        d = tg - cube_xy; dist = float(np.linalg.norm(d)); d /= dist
        axis_new = axis_for(d, SW['flip'], SW['A'], np.sign(contact_s - grasp_s))
        turn = (yaw_of(axis_new) - yaw_of(ax[:2]) + math.pi) % (2 * math.pi) - math.pi; Rs = rotz(turn) @ Rt

        def tcp_for(head_xy, z):
            return np.r_[head_xy - (Rs @ head_in_tcp)[:2], z]
        start = cube_xy - d * (.016 + .013 + .012)
        safe_move(io, 'right', tcp_for(start, z_sweep + .02), Rs, 2.2)
        safe_move(io, 'right', tcp_for(start, z_sweep), Rs, .6)
        end = cube_xy + d * (dist - .029 + OVERSHOOT)
        if max_len is not None:
            end = cube_xy + d * min(dist - .029 + OVERSHOOT, max_len)
        n = max(2, int(np.ceil(np.linalg.norm(end - start) / .04)))
        for k in range(1, n + 1):
            safe_move(io, 'right', tcp_for(start + (end - start) * k / n, z_sweep), Rs, SWEEP_STEP_SEC)
        io.wait(.3)
        io.log('swept', label=label, start=start, end=end, target=target)
        T = io.tcp('right'); up = T[:3, 3] + RETRACT_UP * UP
        safe_move(io, 'right', up, T[:3, :3], .8)
        try:
            safe_move(io, 'right', up - np.r_[d, 0.] * RETRACT_BACK, T[:3, :3], .8)
        except Exception as e:
            if io.error is not None:
                raise
            io.log('retract_back_failed', err=str(e)[:60])
        io.wait(.3)

    sweep(cube[:2], 'first')
    for k in range(MAX_REPUSH):
        c2 = perceive_cube(io)
        past = None if c2 is None else float((c2['center'][:2] - mouth[:2]) @ MN['n'])
        io.log('reobserved', cube=None if c2 is None else c2['center'], past_mouth=past, repush=k)
        if c2 is None or past >= INSIDE_MIN:
            break
        if io.now() > REPUSH_T:
            io.log('repush_skipped_time', t=io.now())
            break
        done = False
        base = (SW['tox'], SW['flip'])
        for tox, flip in [base, (base[0], not base[1]), (base[0] + .02, base[1]), (base[0] - .02, base[1]), (base[0] + .02, not base[1]), (base[0] - .02, not base[1])]:
            SW['tox'], SW['flip'] = tox, flip
            try:
                sweep(c2['center'][:2], 'repush_%d' % k)
                done = True
                break
            except Exception as e:
                if io.error is not None:
                    raise
                io.log('repush_variant_failed', tox=tox, flip=flip, error=str(e)[:60])
                try:
                    T = io.tcp('right')
                    safe_move(io, 'right', T[:3, 3] + .04 * UP, T[:3, :3], .6)
                except Exception:
                    if io.error is not None:
                        raise
        if not done:
            SW['tox'], SW['flip'] = base
            for j in range(3):
                try:
                    sweep(c2['center'][:2], 'short_%d_%d' % (k, j), max_len=.03)
                    done = True
                except Exception as e:
                    if io.error is not None:
                        raise
                    io.log('short_push_failed', j=j, error=str(e)[:60])
                    break
                c3 = perceive_cube(io)
                if c3 is None:
                    break
                c2 = c3
                if float((c2['center'][:2] - mouth[:2]) @ MN['n']) >= INSIDE_MIN or io.now() > REPUSH_T:
                    break
        if done:
            continue
        try:
            raise RuntimeError('all repush variants unreachable')
        except Exception as e:
            if io.error is not None:
                raise
            io.log('repush_failed', error='%s: %s' % (type(e).__name__, e))
            break
