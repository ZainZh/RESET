# cups_r28_ctl (r27 + the measure-and-correct step is repeated (POST_ITERS 2) AFTER lowering into the blue cup, just before the hand opens; corrections under MIN_CORR 1.5 mm are skipped. Why: r27 smoke seeds 7/14: the pre-lowering correction moved the cup toward the axis (truth, post hoc: -9.1 -> -6.6 mm) but the lowering moved it back (-10.3 mm at release), so the last measurement must be at the release pose).
# cups_r27_ctl (r26 + partial correction: if the full measured correction is not IK-reachable, 0.6 and then 0.3 of it are tried (CORR_FRACS code constant). Why: r26 smoke seed 7 measured +5 mm (true need ~+9 mm, same direction) but the full shift was unreachable and the cup landed 6.2 mm off).
# cups_r26_ctl (r25 + the IK reachability check of the correction is guarded: r25 smoke seed 12 raised fk_invalid:encoder_out_of_limit:left_wrist_roll_joint from adapter.tcp at the start of that check (wrist roll physically past its limit under contact, not latched), so the correction is skipped and logged instead).
# r25 header: # cups_r24_ctl (r22 + grasp height .062 m above the cup bottom tried first: GHS (.062, .045, .055), code constant). Why: diag_r22log (diag_cups_r22log_ctl, seeds 7/12/14/19) shows the final off-centre offset is set during the drop, not the retreat; seeds 12/14 opened with the cup ~55-66 mm above its nested height because the insertion was limited by 'room' (hand points near the blue axis must stay above the blue rim) at grasp height .045/.055. A higher grasp on the upward-flaring purple cup leaves more cup below the fingers, so the cup can go deeper before the hand reaches the rim.
# cups_r22_ctl (r21 + slow hand opening at the release: OPEN_T 2.0 s instead of 0.8 s, a code constant, not fitted to object truth). Why: in the clean-start r21 batch (cupsc_r21_t540_dev0_19_b3, 12/20) seeds 7, 14 and 19 were fully inserted, upright and released, and failed only the nesting check: final cup-axis offset 7.5-8.1 mm, radial clearance -1.0 to -2.7 mm against the 1 mm tolerance; passing seeds ended at 2.6-7.4 mm. Hypothesis: a slower opening gives a smaller lateral kick on the cup as it drops to the cavity floor. Hand-point file is read from ../mug_cups_common/ (read-only).
# cups_r21_ctl (r20 with the fallback cutoff applied only in the last SHORT sweep, so a full-insertion candidate at a larger SHORT beats a no-insertion one at SHORT 0
# (offline r20: seeds 6 and 7 returned the no-insertion fallback, diag_feas_r5_cups_r20_5_6_7_8_9.log; r17 passed both with SHORT 15 mm)). r20 = (r17 + (a) geometric thumb-side shift of the held cup: the thumb is fixed and the four fingers close, so the cup ends against the thumb pad; its axis then sits
# (CONTACT_GAP/2 - r_purple(GH)) from the pinch frame toward the thumb (CONTACT_GAP 78 mm from the mug_wide calibration, r_purple from the listed cup radii; ~7.5 mm). No object-truth constant.
# (b) search order: release shortfall SHORT is the outer loop (smallest first) and pick chains are cached, so a candidate that reaches the blue axis is preferred over a shortened release
# (r17 failures 1, 2, 9, 11, 17 tipped on the rim after releases 10-15 mm short). (c) SAM3 client reconnect when a prompt returns no instance at all (r19 seed 12: six empty replies in a row). r17 = (r16 + no io.tcp() on the release path: once the wrist roll is physically past its limit under contact, io.tcp raises fk_invalid, which r16 hit in its own log calls (5 of 15 scored seeds: 1, 3, 13, 14, 19) so the hand never opened; logging uses safe_tcp and the hand always opens). r16 = (r15 + the lowering into the blue cup is guarded: if IK refuses during the lowering (r15 seed 6: ik_invalid:encoder_out_of_limit:left_wrist_roll_joint with the cup part-inserted, uncaught, hand never opened) the controller logs it and releases where it is instead of aborting with the cup held). r15 = (r14 with the HELD_* constants REMOVED: they were fitted from post-hoc simulator object poses, which the method manual section 8 forbids in any runtime parameter; r14 results are kept but are not freeze evidence. No constant here comes from object truth. Also a chain-trial cap MAX_AFTER_FALLBACK so a seed without a full-insertion candidate ends with a record instead of a hard timeout.) r14 = (r13 + (a) held-offset model of the wrap grasp, a G1 grasp constant fitted from DEV smoke physics (r11 seeds 0,7,8,13,17; r13 seeds 0,5,7,13): after closing, the cup axis sits
# horizontally +5 mm further along the approach than planned and (8 mm + 0.23 x DL) toward the thumb (the thumb is fixed, the fingers push the cup); the release target puts THIS predicted axis on the blue axis;
# (b) candidates whose release cannot be lowered by at least MIN_INSERT are skipped while any other candidate remains (r13 seed 7 accepted 0 mm insertion and the cup tipped on the rim). r13 = (r11 + (a) partial insertion: the release pose is lowered until the robot-model hand points near the blue cup (within R_BLUE+12 mm of its axis) are 4 mm above the perceived blue rim,
# so the purple bottom is already inside the cavity when the hand opens (r9/r11/r12 dropped a cup held ~10 deg tilted from 12 mm above the rim; it landed leaning, radial clearance -1.2 to -2.9 mm,
# diag_settle_r1); (b) DLS tried from 0 (cup axis at the pinch frame) up to 30 mm; (c) perception retries with sim-time waits and raw-score logging (r8/r12 'found 0 cups' was intermittent with the same scene).
# r12 measure_held is not used (its rim-based estimate missed the in-hand tilt). r11 = G1 stack_cups controller (purple cup into blue cup), wide wrap grasp with the cup axis AHEAD of the pinch frame. Runtime side: uses only io (rendered RGB-D via SAM3 masks + depth, encoders, demo priors) and the robot model. Never reads object truth, scorer or contacts.
# Shared skill (xTrainer s5/c6, not edited in place): generic SAM3 'cup' masks, roles by matching the demo initial positions, upright cylinder fit on the rim -> pick purple -> carry over blue -> release -> retreat.
# G1-specific plan, from r1-r10 (all kept): with the pinch frame on the cup axis the arm reaches the release pose for only 11/20 DEV seeds (blue axis beyond x ~0.13 m, diag_feas_r3 / diag_feas_r3_cups_r10.log); higher drops do not help.
# r11 uses the 'mug_wide' grasp shape (cups_r11_grasp -> calib_wide_r1.json; 90 mm open, 42 mm at opening 0) and holds the cup axis DL ahead of the pinch frame along the approach (DL up to 30 mm, as mug_r4 which
# passed 16/20), so the cup reaches farther than the hand. Candidates are filtered with robot-model hand points (hand_pts_r2_mug_r1_grasp.npz) against the perceived purple-cup primitive and the table plane,
# then the interpolated IK chain with a joint-limit margin. Release: cup bottom CLEAR above the perceived blue rim, then open and back off along the approach. No joint-space home, no final observe.
# Contact priors (listed): mug_wide grasp shape; purple height 70.5 mm, radii 26.7 mm (base) to 34.2 mm (rim); blue rim radius 45.1 mm; demo close event (which cup is grasped); code-constant grids.
import math
import os
import time
import numpy as np
from itertools import permutations
from PIL import Image
from rsi import perception as P
from rsi import perception_sam3 as S
from rsi import perception_geo as G
import g1_common_r2 as C
from rsi.g1_suite.runtime.public_io import waypoint

GRASP = 'cups_r11_grasp'
HERE = os.path.dirname(os.path.abspath(__file__))
HP = np.load(os.path.join(HERE, '../mug_cups_common/hand_pts_r2_mug_r1_grasp.npz'))
R_PURPLE, R_BLUE, H_PURPLE, R_PBASE = .0342, .0451, .0705, .0267
GHS = (.062, .045, .055)
DLS = (0., .012, .024, .030)
CLEARS = (.012, .025)
NEAR = .012     # hand points within R_BLUE + NEAR of the blue axis must stay above the rim
RIM_GAP = .004
MAX_AFTER_FALLBACK = 120
CONTACT_GAP = .078
MIN_INSERT = .015
SHORTS = (0., .005, .010, .015)
BACK = .08
OPEN_T = 2.0
CORR_FRACS = (1., .6, .3)
POST_ITERS = 2
MIN_CORR = .0015
PLAN_WALL = 150.     # planner wall-clock budget (s); after it the best fallback is used or the plan fails fast
MEAS_R, MEAS_MAX_RES, MEAS_MAX_CORR = .05, .003, .015
BAND = (.006, .022)  # purple points this far above the perceived blue rim form the lower band
STEP = .03
LIM_MARGIN = .08
CLEAR = .004
PHIS = (30, 15, 45, 0, 60)
THETAS = tuple(range(-180, 180, 15))
PSIS = (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90)


def frame(phi, theta, flip):
    a = np.array([math.cos(theta) * math.cos(phi), math.sin(theta) * math.cos(phi), -math.sin(phi)])
    l = np.array([-math.sin(theta), math.cos(theta), 0.]) * (-1. if flip else 1.)
    return np.column_stack([np.cross(l, a), l, a])


def Rz(p):
    c, s_ = math.cos(p), math.sin(p)
    return np.array([[c, -s_, 0.], [s_, c, 0.], [0., 0., 1.]])


def hand_world(side, opening, R, p):
    return p + HP['%s_%d' % (side, int(opening))] @ R.T


def free(side, R, p, axis, bottom, floor):
    W = hand_world(side, 1, R, p)
    if W[:, 2].min() < floor + CLEAR:
        return False
    h = W[:, 2] - bottom
    r = np.linalg.norm(W[:, :2] - axis, axis=1)
    rr = R_PBASE + (R_PURPLE - R_PBASE) * np.clip(h / H_PURPLE, 0, 1) + .0005
    return not bool(np.any((h > -.002) & (h < H_PURPLE + .003) & (r < rr)))


def chain(io, side, pts, start=None):
    if start is None:
        e = io._enc()
        T = io.adapter.tcp(e, side)
    else:
        e, T = dict(start[0]), start[1].copy()
    for p, R in pts:
        p = np.asarray(p, float)
        n = max(1, int(math.ceil(np.linalg.norm(p - T[:3, 3]) / STEP)))
        for k in range(1, n + 1):
            wp, wr = waypoint(T, p, R, k / n)
            try:
                q = io.adapter.plan(e, side, wp, wr)
            except Exception as ex:
                if io.error is not None:
                    raise
                return False, str(ex)[:60], None
            for nm, v in q.items():
                lo, hi = io.adapter.hand.rng[nm]
                if v < lo + LIM_MARGIN or v > hi - LIM_MARGIN:
                    return False, 'margin:' + nm, None
            e.update(q)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
    return True, '', (e, T)


def reconnect_sam():
    # fresh client connection to the same SAM3 service (the service and its protocol are unchanged)
    try:
        S._PROC.conn.close()
    except Exception:
        pass
    S._PROC = None
    return S.get_proc()


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


def axis(c, radius):
    rim = c['points'][c['points'][:, 2] > c['top'] - .012]
    xy, _ = P.fit_upright_cylinder(rim, radius)
    return np.asarray(xy, float), c['top']


def plan(io, pxy, ptop, bxy, btop):
    bottom = ptop - H_PURPLE
    shoulder_xy = np.array([-.38, C.midline(io)])
    u = shoulder_xy - bxy
    u = u / np.linalg.norm(u)
    n = 0
    fallback = None
    first = 'left' if pxy[1] > C.midline(io) else 'right'
    pick_cache = {}
    t_start = time.time()
    for clear in CLEARS:
        dz = btop + clear - bottom
        for sh in SHORTS:
            for side in (first, 'right' if first == 'left' else 'left'):
                for gh in GHS:
                    for dl in DLS:
                        for phi in PHIS:
                            for th in THETAS:
                                for flip in (0, 1):
                                    R = frame(math.radians(phi), math.radians(th), flip)
                                    a = R[:, 2]
                                    c = np.r_[pxy, bottom + gh] - dl * a
                                    pre = c - BACK * a + [0, 0, .02]
                                    lift = c + [0, 0, dz]
                                    key = (clear, side, gh, dl, phi, th, flip)
                                    if key not in pick_cache:
                                        if not all(free(side, R, c + s_ * (pre - c), pxy, bottom, bottom) for s_ in (0., .3, .6, 1.)):
                                            pick_cache[key] = None
                                        else:
                                            n += 1
                                            ok, _, st = chain(io, side, [(pre, R), (c, R), (lift, R)])
                                            pick_cache[key] = st if ok else None
                                    st = pick_cache[key]
                                    if time.time() - t_start > PLAN_WALL:
                                        if fallback is not None:
                                            io.log('plan', fallback=True, wall_cap=True, **fallback[1])
                                            return fallback[0]
                                        raise RuntimeError('plan_wall_cap n_tried=%d' % n)
                                    if st is None:
                                        continue
                                    if fallback is not None and sh == SHORTS[-1] and n > fallback[2] + MAX_AFTER_FALLBACK:
                                        io.log('plan', fallback=True, **fallback[1])
                                        return fallback[0]
                                    for psi in PSIS:
                                        R2 = Rz(math.radians(psi)) @ R
                                        a2 = R2[:, 2]
                                        ah = np.r_[a2[:2], 0.] / max(np.linalg.norm(a2[:2]), 1e-9)
                                        lh = np.r_[R2[:2, 1], 0.]
                                        r_g = R_PBASE + (R_PURPLE - R_PBASE) * gh / H_PURPLE
                                        held = dl * np.linalg.norm(a2[:2]) * ah - (CONTACT_GAP / 2 - r_g) * lh
                                        d = np.r_[bxy + sh * u - held[:2], bottom + gh + dz - dl * a2[2]]
                                        if hand_world(side, 1, R2, d)[:, 2].min() < btop + CLEAR:
                                            continue
                                        ok2, _, st2 = chain(io, side, [(d, R2)], start=st)
                                        if not ok2:
                                            continue
                                        W = hand_world(side, 0, R2, d)
                                        near = W[np.linalg.norm(W[:, :2] - bxy, axis=1) < R_BLUE + NEAR]
                                        room = (near[:, 2].min() - (btop + RIM_GAP)) if len(near) else 0.
                                        ins = d
                                        for dz_ in (min(room, .05), .75 * min(room, .05), .5 * min(room, .05), .25 * min(room, .05)):
                                            if dz_ <= .002:
                                                break
                                            q = d - [0, 0, dz_]
                                            if chain(io, side, [(q, R2)], start=st2)[0]:
                                                ins = q
                                                break
                                        got = float(d[2] - ins[2])
                                        cand = (side, R, R2, pre, c, lift, d, ins)
                                        info = dict(side=side, gh=gh, dl=dl, phi=phi, theta=th, flip=flip, psi=psi, short=sh, clear=clear, n_tried=n, c=c, d=d, insert=got, room=float(room))
                                        if got >= MIN_INSERT:
                                            io.log('plan', **info)
                                            return cand
                                        if fallback is None:
                                            fallback = (cand, info, n)
    if fallback is not None:
        io.log('plan', fallback=True, **fallback[1])
        return fallback[0]
    raise RuntimeError('no_reachable_pick_place n_tried=%d' % n)


def r_at(h):
    return R_PBASE + (R_PURPLE - R_PBASE) * float(np.clip(h / H_PURPLE, 0, 1))


def measure_over(io, bxy, btop):
    # head-camera measurement of the held purple cup over the blue cup: SAM3 'purple cup' mask, depth points above the perceived blue rim within MEAS_R of the blue axis;
    # lower band (just above the rim) fitted as an upright circle with the purple radius at that height; returns its centre or None
    obs = io.observe()
    p, ys, xs = G.safe_points(obs)
    proc = S.get_proc()
    state = proc.set_image(Image.fromarray(obs['rgb']))
    tried = []
    for pr in ('purple cup', 'cup'):
        mk, sc = S._prompt(proc, state, pr)
        for m, s_ in sorted(zip(mk, sc), key=lambda x: -x[1]):
            if s_ < .2:
                continue
            q = p[m[ys, xs] & (p[:, 2] > btop + BAND[0])]
            q = q[np.linalg.norm(q[:, :2] - bxy, axis=1) < MEAS_R]
            if len(q) < 60:
                tried.append((pr, round(float(s_), 3), int(len(q)), 'few'))
                continue
            zb = float(np.percentile(q[:, 2], 5))
            band = q[q[:, 2] < btop + BAND[1]]
            if len(band) < 25:
                tried.append((pr, round(float(s_), 3), int(len(band)), 'few_band'))
                continue
            rr = r_at(float(np.median(band[:, 2])) - zb + .005)
            xy, _ = P.fit_upright_cylinder(band, rr)
            xy = np.asarray(xy, float)
            res = float(np.median(np.abs(np.linalg.norm(band[:, :2] - xy, axis=1) - rr)))
            corr = bxy - xy
            tried.append((pr, round(float(s_), 3), int(len(band)), round(res, 4), [round(float(v), 4) for v in corr]))
            if res < MEAS_MAX_RES and np.linalg.norm(corr) < MEAS_MAX_CORR:
                return xy, tried
    return None, tried


def correct(io, side, R2, meas, bxy, d, ins, tag):
    # horizontal shift (blue axis - measured purple centre), full or partial, IK-checked; d None = only the current (lowered) pose ins moves
    if meas is None:
        return d, ins
    corr0 = np.r_[bxy - meas, 0.]
    if np.linalg.norm(corr0) < MIN_CORR:
        io.log('correct_small', tag=tag, corr=corr0)
        return d, ins
    for frac in CORR_FRACS:
        corr = frac * corr0
        pts = [(ins + corr, R2)] if d is None else [(d + corr, R2), (ins + corr, R2)]
        try:
            reach = chain(io, side, pts)[0]
        except Exception as e:
            if io.error is not None:
                raise
            io.log('correct_check_failed', tag=tag, err=str(e)[:80])
            return d, ins
        if not reach:
            io.log('correct_unreachable', tag=tag, frac=frac, corr=corr)
            continue
        try:
            io.move(side, pts[0][0], R2, .8)
        except Exception as e:
            if io.error is not None:
                raise
            io.log('correct_failed', tag=tag, err=str(e)[:80])
            return d, ins
        io.log('corrected', tag=tag, frac=frac, corr=corr)
        return (None if d is None else d + corr), ins + corr
    return d, ins


def safe_tcp(io, side):
    try:
        return io.tcp(side)[:3, 3]
    except Exception as e:
        if io.error is not None:
            raise
        return 'unavailable:' + str(e)[:60]


def controller(io):
    C.park(io)
    ev = io.demo['events']
    close = next(e for e in ev if e['kind'] == 'close')
    names = list(close['objects'])
    gname = min(names, key=lambda n: np.linalg.norm(close['objects'][n][:2] - close['position'][:2]))
    tname = next(n for n in names if n != gname)
    cups = []
    for k in range(6):
        obs = io.observe()
        cups = find_cups(obs)
        if len(cups) >= 2:
            break
        raw = {}
        for pr in ('cup', 'paper cup'):
            try:
                _, sc = S._prompt(S.get_proc(), {'rgb': obs['rgb']}, pr)
                raw[pr] = [round(float(x), 3) for x in sc][:4]
            except Exception as e:
                raw[pr] = repr(e)[:60]
        io.log('reobserve', attempt=k, n_cups=len(cups), raw_scores=raw)
        if all(isinstance(v, list) and not v for v in raw.values()):
            reconnect_sam()
            io.log('sam_reconnect', attempt=k)
        io.wait(.5 * (k + 1))
    if len(cups) < 2:
        raise RuntimeError('found %d cups' % len(cups))
    cups = cups[:3]
    o0 = io.demo['objects0']
    best = min(permutations(cups, 2), key=lambda pr: np.linalg.norm(pr[0]['c'] - o0[gname][:2]) + np.linalg.norm(pr[1]['c'] - o0[tname][:2]))
    pxy, ptop = axis(best[0], R_PURPLE)
    bxy, btop = axis(best[1], R_BLUE)
    io.log('perceived', grasp=pxy, grasp_top=ptop, target=bxy, target_top=btop, n_cups=len(cups), scores=[c['score'] for c in cups])
    side, R, R2, pre, c, lift, d, ins = plan(io, pxy, ptop, bxy, btop)
    io.grip(side, 1., .3)
    io.move(side, pre, R, 1.5)
    io.move(side, c, R, 1.0)
    io.grip(side, 0., .8)
    io.wait(.2)
    io.log('closed', tcp=io.tcp(side)[:3, 3])
    io.move(side, lift, R, 1.2)
    io.log('lifted', tcp=io.tcp(side)[:3, 3])
    io.move(side, d, R2, 2.0)
    io.wait(.3)
    try:
        meas, tried = measure_over(io, bxy, btop)
    except Exception as e:
        if io.error is not None:
            raise
        meas, tried = None, [repr(e)[:80]]
    io.log('measure_over', meas=meas, tried=tried)
    d, ins = correct(io, side, R2, meas, bxy, d, ins, 'pre')
    if ins[2] < d[2] - 1e-4:
        try:
            io.move(side, ins, R2, 1.0)
        except Exception as e:
            if io.error is not None:
                raise
            io.log('lower_interrupted', err=str(e)[:80], tcp=safe_tcp(io, side))
    io.wait(.3)
    for k in range(POST_ITERS):
        try:
            meas, tried = measure_over(io, bxy, btop)
        except Exception as e:
            if io.error is not None:
                raise
            meas, tried = None, [repr(e)[:80]]
        io.log('measure_low', k=k, meas=meas, tried=tried)
        if meas is None:
            break
        _, ins2 = correct(io, side, R2, meas, bxy, None, ins, 'post%d' % k)
        if ins2 is ins:
            break
        ins = ins2
        io.wait(.3)
    io.log('released_at', side=side, tcp=safe_tcp(io, side))
    io.grip(side, 1., OPEN_T)
    io.wait(.5)
    try:
        io.move(side, d, R2, .8)
        io.move(side, d - BACK * np.r_[R2[:2, 2], 0.] + [0, 0, .02], R2, 1.2)
    except Exception as e:
        if io.error is not None:
            raise
        io.log('retreat_failed', err=str(e)[:80])
    io.wait(2.0)
