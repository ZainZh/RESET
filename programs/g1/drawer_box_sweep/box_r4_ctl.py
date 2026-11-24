# box_r4_ctl (r4 = r3 + (A) XY_SERVO: at the pose above the box the held carton top face is re-perceived (SAM3 mask + depth) and the XY error to the target is corrected, because closing the fingers shifts the carton about 17 mm against the fixed thumb (b3 s7, s10); (B) THUMB_CLEAR: after opening, the hand first moves along -pad axis so the fixed thumb leaves the carton, then withdraws (b3 s10: retreat dragged the carton off the box); r3 = r2 + grasp family tilted along/about the carton long axis (box_grasp_scan_r1: right arm has a grasp in 16/20 seeds only with such tilts), pre-grasp along the approach, standing pose computed from the actual hand-to-carton transform like xTrainer b8 (carton frame C, H = C.T Rg, Rp = C_stand H), TCP target from the carton-centre offset in the hand frame; r2 = r1 + TCP grips near the carton top so the standing carton centre sits CARTON_FWD ahead of the TCP along the approach (box top is at the edge of right-arm reach on b3: box_plan_diag_r1), place candidates given as carton-centre offsets, full-chain IK check including rotation waypoints): G1 lift_box controller (runtime side: uses only io = rendered RGB-D, calibration, encoders, robot-only model joint ranges, demo priors, SAM3).
# Shared skill (from xTrainer b8, semantics unchanged): SAM3 'small box' -> two instances (higher top = wooden box, other = carton); carton long axis from the
# top-face points; grasp the lying carton across its short axis, lift, rotate so the long axis is vertical, carry above the box, lower, release, retreat.
# G1-specific plan (this file): right arm only places (the box top is outside left-arm reach on b3, model-only scan box_place_scan_r1); place pose = approach
# horizontal, pad axis horizontal (hand x vertical = carton long axis vertical), approach yaw and an offset toward the robot from a fixed list, chosen by the
# robot-only IK joint-margin over the whole chain; the hand is closed at the place point and opened only after a settle.
# Contact priors (listed in NOTES.md): grasp across the 65 mm carton faces with the hand v0.4 sphere-68 mm pinch frame + squeeze (trusted-side profile);
# TCP at the carton centre height (carton top - 32.5 mm, half of 65 mm code constant).
import math
import numpy as np
from PIL import Image
from rsi import perception_sam3 as S, perception_geo as G
from drawer_r8_ctl import Rax, margin_plan, safe_move

UP = np.array([0., 0., 1.])
HALF_H = .0325       # carton lying: half of the 65 mm side (task object size, code constant)
HALF_L = .05         # carton standing: half of the 100 mm long side
CLEAR = .010         # place: carton bottom this far above the perceived box top
GRIP_BELOW_TOP = .012  # TCP this far below the lying carton's top face
CARTON_FWD = HALF_H - GRIP_BELOW_TOP  # carton centre ahead of the TCP along the approach once standing
DX = (-.03, -.045, -.06)   # place: carton centre offset toward the robot from the perceived box-top centre (reach on b3)
DY = (0., .02, -.02)
YAWS = (15., 0., -15., -30., -45., -60.)
SIDE = 'right'
XY_SERVO = True
SERVO_MAX = .05     # largest XY correction accepted from the held-carton view
THUMB_CLEAR = .014  # hand shift along -pad axis after opening (0 = off)


def perceive(io):
    obs = io.observe()
    plane = G.fit_plane(obs)
    proc = S.get_proc()
    state = proc.set_image(Image.fromarray(obs['rgb']))
    pp, ys, xs = G.safe_points(obs)
    hh = G.height(pp, plane)
    insts = []
    for prompt in ('small box', 'box', 'carton'):
        mk, sc = S._prompt(proc, state, prompt)
        for m, sv in zip(mk, sc):
            sel = m[ys, xs] & (hh > .005) & (hh < .20)
            if sv >= .4 and sel.sum() >= 100:
                q = pp[sel]
                c = np.median(q[:, :2], 0)
                if all(np.linalg.norm(c - np.median(i['points'][:, :2], 0)) > .05 for i in insts):
                    insts.append({'points': q, 'top': float(np.percentile(q[:, 2], 95)), 'score': float(sv), 'prompt': prompt})
        if len(insts) >= 2:
            break
    if len(insts) < 2:
        raise RuntimeError('found %d box-like instances' % len(insts))
    insts = sorted(insts, key=lambda d: -d['top'])[:2]
    box, carton = insts[0], insts[1]
    top = box['points'][box['points'][:, 2] > box['top'] - .012]
    bxy = top[:, :2].mean(0)
    cp = carton['points']
    ctop = cp[cp[:, 2] > carton['top'] - .01]
    if len(ctop) > 20:
        cxy = (ctop[:, :2].min(0) + ctop[:, :2].max(0)) / 2
        w_, v_ = np.linalg.eigh(np.cov((ctop[:, :2] - ctop[:, :2].mean(0)).T))
        long2 = v_[:, 1]
    else:
        cxy = cp[:, :2].mean(0); long2 = np.array([1., 0.])
    io.log('perceived', carton=cxy, carton_top=carton['top'], long_axis=long2, box=bxy, box_top=box['top'], scores=[box['score'], carton['score']])
    return cxy, carton['top'], np.r_[long2, 0.], bxy, box['top']


def slerp_R(R0, R1, f):
    # geodesic interpolation R0 -> R1 (rotation-vector form), f in [0, 1]
    D = R1 @ R0.T
    c = max(-1., min(1., (np.trace(D) - 1.) / 2.))
    ang = math.acos(c)
    if ang < 1e-9:
        return R1.copy()
    if math.pi - ang < 1e-6:
        w, v = np.linalg.eigh((D + D.T) / 2.)
        k = v[:, int(np.argmax(w))]
    else:
        k = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]]) / (2. * math.sin(ang))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    th = ang * f
    return (np.eye(3) + math.sin(th) * K + (1. - math.cos(th)) * K @ K) @ R0


def grasp_rots(long3):
    # pad axis across the carton (short = horizontal, perpendicular to the long axis); approach tilted tl along the long axis and ts about it; smallest total tilt first
    short = np.cross(UP, long3)
    short /= np.linalg.norm(short)
    out = []
    for tl in (0., 20., -20., 40., -40., 60., -60.):
        for ts in (0., 20., -20.):
            c, d = math.radians(tl), math.radians(ts)
            a = -UP * math.cos(c) * math.cos(d) + long3 * math.sin(c) + short * math.sin(d) * math.cos(c)
            for sg in (1., -1.):
                out.append((abs(tl) + abs(ts), 'tl%+d_ts%+d_s%+d' % (tl, ts, sg), Rax(a, sg * short)))
    out.sort(key=lambda x: x[0])
    return [(n, R) for _, n, R in out]


def stand_frames():
    # standing carton frames [short, long, third]: long axis = +-UP (scorer uses |tilt|), short axis horizontal at yaw
    out = []
    for yaw in (0., 30., -30., 60., -60., 90., -90., 120., -120., 150., -150., 180.):
        u = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.])
        for sg in (1., -1.):
            lg = sg * UP
            out.append(('Y%+d_L%+d' % (yaw, sg), np.column_stack([u, lg, np.cross(u, lg)])))
    return out


def held_top_xy(io, expect_xy, min_z):
    # top-face centre of the held standing carton from a fresh RGB-D frame: SAM3 instance near the expected XY, points above min_z, top 12 mm band
    obs = io.observe()
    proc = S.get_proc()
    state = proc.set_image(Image.fromarray(obs['rgb']))
    pp, ys, xs = G.safe_points(obs)
    best = None
    for prompt in ('small box', 'carton', 'box'):
        mk, sc = S._prompt(proc, state, prompt)
        for m, sv in zip(mk, sc):
            q = pp[m[ys, xs] & (pp[:, 2] > min_z)]
            if sv < .3 or len(q) < 60:
                continue
            band = q[q[:, 2] > np.percentile(q[:, 2], 98) - .012]
            if len(band) < 30:
                continue
            c = (band[:, :2].min(0) + band[:, :2].max(0)) / 2
            d = float(np.linalg.norm(c - expect_xy))
            if d < .08 and (best is None or sv > best[1]):
                best = (c, float(sv), int(len(band)), prompt, d)
        if best is not None:
            break
    return best


def chain_margin(io, side, legs):
    # IK through [(p, R), ...] from the current encoders with interpolated waypoints between legs; returns the worst arm-joint margin (robot-only model) or None
    m = io.adapter.hand.m
    enc = io._enc()
    T = io.adapter.tcp(enc, side)
    p0, R0 = T[:3, 3].copy(), T[:3, :3].copy()
    worst = 9.
    for p, R in legs:
        p = np.asarray(p, float)
        for f in (.25, .5, .75, 1.):
            try:
                q = io.adapter.plan(enc, side, p0 + (p - p0) * f, slerp_R(R0, R, f))
            except Exception:
                return None
            enc.update(q)
            for n, v in q.items():
                lo, hi = m.jnt_range[m.joint(n).id]
                worst = min(worst, v - lo, hi - v)
        p0, R0 = p, R
    return worst


def run(io):
    cxy, ctop, long3, bxy, btop = perceive(io)
    cc = np.r_[cxy, ctop - HALF_H]
    t = np.r_[cxy, ctop - GRIP_BELOW_TOP]
    short = np.cross(UP, long3); short /= np.linalg.norm(short)
    C0 = np.column_stack([short, long3, np.cross(short, long3)])
    plan = None
    nchk = ngr = 0
    for gn, Rg in grasp_rots(long3):
        a0 = Rg[:, 2]
        pre = t - a0 * .08
        hi = t + .15 * UP
        w1 = chain_margin(io, SIDE, [(pre, Rg), (t, Rg), (hi, Rg)])
        if w1 is None or w1 < .06:
            continue
        ngr += 1
        H = C0.T @ Rg
        o_h = Rg.T @ (cc - t)  # carton centre in the hand frame at the grasp
        for dx in DX:
            for dy in DY:
                target = np.r_[bxy[0] + dx, bxy[1] + dy, btop + CLEAR + HALF_L]
                for pn, Cs in stand_frames():
                    Rp = Cs @ H
                    p = target - Rp @ o_h
                    if margin_plan(io, SIDE, [p], Rp) is None:
                        continue
                    above = p + .06 * UP
                    mid = np.r_[0.5 * (hi[:2] + p[:2]), max(above[2], hi[2]) + .02]
                    legs = [(pre, Rg), (t, Rg), (hi, Rg), (mid, slerp_R(Rg, Rp, .5)), (above, Rp), (p, Rp)]
                    nchk += 1
                    w = chain_margin(io, SIDE, legs)
                    if w is not None and w >= .06:
                        plan = dict(gn=gn, Rg=Rg, pn=pn, Rp=Rp, p=p, mid=mid, hi=hi, pre=pre, above=above, margin=w, dx=dx, dy=dy, g_margin=w1)
                        break
                if plan:
                    break
            if plan:
                break
        if plan:
            break
    io.log('plan_counts', grasps_ok=ngr, chains_checked=nchk)
    io.log('plan_search', n_checked=nchk, found=plan is not None)
    if plan is None:
        raise RuntimeError('no feasible grasp/place plan')
    io.log('plan', grasp=plan['gn'], place=plan['pn'], p=plan['p'], margin=plan['margin'], dx=plan['dx'], dy=plan['dy'])
    Rg, Rp, p, mid, hi, pre, above = plan['Rg'], plan['Rp'], plan['p'], plan['mid'], plan['hi'], plan['pre'], plan['above']
    io.grip(SIDE, 1., .3)
    safe_move(io, SIDE, pre, Rg, 1.5)
    safe_move(io, SIDE, t, Rg, 1.0)
    io.grip(SIDE, 0., .7)
    io.wait(.2)
    io.log('closed')
    safe_move(io, SIDE, hi, Rg, 1.2)
    io.log('lifted')
    safe_move(io, SIDE, mid, slerp_R(Rg, Rp, .5), 2.5)
    safe_move(io, SIDE, above, Rp, 2.5)
    io.log('above_place')
    if XY_SERVO:
        io.wait(.3)
        target_xy = np.r_[bxy[0] + plan['dx'], bxy[1] + plan['dy']]
        v = held_top_xy(io, target_xy, btop + .06)
        if v is None:
            io.log('servo_no_view')
        else:
            err = target_xy - v[0]
            io.log('servo', top_xy=v[0], err=err, score=v[1], n=v[2], prompt=v[3])
            if np.linalg.norm(err) <= SERVO_MAX:
                p2 = p + np.r_[err, 0.]
                if margin_plan(io, SIDE, [p2 + .06 * UP, p2], Rp) is not None:
                    p = p2
                    safe_move(io, SIDE, p + .06 * UP, Rp, 1.2)
                else:
                    io.log('servo_unreachable')
    safe_move(io, SIDE, p, Rp, 1.5)
    io.wait(.4)
    io.log('at_place', residual=float(np.linalg.norm(io.tcp(SIDE)[:3, 3] - p)))
    io.grip(SIDE, 1., .6)
    io.wait(.4)
    io.log('released')
    a = Rp[:, 2]
    y = Rp[:, 1]
    q0 = p
    if THUMB_CLEAR > 0:
        try:
            safe_move(io, SIDE, p - THUMB_CLEAR * y, Rp, .8)
            q0 = p - THUMB_CLEAR * y
        except Exception as e:
            if io.error is not None:
                raise
            io.log('thumb_clear_failed', err=str(e)[:60])
    for off in (-.07 * a + .02 * UP, -.05 * a, .05 * UP):
        try:
            safe_move(io, SIDE, q0 + off, Rp, 1.0); break
        except Exception as e:
            if io.error is not None:
                raise
            io.log('retreat_retry', err=str(e)[:60])
    try:
        io.home(SIDE, 2.0)
    except Exception as e:
        if io.error is not None:
            raise
        io.log('home_failed', err=str(e)[:60])
    io.wait(1.0)
    io.observe()
