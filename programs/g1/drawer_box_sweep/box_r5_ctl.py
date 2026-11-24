# box_r5_ctl (r5 = r4 + left-arm relocation): b3 DEV0-19 of r4 was 8/20. The plan tl+0_ts+0 + Y+120_L+1 passed 8/9; every other executed plan failed (ts+20 grasps slip in transit, other stand frames knock the box on retreat) and 5 seeds had no right-arm plan.
# r5 uses ONLY that preferred plan directly. When it is infeasible from the perceived carton pose, the left arm lifts the carton and lays it flat at a fixed spot (RELOC_XY, RELOC_LONG, code constants) where the right arm has it, the scene is re-perceived, and the right arm continues. The r4 search remains the last fallback (logged).
# Runtime side: io only (rendered RGB-D, calibration, encoders, robot-only model joint ranges, demo priors, SAM3). Perception, servo, thumb clear and retreat are r4 unchanged.
import math
import numpy as np
from drawer_r8_ctl import margin_plan, safe_move
import box_r4_ctl as B
from box_r4_ctl import UP, HALF_H, HALF_L, GRIP_BELOW_TOP, CLEAR, DX, DY, SIDE, SERVO_MAX, THUMB_CLEAR, perceive, grasp_rots, stand_frames, chain_margin, slerp_R, held_top_xy

RELOC_XY = np.array([-.06, -.45])
RELOC_LONG = np.array([.32, -.95, 0.]) / np.linalg.norm([.32, -.95])
PREF_GRASP, PREF_STAND = 'tl+0_ts+0', 'Y+120_L+1'
LEFT = 'left'


def frames_of(long3):
    short = np.cross(UP, long3)
    short = short / np.linalg.norm(short)
    return np.column_stack([short, long3, np.cross(short, long3)])


def find_plan(io, cxy, ctop, long3, bxy, btop, strict):
    cc = np.r_[cxy, ctop - HALF_H]
    t = np.r_[cxy, ctop - GRIP_BELOW_TOP]
    C0 = frames_of(long3)
    hi = t + .15 * UP
    n = 0
    for gn, Rg in grasp_rots(long3):
        if strict and not gn.startswith(PREF_GRASP):
            continue
        pre = t - Rg[:, 2] * .08
        w1 = chain_margin(io, SIDE, [(pre, Rg), (t, Rg), (hi, Rg)])
        if w1 is None or w1 < .06:
            continue
        H = C0.T @ Rg
        o_h = Rg.T @ (cc - t)
        for dx in DX:
            for dy in DY:
                target = np.r_[bxy[0] + dx, bxy[1] + dy, btop + CLEAR + HALF_L]
                for pn, Cs in stand_frames():
                    if strict and pn != PREF_STAND:
                        continue
                    Rp = Cs @ H
                    p = target - Rp @ o_h
                    if margin_plan(io, SIDE, [p], Rp) is None:
                        continue
                    above = p + .06 * UP
                    mid = np.r_[0.5 * (hi[:2] + p[:2]), max(above[2], hi[2]) + .02]
                    n += 1
                    w = chain_margin(io, SIDE, [(pre, Rg), (t, Rg), (hi, Rg), (mid, slerp_R(Rg, Rp, .5)), (above, Rp), (p, Rp)])
                    if w is not None and w >= .06:
                        return dict(gn=gn, Rg=Rg, pn=pn, Rp=Rp, p=p, mid=mid, hi=hi, pre=pre, above=above, margin=w, dx=dx, dy=dy, t=t, strict=strict)
    io.log('plan_none', strict=strict, chains_checked=n)
    return None


def relocate(io, cxy, ctop, long3):
    # left arm: pick the lying carton and lay it flat at RELOC_XY with its long axis along RELOC_LONG (either sign); returns True when the motion was executed
    cc = np.r_[cxy, ctop - HALF_H]
    t = np.r_[cxy, ctop - GRIP_BELOW_TOP]
    C0 = frames_of(long3)
    hi = t + .12 * UP
    for ts_ok in (True, False):  # ts=0 grasps first: ts+-20 lost the thumb-proximal contact in r4
        for gn, Rg in grasp_rots(long3):
            if ts_ok != ('_ts+0_' in gn):
                continue
            pre = t - Rg[:, 2] * .08
            w1 = chain_margin(io, LEFT, [(pre, Rg), (t, Rg), (hi, Rg)])
            if w1 is None or w1 < .06:
                continue
            H = C0.T @ Rg
            o_h = Rg.T @ (cc - t)
            for sg in (1., -1.):
                Rp = frames_of(sg * RELOC_LONG) @ H
                p = np.r_[RELOC_XY, cc[2] + .006] - Rp @ o_h
                above = p + .10 * UP
                w = chain_margin(io, LEFT, [(pre, Rg), (t, Rg), (hi, Rg), (above, Rp), (p, Rp), (above, Rp)])
                if w is None or w < .06:
                    continue
                io.log('relocate', grasp=gn, sign=sg, margin=w, p=p)
                io.grip(LEFT, 1., .3)
                safe_move(io, LEFT, pre, Rg, 1.5)
                safe_move(io, LEFT, t, Rg, 1.0)
                io.grip(LEFT, 0., .7)
                io.wait(.2)
                safe_move(io, LEFT, hi, Rg, 1.0)
                safe_move(io, LEFT, above, Rp, 2.0)
                safe_move(io, LEFT, p, Rp, 1.0)
                io.wait(.2)
                io.grip(LEFT, 1., .5)
                io.wait(.2)
                safe_move(io, LEFT, above, Rp, 1.0)
                io.home(LEFT, 1.5)
                io.wait(.3)
                io.log('relocated')
                return True
    io.log('relocate_infeasible')
    return False


def execute(io, plan, bxy, btop):
    Rg, Rp, p, mid, hi, pre, above, t = plan['Rg'], plan['Rp'], plan['p'], plan['mid'], plan['hi'], plan['pre'], plan['above'], plan['t']
    io.log('plan', grasp=plan['gn'], place=plan['pn'], p=p, margin=plan['margin'], dx=plan['dx'], dy=plan['dy'], strict=plan['strict'])
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
    a, y = Rp[:, 2], Rp[:, 1]
    q0 = p
    try:
        safe_move(io, SIDE, p - THUMB_CLEAR * y, Rp, .8)
        q0 = p - THUMB_CLEAR * y
    except Exception as e:
        if io.error is not None:
            raise
        io.log('thumb_clear_failed', err=str(e)[:60])
    for off in (-.07 * a + .02 * UP, -.05 * a, .05 * UP):
        try:
            safe_move(io, SIDE, q0 + off, Rp, 1.0)
            break
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


def run(io):
    cxy, ctop, long3, bxy, btop = perceive(io)
    plan = find_plan(io, cxy, ctop, long3, bxy, btop, True)
    if plan is None:
        if relocate(io, cxy, ctop, long3):
            cxy, ctop, long3, bxy, btop = perceive(io)
            plan = find_plan(io, cxy, ctop, long3, bxy, btop, True)
        if plan is None:
            plan = find_plan(io, cxy, ctop, long3, bxy, btop, False)
    if plan is None:
        raise RuntimeError('no feasible grasp/place plan')
    execute(io, plan, bxy, btop)
