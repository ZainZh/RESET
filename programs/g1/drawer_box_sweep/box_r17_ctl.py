# box_r17_ctl (r17 = r16 + retreat point list (first reachable): up 20 cm, up 20 cm + 8 cm toward the arm's own side (+y left / -y right), up 16 cm + 12 cm toward its side; the r16 single point was IK-infeasible in 4/4; r16 = r15 + before the joint-space home move the hand goes in Cartesian space to a retreat point 12 cm toward the robot (-x) and 20 cm above the table, hand still in the set-down orientation (r15: carton untouched after the 12 cm lift, then swept by home()); r15 = r14 + after the thumb clear the hand lifts 12 cm (logged) before the joint-space home move; r14 seeds 2,5,16,17: carton free after the thumb clear but swept toward the home side afterwards; r14 = r13 + after setting the carton down the opened hand first moves THUMB_CLEAR along -pad axis (the fixed thumb otherwise keeps the carton: r13 seeds 2,5,16,4,6,13
# dragged it to the home side and stood it up, 'released' failed), then lifts 6 cm; same as execute()'s release. Everything else = r13.
import math
import numpy as np
import box_r13_ctl as R13
import box_r11_ctl as R11
import box_r10_ctl as R10
import box_r5_ctl as B5
from drawer_r8_ctl import safe_move
from box_r5_ctl import UP, HALF_H, perceive, grasp_rots, execute, frames_of
from box_r4_ctl import THUMB_CLEAR
GRIP_BELOW_TOP = HALF_H
REYAWS, RESHIFTS, CARRY = R10.REYAWS, R10.RESHIFTS, R10.CARRY
find_plan = R10.find_plan


def reorient(io, cxy, ctop, long3, bxy, btop):
    cc = np.r_[cxy, ctop - HALF_H]
    t = np.r_[cxy, ctop - GRIP_BELOW_TOP]
    C0 = frames_of(long3)
    hi = t + CARRY * UP
    n = 0
    for side in ('left', 'right'):
        for gn, Rg in grasp_rots(long3):
            pre = t - Rg[:, 2] * .08
            w1 = B5.chain_margin(io, side, [(pre, Rg), (t, Rg), (hi, Rg)])
            if w1 is None or w1 < .06:
                continue
            H = C0.T @ Rg
            o_h = Rg.T @ (cc - t)
            for sh in RESHIFTS:
                xy = cxy + np.array(sh)
                for yaw in REYAWS:
                    L = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.])
                    for sg in (1., -1.):
                        Rp = frames_of(sg * L) @ H
                        p = np.r_[xy, cc[2] + .004] - Rp @ o_h
                        above = p + CARRY * UP
                        n += 1
                        w = R11.reo_chain(io, side, [(pre, Rg), (t, Rg), (hi, Rg), (above, Rp), (p, Rp), (p - THUMB_CLEAR * Rp[:, 1] + .06 * UP, Rp)])
                        if w is None or w < .06:
                            continue
                        if find_plan(io, xy, ctop, L, bxy, btop, True) is None:
                            continue
                        io.log('reorient', side=side, grasp=gn, xy=xy, yaw=yaw, sign=sg, margin=w, n_checked=n)
                        io.grip(side, 1., .3)
                        safe_move(io, side, pre, Rg, 1.5)
                        safe_move(io, side, t, Rg, 1.0)
                        io.grip(side, 0., .7)
                        io.wait(.2)
                        safe_move(io, side, hi, Rg, 1.0)
                        safe_move(io, side, above, Rp, 1.8)
                        safe_move(io, side, p, Rp, 1.0)
                        io.wait(.2)
                        io.log('set_down', side=side)
                        io.grip(side, 1., .5)
                        io.wait(.2)
                        q0 = p
                        try:
                            safe_move(io, side, p - THUMB_CLEAR * Rp[:, 1], Rp, .8)
                            q0 = p - THUMB_CLEAR * Rp[:, 1]
                        except Exception as e:
                            if io.error is not None:
                                raise
                            io.log('thumb_clear_failed', err=str(e)[:60])
                        io.log('thumb_cleared', side=side)
                        safe_move(io, side, q0 + .06 * UP, Rp, 1.0)
                        io.log("lift6", side=side)
                        try:
                            safe_move(io, side, q0 + .12 * UP, Rp, .8)
                        except Exception as e:
                            if io.error is not None:
                                raise
                        io.log("lift12", side=side)
                        try:
                            sy = 1. if side == 'left' else -1.
                            for rp_ in (q0 + [0, 0, .20], q0 + [0, .08 * sy, .20], q0 + [-.04, .12 * sy, .16]):
                                try:
                                    safe_move(io, side, rp_, Rp, 1.0)
                                except Exception as e2:
                                    if io.error is not None:
                                        raise
                                    io.log('retreat_pt_failed', err=str(e2)[:50])
                        except Exception as e:
                            if io.error is not None:
                                raise
                            io.log("retreat_cart_failed", err=str(e)[:60])
                        io.log("retreated", side=side)
                        io.home(side, 1.5)
                        io.wait(.3)
                        io.log('reoriented', side=side)
                        return True
    io.log('reorient_infeasible', n_checked=n)
    return False


R10.reorient = reorient


def run(io):
    io.log('variant', name='box_r17')
    cxy, ctop, long3, bxy, btop = perceive(io)
    plan = R10.find_plan(io, cxy, ctop, long3, bxy, btop, True)
    if plan is None and reorient(io, cxy, ctop, long3, bxy, btop):
        cxy, ctop, long3, bxy, btop = perceive(io)
        plan = R10.find_plan(io, cxy, ctop, long3, bxy, btop, True)
    if plan is None:
        raise RuntimeError('no feasible grasp/place plan')
    execute(io, plan, bxy, btop)
