# box_r10_ctl (r10 = r9 + in-place re-orientation hand-over): r9 failures 0,2,5,6,15,16,17 have the carton long axis at -90..-113 deg yaw, the proven right-arm plan
# (straight-down grasp + stand frame Y+120_L+1) succeeded at -58..-82 deg. When that plan is infeasible, one arm (left first, then right) lifts the carton 5 cm, turns it
# about vertical to a target yaw (REYAWS) near its current spot (RESHIFTS, toward the right arm), sets it down flat and releases; the scene is re-perceived and the
# proven right-arm plan runs. A re-orientation is executed only if its own chain AND the following right-arm plan pass the model-only IK/margin check.
# Runtime side: io only (rendered RGB-D, calibration, encoders, robot-only model joint ranges, demo priors, SAM3).
import math
import numpy as np
import box_r9_ctl as R9  # installs the r9 transit-leg rule into r4/r5/r8
from drawer_r8_ctl import safe_move
from box_r5_ctl import UP, HALF_H, GRIP_BELOW_TOP, perceive, grasp_rots, find_plan, execute, frames_of
import box_r5_ctl as B5

REYAWS = (-70., -62., -78., -55., -85.)
RESHIFTS = ((0., 0.), (0., -.03), (.03, -.03), (0., -.06), (-.03, -.03))
CARRY = .05


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
                        w = B5.chain_margin(io, side, [(pre, Rg), (t, Rg), (hi, Rg), (above, Rp), (p, Rp), (above + .03 * UP, Rp)])
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
                        io.grip(side, 1., .5)
                        io.wait(.2)
                        safe_move(io, side, above + .03 * UP, Rp, 1.0)
                        io.home(side, 1.5)
                        io.wait(.3)
                        io.log('reoriented', side=side)
                        return True
    io.log('reorient_infeasible', n_checked=n)
    return False


def run(io):
    io.log('variant', name='box_r10')
    cxy, ctop, long3, bxy, btop = perceive(io)
    plan = find_plan(io, cxy, ctop, long3, bxy, btop, True)
    for k in range(2):
        if plan is not None:
            break
        if not reorient(io, cxy, ctop, long3, bxy, btop):
            break
        cxy, ctop, long3, bxy, btop = perceive(io)
        plan = find_plan(io, cxy, ctop, long3, bxy, btop, True)
    if plan is None:
        raise RuntimeError('no feasible grasp/place plan')
    execute(io, plan, bxy, btop)
