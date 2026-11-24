# box_r8_ctl (r8 = r7 + two-stage relocation plan): r7's fixed relocation spot was unreachable in 10/11 seeds that needed it. r8 searches the relocation spot
# (RELOC_XYS, code constants: carton positions where the preferred right-arm plan worked in r4) and the new long-axis yaw (RELOC_YAWS), and accepts a
# relocation only if (a) the relocating arm's straight-down grasp -> lift 5 cm -> lay-down chain is IK-feasible with margin >= .06 rad AND (b) the preferred
# right-arm plan is feasible from the resulting carton pose (same model-only check as the real plan). The lying carton keeps its height, so ctop is unchanged.
# Runtime side: io only (rendered RGB-D, calibration, encoders, robot-only model joint ranges, demo priors, SAM3).
import math
import numpy as np
from drawer_r8_ctl import safe_move
from box_r5_ctl import UP, HALF_H, GRIP_BELOW_TOP, perceive, grasp_rots, chain_margin, find_plan, execute, frames_of

RELOC_XYS = (np.array([-.07, -.45]), np.array([-.05, -.46]), np.array([-.08, -.43]), np.array([-.06, -.48]))
RELOC_YAWS = (-70., -62., -78.)
CARRY = .05


def relocate(io, cxy, ctop, long3, bxy, btop):
    cc = np.r_[cxy, ctop - HALF_H]
    t = np.r_[cxy, ctop - GRIP_BELOW_TOP]
    C0 = frames_of(long3)
    hi = t + CARRY * UP
    n = 0
    for side in ('left', 'right'):
        for gn, Rg in grasp_rots(long3):
            if not gn.startswith('tl+0_ts+0'):
                continue
            pre = t - Rg[:, 2] * .08
            w1 = chain_margin(io, side, [(pre, Rg), (t, Rg), (hi, Rg)])
            if w1 is None or w1 < .06:
                continue
            H = C0.T @ Rg
            o_h = Rg.T @ (cc - t)
            for xy in RELOC_XYS:
                for yaw in RELOC_YAWS:
                    L = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.])
                    for sg in (1., -1.):
                        Rp = frames_of(sg * L) @ H
                        p = np.r_[xy, cc[2] + .006] - Rp @ o_h
                        above = p + CARRY * UP
                        n += 1
                        w = chain_margin(io, side, [(pre, Rg), (t, Rg), (hi, Rg), (above, Rp), (p, Rp), (above, Rp)])
                        if w is None or w < .06:
                            continue
                        # (b) would the preferred right-arm plan be feasible from there? (encoders still at the current pose: the right arm is parked or at the
                        # same park pose after the relocation, so the check starts from a comparable seed)
                        if find_plan(io, xy, ctop, L, bxy, btop, True) is None:
                            continue
                        io.log('relocate', side=side, grasp=gn, xy=xy, yaw=yaw, sign=sg, margin=w, n_checked=n)
                        io.grip(side, 1., .3)
                        safe_move(io, side, pre, Rg, 1.5)
                        safe_move(io, side, t, Rg, 1.0)
                        io.grip(side, 0., .7)
                        io.wait(.2)
                        safe_move(io, side, hi, Rg, 1.0)
                        safe_move(io, side, above, Rp, 2.0)
                        safe_move(io, side, p, Rp, 1.0)
                        io.wait(.2)
                        io.grip(side, 1., .5)
                        io.wait(.2)
                        safe_move(io, side, above, Rp, 1.0)
                        io.home(side, 1.5)
                        io.wait(.3)
                        io.log('relocated', side=side)
                        return True
    io.log('relocate_infeasible', n_checked=n)
    return False


def run(io):
    cxy, ctop, long3, bxy, btop = perceive(io)
    plan = find_plan(io, cxy, ctop, long3, bxy, btop, True)
    if plan is None and relocate(io, cxy, ctop, long3, bxy, btop):
        cxy, ctop, long3, bxy, btop = perceive(io)
        plan = find_plan(io, cxy, ctop, long3, bxy, btop, True)
    if plan is None:
        raise RuntimeError('no feasible grasp/place plan')
    execute(io, plan, bxy, btop)
