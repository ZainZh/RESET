# box_r9_ctl (r9 = r8 + transit-leg rule): every grasp/relocation chain check ignores the joint margin on the FIRST leg (current pose -> pre-grasp, a free-space
# transit where the IK sits on a joint limit: box_reloc_dbg_r1 / box_via_dbg_r1 show margin 0.0 there and >= .09 on all task legs), requiring only IK feasibility
# for it; the margin >= .06 rule stays on every other leg. Transit overshoot is handled by recover() in safe_move (drawer r8). Everything else = r8.
# Runtime side: io only (rendered RGB-D, calibration, encoders, robot-only model joint ranges, demo priors, SAM3).
import numpy as np
import box_r4_ctl as B4
import box_r5_ctl as B5
import box_r8_ctl as B8
from box_r4_ctl import slerp_R

_orig = B4.chain_margin


def chain_margin(io, side, legs):
    m = io.adapter.hand.m
    enc = io._enc()
    T = io.adapter.tcp(enc, side)
    p0, R0 = T[:3, 3].copy(), T[:3, :3].copy()
    worst = 9.
    for i, (p, R) in enumerate(legs):
        p = np.asarray(p, float)
        for f in (.25, .5, .75, 1.):
            try:
                q = io.adapter.plan(enc, side, p0 + (p - p0) * f, slerp_R(R0, R, f))
            except Exception:
                return None
            enc.update(q)
            if i > 0 or f == 1.:  # transit interior (first leg, f < 1) is feasibility only
                for n, v in q.items():
                    lo, hi = m.jnt_range[m.joint(n).id]
                    worst = min(worst, v - lo, hi - v)
        p0, R0 = p, R
    return worst


B4.chain_margin = chain_margin
B5.chain_margin = chain_margin
B8.chain_margin = chain_margin


def run(io):
    io.log('variant', name='box_r9', transit_margin_rule='first leg interior feasibility only')
    B8.run(io)
