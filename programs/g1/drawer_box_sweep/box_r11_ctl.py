# box_r11_ctl (r11 = r10 + reorient chain check: the carrying rotation leg (lift -> above set-down, carton turned about vertical) is a free-space transit, so only IK feasibility
# is required in its interior; the set-down endpoints need margin >= .02 rad (box_reorient_dbg_r2: interior margin 0.0 on that leg, endpoints .02-.05). recover() in safe_move
# handles transit overshoot. Everything else = r10.
import numpy as np
import box_r10_ctl as R10
import box_r5_ctl as B5
from box_r4_ctl import slerp_R

R10.REYAWS = (-60., -70., -75., -55., -80.)
R10.RESHIFTS = ((.04, 0.), (0., 0.), (.04, -.04), (0., -.04), (.02, -.02))
_cm = B5.chain_margin


def reo_chain(io, side, legs):
    # legs: pre, t, hi, above, p, above2 ; legs index 3 interior = transit
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
            if (i == 0 or i == 3) and f < 1.:
                continue
            for n, v in q.items():
                lo, hi = m.jnt_range[m.joint(n).id]
                worst = min(worst, v - lo, hi - v)
        p0, R0 = p, R
    # scale so the r10 threshold (.06) means .02 here
    return worst + .04 if worst >= .02 else None


class _Shim:
    pass


_orig_reorient = R10.reorient


def reorient(io, cxy, ctop, long3, bxy, btop):
    saved = B5.chain_margin
    B5.chain_margin = lambda io_, side, legs: reo_chain(io_, side, legs) if len(legs) == 6 else saved(io_, side, legs)
    try:
        return _orig_reorient(io, cxy, ctop, long3, bxy, btop)
    finally:
        B5.chain_margin = saved


R10.reorient = reorient


def run(io):
    io.log('variant', name='box_r11')
    R10.run(io)
