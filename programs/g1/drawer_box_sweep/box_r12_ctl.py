# box_r12_ctl (r12 = r11 + long-axis sign symmetry): the perceived carton long axis is a PCA eigenvector whose sign is arbitrary, but find_plan's grasp/stand frames
# depend on it. The carton is symmetric under a 180 deg turn about vertical, so r12 tries the proven plan with long3 AND -long3 (box_plan_region_r1: seed 0 has 0 feasible
# targets with one sign and 70 with the other). The re-orientation fallback (r10/r11) also checks both signs. Everything else = r11.
import numpy as np
import box_r11_ctl as R11  # r9 transit rule + r11 reorient chain rule
import box_r10_ctl as R10
import box_r5_ctl as B5

_fp = B5.find_plan


def find_plan_sym(io, cxy, ctop, long3, bxy, btop, strict):
    p = _fp(io, cxy, ctop, long3, bxy, btop, strict)
    if p is None:
        p = _fp(io, cxy, ctop, -np.asarray(long3), bxy, btop, strict)
        if p is not None:
            io.log('long_axis_flipped')
    return p


B5.find_plan = find_plan_sym
R10.find_plan = find_plan_sym


def run(io):
    io.log('variant', name='box_r12')
    R10.run(io)
