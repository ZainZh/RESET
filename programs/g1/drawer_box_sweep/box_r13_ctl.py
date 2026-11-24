# box_r13_ctl (r13 = r12 + reorientation grip at the carton centre height (pinch axis through the centre of mass): r12 gripped 12 mm below the top, the carton swung
# about the pinch axis when lifted and was set down standing (seeds 2, 5, 16: z .105 after reorient). Main right-arm plan grip unchanged. At most one reorientation
# (seed 4 used two and hit the 40 s cap). Everything else = r12.
import box_r12_ctl as R12
import box_r10_ctl as R10
from box_r5_ctl import HALF_H, perceive, execute

R10.GRIP_BELOW_TOP = HALF_H


def run(io):
    io.log('variant', name='box_r13')
    cxy, ctop, long3, bxy, btop = perceive(io)
    plan = R10.find_plan(io, cxy, ctop, long3, bxy, btop, True)
    if plan is None and R10.reorient(io, cxy, ctop, long3, bxy, btop):
        cxy, ctop, long3, bxy, btop = perceive(io)
        plan = R10.find_plan(io, cxy, ctop, long3, bxy, btop, True)
    if plan is None:
        raise RuntimeError('no feasible grasp/place plan')
    execute(io, plan, bxy, btop)
