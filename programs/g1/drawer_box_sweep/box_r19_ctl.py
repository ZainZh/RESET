# box_r19_ctl (r19 = r17 + PUSH fallback): if the proven right-arm plan is infeasible from the perceived carton pose, the LEFT hand (open, fingertips
# down, flat side) pushes the lying carton on the table by a fixed vector chosen from PUSHES (toward -y / +x, i.e. into the right arm's reach), only if the proven plan is
# feasible from the predicted pushed pose (model-only check; prediction = pure translation). No grasp, so the carton yaw is kept. Then re-perceive -> proven plan.
# If no push candidate qualifies, r17's re-orientation fallback runs. Runtime: io only.
import math
import numpy as np
from drawer_r8_ctl import Rax, safe_move
import box_r17_ctl as R17
import box_r10_ctl as R10
from box_r5_ctl import UP, HALF_H, perceive, execute
import box_r5_ctl as B5

PUSHES = [np.array(v) for v in ((.0, -.04), (.0, -.06), (.03, -.04), (.04, 0.), (.03, -.06), (.0, -.08), (.05, -.03))]


def push(io, cxy, ctop, long3, bxy, btop):
    short = np.cross(UP, long3)[:2]; short /= np.linalg.norm(short)
    for v in PUSHES:
        nxy = cxy + v
        if R10.find_plan(io, nxy, ctop, long3, bxy, btop, True) is None:
            continue
        dirv = v / np.linalg.norm(v)
        # contact point on the carton side opposite to the push, at mid height; hand approaches from above behind that side
        half = .0325 * abs(dirv @ short) + .05 * abs(dirv @ long3[:2])
        c0 = np.r_[cxy - dirv * (half + .025), ctop - .030]
        a = np.array([0., 0., -1.])
        R = Rax(a, np.r_[dirv, 0.])  # pad axis along the push: palm/fingers face the carton side
        w = B5.chain_margin(io, 'left', [(c0 + .08 * UP, R), (c0, R), (np.r_[c0[:2] + v, c0[2]], R), (np.r_[c0[:2] + v, c0[2]] + .08 * UP, R)])
        if w is None or w < .04:
            continue
        io.log('push', v=v, margin=w)
        io.grip('left', 1., .3)
        safe_move(io, 'left', c0 + .08 * UP, R, 1.5)
        safe_move(io, 'left', c0, R, .8)
        n = 4
        for k in range(1, n + 1):
            safe_move(io, 'left', np.r_[c0[:2] + v * k / n, c0[2]], R, .6)
        safe_move(io, 'left', np.r_[c0[:2] + v, c0[2]] + .08 * UP, R, .8)
        io.home('left', 1.5)
        io.wait(.3)
        io.log('pushed')
        return True
    io.log('push_infeasible')
    return False


def run(io):
    io.log('variant', name='box_r19')
    cxy, ctop, long3, bxy, btop = perceive(io)
    plan = R10.find_plan(io, cxy, ctop, long3, bxy, btop, True)
    if plan is None and push(io, cxy, ctop, long3, bxy, btop):
        cxy, ctop, long3, bxy, btop = perceive(io)
        plan = R10.find_plan(io, cxy, ctop, long3, bxy, btop, True)
    if plan is None and R17.reorient(io, cxy, ctop, long3, bxy, btop):
        cxy, ctop, long3, bxy, btop = perceive(io)
        plan = R10.find_plan(io, cxy, ctop, long3, bxy, btop, True)
    if plan is None:
        raise RuntimeError('no feasible grasp/place plan')
    execute(io, plan, bxy, btop)
