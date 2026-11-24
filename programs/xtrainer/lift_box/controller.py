"""lift_box b8 = b7 + world-z yaw search {0,+-30,+-60,90} of the stand pose in the pre-grasp plan and the replan.
b7 docstring follows.
b7 = b6 with a wider mid-offset grid (100 options, nearest-first) for the stand/transfer chain.
b6 docstring follows.
b6 = b5 + failure counters in chain_ok (diagnostic only, planning unchanged).
b5 docstring follows.
b5 = b4 with the self-contact check moved to a robot-only helper (joint vectors + static
robot_only.xml, no live world qpos/data).
b4 docstring follows.
b4 = b3 + same-arm self-contact rejection (scratch-data kinematics) at every IK waypoint of chain_ok.
b3 docstring follows.
b3 = b2 with: stand sense/offset re-planned AFTER the lift from actual encoders (the grasp yaw
flip is still chosen pre-grasp), rotation steps 2.0 s, no settle at mid (b2 mid settle drove the carton into a finger stop).
b2 docstring follows.
b2 = b1 + closed-loop hand pose settling + pre-grasp feasibility over grasp yaw flips.
Evidence (diag_b1_trace1): with the b1 spring grip the carton is fixed in the hand, so its tilt is the hand's
orientation error: 12-17 deg after the stand rotation, up to ~39 deg during the 3.5 s transfer (joint-rate-limited
tracking), and 16-56 deg / cm-level position error at the box, so it lands on an edge and falls flat (dev 0, 15).
Changes:
(1) settle(): after each key move, re-issue the same target (up to 3x) until the encoder-FK TCP is within 2 deg / 5 mm.
(2) transfer in two legs (high waypoint above the box, then the pre-drop pose) with settle after each.
(3) before grasping, search grasp yaw {0, 180 deg} (parallel jaw symmetric about the closing axis) x stand sense
    {+y up, -y up} x mid offsets (incl. lateral) for a fully IK-feasible rotate/transfer/place chain from the predicted
    lifted pose (robot model only). dev 2/17 had no feasible chain for the single l10 option.
Grasp prior, perception and spring grip are b1 unchanged."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp
from rsi import perception_sam3 as S
from rsi import perception_geo as G
from twin.practice.motion import waypoint
from rsi.xtrainer_lift_box.robot_collision import RobotCollision

UP = np.array([0, 0, 1.])
HOLD = .80
YAWS = (0., 30., -30., 60., -60., 90.)  # world-z yaw of the stand pose: carton cross-section is square, scorer ignores yaw
STEPS = (1. / 3, 2. / 3, 5. / 6, 11. / 12, 1.)
OFFSETS = tuple(sorted(((dz, dx, dy) for dz in (0., .05, .10, .15, .20) for dx in (0., .04, .08, .12) for dy in (-.10, -.05, 0., .05, .10)),
   key=lambda o: abs(o[0] - .10) + abs(o[1]) + abs(o[2])))


def angle_deg(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rot.from_matrix(Ra @ Rb.T).as_rotvec())))


COLL = RobotCollision()
STATS = {}


def self_clear(arm, q):
    """b5: robot-only planning check (joint vectors + static robot-only model; no live world state)."""
    return COLL.self_clear(arm, q)


def chain_ok(io, arm, q0, T0, targets):
    """IK-step through [(p, R), ...] from joint seed q0 at pose T0 (20 waypoints per leg); return final q or None."""
    K = io._w.kinematics; sd = q0; st = T0
    for li, (p, R) in enumerate(targets):
        for k in range(1, 21):
            a, b = waypoint(st, p, R, k / 20)
            try:
                sd = K.ik(arm, a, b, sd, max_iterations=400)
            except Exception:
                STATS['ik:n%d:leg%d' % (len(targets), li)] = STATS.get('ik:n%d:leg%d' % (len(targets), li), 0) + 1
                return None
            if not self_clear(arm, sd):
                STATS['self:n%d:leg%d' % (len(targets), li)] = STATS.get('self:n%d:leg%d' % (len(targets), li), 0) + 1
                return None
        st = K.fk(arm, sd)
    return sd


def settle(io, arm, p, R, dur, label, tries=3):
    for k in range(tries):
        T = io.tcp(arm); ep = float(np.linalg.norm(T[:3, 3] - p)); er = angle_deg(T[:3, :3], R)
        if ep < .005 and er < 2.:
            break
        io.move(arm, p, R, dur)
    T = io.tcp(arm)
    io.log('settle', what=label, pos_err_mm=round(float(np.linalg.norm(T[:3, 3] - p)) * 1000, 1), rot_err_deg=round(angle_deg(T[:3, :3], R), 1))


def run(io):
    ev = io.demo['events']
    close = next(e for e in ev if e['kind'] == 'close')
    arm, R0 = close['arm'], close['rotation']
    ai = ('left', 'right').index(arm)
    obs = io.observe()
    COLL.set_encoders(obs['encoders'])  # robot joint encoders only
    plane = G.fit_plane(obs)
    from PIL import Image
    proc = S.get_proc()
    state = proc.set_image(Image.fromarray(obs["rgb"]))
    pp, ys, xs = G.safe_points(obs)
    hh = G.height(pp, plane)
    insts = []
    mk, sc = S._prompt(proc, state, "small box")
    for m, sv in zip(mk, sc):
        sel = m[ys, xs] & (hh > 0.005) & (hh < 0.20)
        if sv >= 0.5 and sel.sum() >= 100:
            q = pp[sel]
            insts.append({"points": q, "top": float(np.percentile(q[:, 2], 95)), "score": float(sv)})
    if len(insts) < 2:
        raise RuntimeError("small box found %d instances" % len(insts))
    insts = sorted(insts, key=lambda d: -d["top"])[:2]
    box, carton = insts[0], insts[1]
    top = box["points"][box["points"][:, 2] > box["top"] - .012]
    bxy = top[:, :2].mean(0)
    cp = carton['points']
    ctop = cp[cp[:, 2] > carton['top'] - .01]
    cxy = (ctop[:, :2].min(0) + ctop[:, :2].max(0)) / 2 if len(ctop) > 20 else cp[:, :2].mean(0)
    xy0 = ctop[:, :2] - ctop[:, :2].mean(0)
    w_, v_ = np.linalg.eigh(np.cov(xy0.T)) if len(ctop) > 20 else (None, np.eye(2))
    short = v_[:, 0]
    io.log('perceived', carton=cxy, carton_top=carton['top'], box=bxy, box_top=box['top'])
    t = np.r_[cxy, carton['top'] - .03]
    y_now = R0[:2, 1] / max(np.linalg.norm(R0[:2, 1]), 1e-9)
    ang = np.arctan2(short[1], short[0]) - np.arctan2(y_now[1], y_now[0])
    ang = (ang + np.pi / 2) % np.pi - np.pi / 2
    long_dir = np.r_[-short[1], short[0], 0.]; long_dir = long_dir / max(np.linalg.norm(long_dir), 1e-9)
    Rc = np.column_stack([np.cross(long_dir, UP), long_dir, UP])
    drop = np.r_[bxy, box['top'] + .05 + .008]
    high = np.r_[bxy, max(drop[2] + .15, .33)]
    K = io._w.kinematics
    q_now = io._w.encoders()[ai]; T_now = io.tcp(arm)
    plan = None
    for flip in (0., np.pi):
        Rg = Rot.from_euler('z', ang + flip).as_matrix() @ R0
        qg = chain_ok(io, arm, q_now, T_now, [(t + .10 * UP, Rg), (t, Rg), (t + .18 * UP, Rg)])
        if qg is None:
            io.log('plan_grasp_infeasible', flip_deg=round(float(np.degrees(flip))))
            continue
        Tl = K.fk(arm, qg); Rh = Tl[:3, :3]
        H = Rc.T @ Rh
        for sgn, yaw in [(s_, y_) for y_ in YAWS for s_ in (1., -1.)]:
            yg = sgn * UP
            Rs = Rot.from_euler('z', yaw, degrees=True).as_matrix() @ np.column_stack([Rc[:, 0], yg, np.cross(Rc[:, 0], yg)]) @ H
            sl = Slerp([0., 1.], Rot.from_matrix(np.stack([Rh, Rs])))
            base = Tl[:3, 3] + .04 * UP
            for dz, dx, dy in OFFSETS:
                mid = base + np.array([dx, dy, dz])
                legs = [(mid, sl([f]).as_matrix()[0]) for f in STEPS] + [(high, Rs), (drop + .06 * UP, Rs), (drop, Rs)]
                if chain_ok(io, arm, qg, Tl, legs) is not None:
                    plan = dict(Rg=Rg, Rs=Rs, sl=sl, mid=mid, flip=flip, sgn=sgn, off=(dz, dx, dy), yaw=yaw); break
            if plan: break
        if plan: break
    if plan is None:
        io.log('plan_none', stats=dict(STATS))
        Rg = Rot.from_euler('z', ang).as_matrix() @ R0
        plan = None
    else:
        Rg = plan['Rg']
        io.log('plan', flip_deg=round(float(np.degrees(plan['flip']))), stand_sign=plan['sgn'], offset=plan['off'], yaw=plan['yaw'])
    io.grip(arm, 1., .3)
    io.move(arm, t + .10 * UP, Rg, 1.5)
    io.move(arm, t, Rg, 1.0)
    io.grip(arm, 0., .6)
    io.grip(arm, HOLD, .8)
    io.move(arm, t + .18 * UP, Rg, 1.2)
    if plan is None:
        raise RuntimeError('no feasible stand plan')
    # re-plan from the actual lifted pose (encoder FK), same hand-carton relation; first option = the pre-grasp choice
    Tl = io.tcp(arm); Rh = Tl[:3, :3]; H = Rc.T @ Rh; ql = io._w.encoders()[ai]
    found = None
    for sgn, yaw in [(s_, y_) for y_ in (plan['yaw'],) + tuple(v for v in YAWS if v != plan['yaw']) for s_ in (plan['sgn'], -plan['sgn'])]:
        yg = sgn * UP
        Rs = Rot.from_euler('z', yaw, degrees=True).as_matrix() @ np.column_stack([Rc[:, 0], yg, np.cross(Rc[:, 0], yg)]) @ H
        sl = Slerp([0., 1.], Rot.from_matrix(np.stack([Rh, Rs])))
        for dz, dx, dy in (plan['off'],) + OFFSETS:
            mid = Tl[:3, 3] + .04 * UP + np.array([dx, dy, dz])
            legs = [(mid, sl([f]).as_matrix()[0]) for f in STEPS] + [(high, Rs), (drop + .06 * UP, Rs), (drop, Rs)]
            if chain_ok(io, arm, ql, Tl, legs) is not None:
                found = (Rs, sl, mid); io.log('replan', stand_sign=sgn, offset=(dz, dx, dy), yaw=yaw); break
        if found: break
    if found is None:
        raise RuntimeError('no feasible stand plan from the actual lifted pose')
    Rs, sl, mid = found
    for f in STEPS:
        io.move(arm, mid, sl([f]).as_matrix()[0], 2.0)
    io.move(arm, high, Rs, 3.0)
    settle(io, arm, high, Rs, 2.0, 'high')
    io.move(arm, drop + .06 * UP, Rs, 2.5)
    settle(io, arm, drop + .06 * UP, Rs, 1.5, 'predrop')
    io.move(arm, drop, Rs, 2.0)
    settle(io, arm, drop, Rs, 1.0, 'drop')
    io.grip(arm, 1., .6)
    io.wait(.5)
    for h in (.10, .06, .03):
        try:
            io.move(arm, drop + h * UP, Rs, 1.6)
            break
        except Exception as e:
            if type(e).__name__ in ('NumericalFailure', 'Limit'):
                raise
            io.log('lift_retry', h=h)
    io.wait(2.0)
    io.observe()
