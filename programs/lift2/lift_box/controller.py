"""lift_box lift2_bx9 = bx8 + drop point search on the observed box top. Evidence: bx8 DEV0-19 11/20, 7 of 9 failures plan_none before any motion; offline probe (robot model only) finds full stand plans off the box-top centre on failing layouts. Candidates must keep the carton centre at least FOOT from the edges of the observed top rectangle. The replan after the lift tries the planned drop first, then the other candidates.
lift_box bx8 = bx7 with the stand yaw grid refined from 30 to 10 deg (36 yaws, nearest to 0 first). Evidence: bx7 fails 10 of the first 17 DEV seeds with
plan_none; offline diag_bx7 shows 48/48 stands rejected at the drop IK on failing layouts, while DEV0 and DEV1 have nearly equal box positions and differ in carton yaw sign,
which suggests a narrow window of workable approach directions stepped over by the 30 deg grid. Only this grid changes.
lift_box bx7 = bx6 with a drop-first stand filter: each stand rotation must first have a robot-clear IK solution AT THE DROP pose (seeds as bx5);
that solution seeds the transfer-pose IK, so the transfer pose and the place legs lie on one IK branch. Evidence: bx5 DEV0-5 3/6, all failures plan_none; bx6 offline
search exceeded 560 s on most layouts because every stand x height x seed ran a joint-path check before the place chain was known to exist.
lift_box bx6 = bx5 + several transfer waypoints: the joint-space target pose may be any of HIGHS (height above the drop point, base-ward
xy shift); the Cartesian legs from there to predrop/drop are robot-checked as before. Evidence: offline diag_bx5 (DEV 0/1/3/6): stand rejections are
dominated by IK failure at the single high pose (box top + .208 m, z .33): left 39-61 of ~64 per layout.
lift_box bx5 = bx2 perception/grasp + a JOINT-SPACE stand transfer (LIFT2-specific plan). Evidence: bx2 DEV0-19 partial
2/3, failures plan_none: the b8 Cartesian slerp stand chain from the top-down grasp branch is rejected ~1000x on its first leg (left_link13|left_link15
self-contact, real in the task scene), and bx3/bx4 (other arm, 12 yaws) still found no Cartesian chain (offline: 4/20 layouts, others > 10 min search).
Plan: grasp chain as bx2 (arm left then right, flips x tilts); for every stand rotation (2 signs x 12 yaws) IK the place pose at `high` from several
seeds, require the Cartesian high -> predrop -> drop legs (chain_ok, robot-only check) and a robot-clear linear joint path lifted -> high with the TCP
kept above the box top + 7 cm; execute that path as a slew-limited joint move (io.targets, same channel as io.home), then settle/drop/release as b8.
lift_box bx2 = bx1 + (a) HOLD 0.80 -> 0.0: on LIFT2 opening 0.80 = 67 mm > 65 mm carton, so the b8 'hold' re-opened the
fingers (bx1 smoke DEV0: carton final xy == initial xy, never lifted); (b) grasp rotation candidates: b8 flips {0,180} x approach tilt
about the closing axis {0,+15,-15,+30,-30} deg (bx1 smoke DEV3: plan_grasp_infeasible for both flips).
lift_box bx1 (LIFT2, smooth_table_v1 + constant nominal offsets, setup lift_box_o4) = source b8 with two LIFT2 changes:
(1) table plane: lowest dominant plane (two-pass RANSAC on head RGB-D points; keep the lower plane when it has >= 12% of points).
    Evidence: bx0_diag DEV0/3 -> rsi.perception_geo.fit_plane locked onto the wooden box top (z0 0.143, box top covers much of the
    view on this setup), every SAM3 mask point had height ~0, b8 raised 'small box found 0 instances' on 20/20 (source baseline).
(2) robot-only planning check: LIFT2 checked MJCF (robot bodies only, test object geoms disabled), joint vectors only; rejects new
    robot-robot contacts deeper than 0 m relative to the home pose (replaces the xTrainer-only RobotCollision helper).
Everything else (SAM3 'small box' prompt, grasp prior rotation, stand/yaw/offset search, settle, drop, release) = b8.
b8 docstring follows.
lift_box b8 = b7 + world-z yaw search {0,+-30,+-60,90} of the stand pose in the pre-grasp plan and the replan.
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
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp
from rsi import perception_sam3 as S
from rsi import perception_geo as G
from twin.practice.motion import waypoint
import mujoco
from rsi.lift2_suite.common.embodiment import lift2 as ROBOT


class RobotCollision:
    """bx1: LIFT2 robot-only self/inter-arm check from joint vectors (static robot model, no task objects, no live sim data)."""
    def __init__(self):
        m = self.m = mujoco.MjModel.from_xml_path(str(ROBOT.MJCF)); self.d = mujoco.MjData(m)
        for g in range(m.ngeom):
            if m.body(int(m.geom_bodyid[g])).name.startswith('object'):
                m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0
        self.adr = {a: np.array([m.joint(j).qposadr[0] for j in ROBOT.ARMS[a]['joints']]) for a in ('left', 'right')}
        self.other = None; self.home_pairs = None

    def _contacts(self, arm, q):
        m, d = self.m, self.d
        for a, k in (('left', 0), ('right', 1)):
            d.qpos[self.adr[a]] = np.asarray(q, float) if a == arm else self.other[k]
        mujoco.mj_forward(m, d)
        return {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)): float(c.dist) for c in d.contact[:d.ncon]}

    def set_encoders(self, enc):
        self.other = np.asarray(enc, float).reshape(2, 6).copy()
        self.home_pairs = self._contacts('left', self.other[0])

    def self_clear(self, arm, q):
        for k, dist in self._contacts(arm, q).items():
            if dist < 0 and (k not in self.home_pairs or dist < self.home_pairs[k] - .0003):
                return False
        return True


def fit_plane_low(obs, band=.006, iters=150, min_frac=.12):
    """bx1: lowest dominant plane. RANSAC (fixed seed) best plane, then RANSAC on the remaining points; keep the lower one if it holds
    >= min_frac of the points. Returns (centre, unit normal with +z) like perception_geo.fit_plane."""
    p, _, _ = G.safe_points(obs); p = p[::4]; rng = np.random.default_rng(0)

    def ransac(pts):
        best = (0, None)
        for _ in range(iters):
            a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
            n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
            if nn < 1e-9: continue
            n = n / nn * np.sign(n[2] or 1.)
            if n[2] < .95: continue
            k = np.abs((pts - a) @ n) < band
            if k.sum() > best[0]: best = (int(k.sum()), k)
        k = best[1]; q = pts[k]; c = q.mean(0); _, _, vt = np.linalg.svd(q[::3] - c, full_matrices=False); n = vt[-1] * np.sign(vt[-1][2])
        return c, n, np.abs((pts - c) @ n) < band
    c1, n1, k1 = ransac(p); rest = p[~k1]
    if len(rest) > .12 * len(p):
        c2, n2, k2 = ransac(rest)
        if k2.sum() >= min_frac * len(p) and (c2 - c1) @ n1 < -.02:
            return c2, n2
    return c1, n1

UP = np.array([0, 0, 1.])
HOLD = 0.0  # bx2: LIFT2 aperture mapping (0.80 would be 67 mm > carton 65 mm)
TILTS = (0., 15., -15., 30., -30.)
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


STAND_YAWS = tuple(sorted((float(y) for y in range(-170, 181, 10)), key=abs))
JSPEED = .6  # rad/s joint-space transfer speed
SEEDS = ([0., 1.5, 1.5, 0., 0., 0.], [0., 1.2, 1.8, 1.2, 0., 0.], [0., 1.2, 1.8, -1.2, 0., 0.], [.6, 1.4, 1.6, 0., .8, 0.], [-.6, 1.4, 1.6, 0., -.8, 0.],
         [0., 2.0, 1.0, 1.5, 0., 1.5], [0., 2.0, 1.0, -1.5, 0., -1.5])


def joint_path_ok(io, arm, qa, qb, zmin, n=40):
    K = io._w.kinematics
    for k in range(1, n + 1):
        q = qa + (qb - qa) * k / n
        if not self_clear(arm, q) or K.fk(arm, q)[2, 3] < zmin:
            return False
    return True


HIGHS = ((.13, 0.), (.09, 0.), (.13, .05), (.09, .05), (.06, 0.))  # (dz above drop, xy shift toward the robot base) m


def stand_plan(io, arm, ql, Tl, Rc, high, drop, zmin):
    """From the lifted joint vector ql (pose Tl): first stand rotation with an IK solution at high, a clear Cartesian place chain and a clear joint path."""
    K = io._w.kinematics; Rh = Tl[:3, :3]; H = Rc.T @ Rh
    for sgn, yaw in [(s_, y_) for y_ in STAND_YAWS for s_ in (1., -1.)]:
        yg = sgn * UP
        Rs = Rot.from_euler('z', yaw, degrees=True).as_matrix() @ np.column_stack([Rc[:, 0], yg, np.cross(Rc[:, 0], yg)]) @ H
        drop_seeds = []
        for sd in (ql,) + tuple(np.array(x) for x in SEEDS):
            try:
                qd = K.ik(arm, drop, Rs, np.asarray(sd, float), max_iterations=200)
            except Exception:
                continue
            if self_clear(arm, qd) and not any(np.abs(qd - x).max() < .05 for x in drop_seeds):
                drop_seeds.append(qd)
        if not drop_seeds:
            continue
        base_xy = K.T_task_base[:2, 3]
        for dzh, back in HIGHS:
            v = base_xy - drop[:2]; v = v / max(np.linalg.norm(v), 1e-9)
            hp = np.r_[drop[:2] + back * v, drop[2] + dzh]
            for sd in drop_seeds:
                try:
                    qh = K.ik(arm, hp, Rs, np.asarray(sd, float), max_iterations=300)
                except Exception:
                    continue
                if not self_clear(arm, qh) or not joint_path_ok(io, arm, ql, qh, zmin):
                    continue
                legs = ([] if back == 0 else [(np.r_[drop[:2], hp[2]], Rs)]) + [(drop + .06 * UP, Rs), (drop, Rs)]
                if chain_ok(io, arm, qh, K.fk(arm, qh), legs) is None:
                    continue
                return dict(Rs=Rs, qh=qh, sgn=sgn, yaw=yaw, high=hp, legs=legs)
    return None


FOOT = .0375  # carton half width .0325 + 5 mm
DROP_OFFSETS = tuple(sorted(((dx, dy) for dx in (-.05, -.025, 0., .025, .05) for dy in (-.06, -.03, 0., .03, .06)), key=lambda o: np.hypot(*o)))


def drop_candidates(top_xy, centre, z):
    """Drop points on the observed box top (observation only): offsets around the perceived centre, centre first, inside the minAreaRect of the top points shrunk by FOOT."""
    rect = cv2.minAreaRect(np.asarray(top_xy, np.float32) * 1000)
    (cx, cy), (w, h), a = rect; a = np.radians(a); ax = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    out = []
    for dx, dy in DROP_OFFSETS:
        xy = centre + [dx, dy]; u = ax @ (xy * 1000 - [cx, cy])
        if abs(u[0]) <= w / 2 - FOOT * 1000 and abs(u[1]) <= h / 2 - FOOT * 1000:
            out.append(np.r_[xy, z])
    return out, dict(rect_mm=[round(float(w)), round(float(h))], n=len(out))


def jmove(io, arm, q):
    i = ('left', 'right').index(arm); q0 = io.targets[i].copy(); q = np.asarray(q, float)
    n = max(1, int(np.ceil(max(.8, float(np.abs(q - q0).max()) / JSPEED) / io._w.dt)))
    for k in range(n):
        io.targets[i] = q0 + (q - q0) * (k + 1) / n
        io.wait(io._w.dt)
    io.wait(.4)
    io.log('jmove', err=float(np.abs(io._w.encoders()[i] - q).max()))


def run(io):
    ev = io.demo['events']
    close = next(e for e in ev if e['kind'] == 'close')
    arm0, R0 = close['arm'], close['rotation']
    obs = io.observe()
    COLL.set_encoders(obs['encoders'])  # robot joint encoders only
    plane = fit_plane_low(obs)
    io.log("plane", c=plane[0], n=plane[1])
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
    drops, dinfo = drop_candidates(top[:, :2], bxy, drop[2])
    io.log('drop_candidates', **dinfo)
    if not drops:
        drops = [drop]
    high = np.r_[bxy, max(drop[2] + .15, .33)]
    zmin = box['top'] + .07
    K = io._w.kinematics
    plan = None
    for arm in (arm0, 'right' if arm0 == 'left' else 'left'):
        ai = ('left', 'right').index(arm); q_now = io._w.encoders()[ai]; T_now = io.tcp(arm)
        for flip, tilt in [(f_, t_) for t_ in TILTS for f_ in (0., np.pi)]:
            Rg = Rot.from_euler('z', ang + flip).as_matrix() @ R0 @ Rot.from_euler('y', tilt, degrees=True).as_matrix()
            qg = chain_ok(io, arm, q_now, T_now, [(t + .10 * UP, Rg), (t, Rg), (t + .18 * UP, Rg)])
            if qg is None:
                continue
            sp = None
            for dc in drops:
                sp = stand_plan(io, arm, qg, K.fk(arm, qg), Rc, high, dc, zmin)
                if sp is not None:
                    sp['drop'] = dc; break
            if sp is not None:
                plan = dict(arm=arm, Rg=Rg, flip=flip, tilt=tilt, **sp); break
            io.log('stand_infeasible', arm=arm, flip_deg=round(float(np.degrees(flip))), tilt=tilt)
        if plan: break
    if plan is None:
        io.log('plan_none', stats=dict(STATS))
        raise RuntimeError('no feasible grasp + joint-space stand plan')
    arm = plan['arm']; Rg = plan['Rg']
    io.log('plan', drop=plan['drop'], arm=arm, tilt=plan['tilt'], flip_deg=round(float(np.degrees(plan['flip']))), stand_sign=plan['sgn'], yaw=plan['yaw'])
    io.grip(arm, 1., .3)
    io.move(arm, t + .10 * UP, Rg, 1.5)
    io.move(arm, t, Rg, 1.0)
    io.grip(arm, 0., .6)
    io.grip(arm, HOLD, .8)
    io.move(arm, t + .18 * UP, Rg, 1.2)
    # re-plan the stand from the actual lifted encoders (same hand-carton relation), first try the pre-grasp choice
    ai = ('left', 'right').index(arm); ql = io._w.encoders()[ai]
    sp = None
    for dc in [plan['drop']] + [x for x in drops if x is not plan['drop']]:
        sp = stand_plan(io, arm, ql, io.tcp(arm), Rc, high, dc, zmin)
        if sp is not None:
            drop = dc; break
    if sp is None:
        raise RuntimeError('no joint-space stand plan from the actual lifted pose')
    io.log('replan', drop=drop, stand_sign=sp['sgn'], yaw=sp['yaw'], high=sp['high'])
    Rs = sp['Rs']
    jmove(io, arm, sp['qh'])
    settle(io, arm, sp['high'], Rs, 2.0, 'high')
    if len(sp['legs']) == 3:
        io.move(arm, sp['legs'][0][0], Rs, 2.0)
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
