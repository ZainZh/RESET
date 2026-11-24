'''put_mug_on_box r2 = r1 + (a) HIGH carry: lateral move to the box at drop height + CARRY_UP (6 cm), then vertical descent (r1 DEV1/12/16/17: the held mug slips ~7 mm and tilts ~18 deg; its lower rim hit the box edge on r1's 1.5 cm-clearance lateral move and the mug tipped); low r1 carry kept only as fallback if no high-carry plan; (b) if SAM3 misses the mug after the relay (r1 DEV3/11/19: mug at the bottom edge of the head image), its centre is taken from the right TCP forward kinematics at release minus the planned wall offset (own commanded action + encoders).
r1 = l5 + table RELAY (right arm moves the mug to a handover spot, left arm places it on the box)
+ box centre = bbox midpoint of the top 4 mm band (l10) + empty-grasp check with re-plan (l9).
Evidence: on o13 the RIGHT arm reaches the mug wall on every DEV seed (18-36 (pitch, yaw) grasps) but reaches
the box top on almost none; the LEFT arm reaches the box but only 1-5 mug grasps on most seeds, and in l5 DEV3/7/8 its wrist camera
(left_d405_link / mount) hit the box while reaching across it to the mug, stopping the TCP 2.5-7 cm above the grasp (empty close).
Both arms reach a table band in front of the robot (y -0.40..-0.46, x -0.10..0.11), away from the box.
Observation-only: SAM3 masks on head RGB-D, fitted table plane, encoders, canonical demo priors (grasp depth below rim).'''
import numpy as np
from rsi import perception as P
from rsi import perception_sam3 as S
from rsi import perception_geo as G
from twin.practice.motion import waypoint
import mujoco
from scipy.spatial.transform import Rotation as Rot
from rsi.lift2_suite.common.embodiment import lift2 as ROBOT

UP = np.array([0, 0, 1.])
YAWS = np.radians(np.arange(0, 360, 15))
PITCH_L = np.radians([35, 20, 0, 50, 65])   # left (placing) arm; pitch-50 grasps all failed on o13 l5 (mug swings), so tried last
PITCH_R = np.radians([0, 20, 35])           # right (relay) arm: low pitch, short table-to-table move
HANDOVER = [(.00, -.44), (-.03, -.44), (.03, -.44), (.00, -.41), (-.03, -.41), (.03, -.41), (.00, -.47), (-.06, -.44), (.06, -.44), (-.06, -.41)]
BOX_KEEP = .10      # handover mug centre must be this far outside the perceived box top rectangle
RC = None
MAX_CHECKS = 9000
CHECKS = 0
GDEPTH, WALL = .025, .042   # demo prior: hand grasps the mug wall ~1-2.5 cm below the rim
GRIP_EMPTY = .0015
CARRY_UP = .06
BAND = .004


def fit_table(obs, band=.008, frac=.05):
    p = G.safe_points(obs)[0]; z = p[:, 2]
    edges = np.arange(np.percentile(z, .5), np.percentile(z, 99.5) + .005, .005); cnt, _ = np.histogram(z, edges)
    k = int(np.nonzero(cnt >= frac * len(z))[0][0]); z0 = (edges[k] + edges[k + 1]) / 2
    sel = np.abs(z - z0) < .012; c = np.array([0, 0, z0]); n = np.array([0, 0, 1.])
    for _ in range(4):
        near = p[sel]
        if len(near) < 200: break
        c = near.mean(0); _, _, vt = np.linalg.svd(near[::7] - c); n = vt[-1] * np.sign(vt[-1][2]); sel = np.abs((p - c) @ n) < band
    return c, n


class RobotCollision:
    '''As l5: robot-only clearance from joint vectors on the static LIFT2 planning model (table moved to the fitted plane).'''
    CLEAR = .002

    def __init__(self, enc, plane, T_task_base):
        m = self.m = mujoco.MjModel.from_xml_path(str(ROBOT.MJCF)); self.d = mujoco.MjData(m)
        m.geom_margin[:] = .003
        self.adr = [np.array([m.joint(j).qposadr[0] for j in ROBOT.ARMS[a]['joints']]) for a in ('left', 'right')]
        mujoco.mj_forward(m, self.d); b = m.body(ROBOT.BASE_BODY).id
        Tm = np.eye(4); Tm[:3, :3] = self.d.xmat[b].reshape(3, 3); Tm[:3, 3] = self.d.xpos[b]
        M = Tm @ np.linalg.inv(np.asarray(T_task_base, float))
        c = M[:3, :3] @ np.asarray(plane[0], float) + M[:3, 3]; n = M[:3, :3] @ np.asarray(plane[1], float)
        t = m.geom('table').id; hz = float(m.geom_size[t][2])
        q = np.roll(Rot.align_vectors([n], [[0, 0, 1.]])[0].as_quat(), 1)
        m.geom_pos[t] = c - n * hz; m.geom_quat[t] = q; m.geom_size[t][:2] = 2.; m.geom_rbound[t] = np.linalg.norm(m.geom_size[t]); m.geom_aabb[t][3:] = m.geom_size[t]
        self.enc = np.asarray(enc, float).reshape(2, 6).copy()
        self.base = self.contacts(self.enc[0], 0)

    def contacts(self, q, i):
        d = self.d
        d.qpos[self.adr[i]] = q; d.qpos[self.adr[1 - i]] = self.enc[1 - i]
        mujoco.mj_forward(self.m, d)
        return {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)): float(c.dist) for c in d.contact[:d.ncon]}

    def clear(self, q, i):
        for k, v in self.contacts(q, i).items():
            if k in self.base:
                if v < 0 and v < self.base[k] - .0003:
                    return False
            elif v < self.CLEAR:
                return False
        return True


def locate(obs, prompts, max_height):
    proc = S.get_proc(); plane = fit_table(obs)
    for pr in prompts:
        r = S.locate_object(proc, obs, pr, plane=plane, max_height=max_height, min_score=.3)
        if r is not None:
            return r, pr
    return None, None


def mug_pose(mug):
    rim = mug['points'][mug['points'][:, 2] > mug['top'] - .02]
    return np.asarray(P.fit_upright_cylinder(rim, .042)[0], float), float(mug['top'])


def frame(u, pitch=0.):
    y = np.r_[u, 0.]; x0 = np.cross(y, -UP); z = -np.cos(pitch) * UP + np.sin(pitch) * x0; return np.stack([np.cross(y, z), y, z], 1)


def feasible(io, arm, path):
    global CHECKS
    CHECKS += 1
    if CHECKS > MAX_CHECKS:
        raise RuntimeError('plan search bound reached')
    kin = io._w.kinematics; i = ('left', 'right').index(arm)
    q = io._w.encoders()[i]; start = io.tcp(arm)
    for p, R in path:
        for k in range(1, 11):
            wp, wr = waypoint(start, p, R, k / 10)
            try:
                q = io._ik(arm, wp, wr, q)
            except Exception:
                return False
            if RC is not None and not RC.clear(q, i):
                return False
        start = kin.fk(arm, q)
    return True


def plan_place(io, arm, mxy, mtop, bxy, btop, table, skip=(), pitches=PITCH_L, first_only=False):
    '''l5 search for one arm: grasp the wall at mxy, carry low, drop on the box (same pitch, any drop yaw).'''
    for pitch in pitches:
        for yaw in YAWS:
            key = (arm, round(float(pitch), 3), round(float(yaw), 3))
            if key in skip:
                continue
            u = np.array([np.cos(yaw), np.sin(yaw)]); R = frame(u, pitch)
            t = np.r_[mxy + WALL * u, mtop - GDEPTH]; hold = t[2] - table
            if not feasible(io, arm, [(t + .05 * UP, R), (t, R)]):
                continue
            for yd in YAWS:
                ud = np.array([np.cos(yd), np.sin(yd)]); Rd = frame(ud, pitch)
                for up in (CARRY_UP, .015):
                    for dz in (.008, .02):
                        drop = np.r_[bxy + WALL * ud, btop + hold + dz]
                        path = [(t + .05 * UP, R), (t, R), (np.r_[t[:2], drop[2] + up], R), (drop + up * UP, Rd), (drop, Rd), (drop + .03 * UP, Rd)]
                        if feasible(io, arm, path):
                            return key, path
            if first_only:
                return None
    return None


def outside_box(xy, blo, bhi, keep):
    d = np.maximum(np.maximum(blo - xy, xy - bhi), 0)
    return float(np.linalg.norm(d)) >= keep


def plan_relay(io, mxy, mtop, bxy, btop, blo, bhi, table, skip=()):
    '''Right arm: grasp the wall at mxy, lift, translate (same orientation) to a handover spot H, set the mug down. H must be reachable by the
    left arm for a full place plan from H (nominal: mug top unchanged).'''
    grasps = []
    for pitch in PITCH_R:
        for yaw in YAWS:
            key = ('right', round(float(pitch), 3), round(float(yaw), 3))
            if key in skip:
                continue
            u = np.array([np.cos(yaw), np.sin(yaw)]); R = frame(u, pitch); t = np.r_[mxy + WALL * u, mtop - GDEPTH]
            if feasible(io, 'right', [(t + .05 * UP, R), (t, R)]):
                grasps.append((key, u, R, t))
            if len(grasps) >= 6:
                break
        if len(grasps) >= 6:
            break
    io.log('relay_grasps', n=len(grasps))
    left_cache = {}
    for H in HANDOVER:
        H = np.array(H)
        if not outside_box(H, blo, bhi, BOX_KEEP):
            continue
        if np.linalg.norm(H - mxy) < .03:
            continue
        for key, u, R, t in grasps:
            th = np.r_[H + WALL * u, t[2] + .004]; lift = t[2] + .06
            path = [(t + .05 * UP, R), (t, R), (np.r_[t[:2], lift], R), (np.r_[th[:2], lift], R), (th, R), (th + .06 * UP, R)]
            if not feasible(io, 'right', path):
                continue
            hk = tuple(H)
            if hk not in left_cache:
                left_cache[hk] = plan_place(io, 'left', H, mtop, bxy, btop, table)
            if left_cache[hk] is None:
                break  # no left place plan from this H: next H
            return key, H, path, left_cache[hk]
    return None


def grasp(io, arm, path, tg=1.0):
    i = ('left', 'right').index(arm)
    io.grip(arm, 1., .3)
    io.move(arm, *path[0], 1.5)
    io.move(arm, *path[1], tg)
    io.grip(arm, 0., .6)
    g = float(io._w.gripper_positions()[i])
    return g


def retreat(io, arm, pre):
    io.grip(arm, 1., .4)
    io.move(arm, *pre, .8)
    try:
        io.home(arm, 1.5)
    except Exception as e:
        io.log('home_failed', arm=arm, err=str(e)[:60])
    RC.enc = io._w.encoders().reshape(2, 6).copy()


def perceive_mug(io, maxh=.14):
    obs = io.observe(); mug, _ = locate(obs, ('gray mug', 'mug', 'cup'), maxh)
    return None if mug is None else mug_pose(mug)


def run(io):
    global RC
    obs = io.observe()
    RC = RobotCollision(io._w.encoders(), fit_table(obs), io._w.kinematics.T_task_base)
    mug, mp = locate(obs, ('gray mug', 'mug', 'cup'), .14)
    box, bp = locate(obs, ('wooden box', 'box'), .16)
    if mug is None or box is None:
        raise RuntimeError(f'perception failed mug={mug is not None} box={box is not None}')
    mxy, mtop = mug_pose(mug)
    tb = box['points'][box['points'][:, 2] > box['top'] - BAND]
    blo, bhi = tb[:, :2].min(0), tb[:, :2].max(0)
    bxy = (blo + bhi) / 2; btop = box['top']; table = mug['table']
    io.log('perceived', mug=mxy, mug_top=mtop, mug_prompt=mp, box=bxy, box_lo=blo, box_hi=bhi, box_top=btop, box_prompt=bp, table=table)

    # Stage 1: relay with the right arm (up to 3 grasp attempts)
    skip = set()
    for attempt in range(3):
        res = plan_relay(io, mxy, mtop, bxy, btop, blo, bhi, table, skip)
        if res is None:
            raise RuntimeError('no feasible relay plan')
        key, H, path, _ = res
        io.log('relay_plan', attempt=attempt, H=H, grasp=path[1][0], put=path[4][0], checks=CHECKS)
        g = grasp(io, 'right', path)
        io.log('relay_grasped', finger=g)
        if g >= GRIP_EMPTY:
            break
        skip.add(key); retreat(io, 'right', path[0])
        m2 = perceive_mug(io)
        if m2 is not None:
            mxy, mtop = m2
    else:
        raise RuntimeError('relay grasp empty after 3 attempts')
    io.move('right', *path[2], 1.0)
    io.move('right', *path[3], 2.0)
    io.move('right', *path[4], 1.2)
    put_tcp = io.tcp('right')[:3, 3].copy(); relay_u = np.asarray(path[4][1], float)[:2, 1]
    io.grip('right', 1., .8)
    io.move('right', *path[5], 1.0)
    try:
        io.home('right', 2.0)
    except Exception as e:
        io.log('home_failed', arm='right', err=str(e)[:60])
    RC.enc = io._w.encoders().reshape(2, 6).copy()
    io.wait(.3)

    # Stage 2: re-perceive the mug at the handover spot, left arm places it on the box
    m2 = perceive_mug(io)
    if m2 is None:
        mxy = np.asarray(put_tcp[:2], float) - WALL * relay_u
        io.log('relay_mug_from_fk', mug=mxy)
    else:
        mxy, mtop = m2
    io.log('relay_done', mug=mxy, mug_top=mtop, upright=bool(mtop - table > .06))
    skip = set()
    for attempt in range(3):
        res = plan_place(io, 'left', mxy, mtop, bxy, btop, table, skip)
        if res is None:
            raise RuntimeError('no feasible left place plan after relay')
        key, path = res
        io.log('plan', arm='left', attempt=attempt, grasp=path[1][0], drop=path[4][0], checks=CHECKS)
        g = grasp(io, 'left', path)
        io.log('stage_grasped', finger=g)
        if g >= GRIP_EMPTY:
            break
        skip.add(key); retreat(io, 'left', path[0])
        m2 = perceive_mug(io)
        if m2 is not None:
            mxy, mtop = m2
    else:
        raise RuntimeError('left grasp empty after 3 attempts')
    io.move('left', *path[2], 1.2)
    io.move('left', *path[3], 1.5)
    io.move('left', *path[4], 1.0)
    io.log('stage_pre_release')
    io.grip('left', 1., .6)
    io.move('left', *path[5], .8)
    io.wait(2.5)
    io.observe()
