'''collect_apple l10 (l9 + rim-clear release: stop vertically above the bowl axis with the apple bottom >= 1.5 cm over the rim, then descend straight down to a release with the apple bottom ~0.5 cm over the rim; one rule for every seed; l8 + clearance 0 mm (no penetration accepted, was 2 mm); grasp yaw over the full 0-345 deg so the wrist can take the flipped (tilt-mirrored) solution; relay arm-B path re-checked after re-perception; l7 + robot-only self/inter-arm/table clearance check on every planned waypoint and on the joint path home; l6 + relay handover grid 3 cm, 7x9 points around the apple-bowl midpoint, nearest first; l5 + yaw candidates every 15 deg; l3 + approach pitch candidates at GRASP, single-arm and relay; l3 tried pitch 0 at grasp only). Based on l3 (l2 with lower release rim+2.5 cm and carry rim+5 cm: l2 DEV0/1 had no plan at carry z ~.22 m; a5 perception and stages; LIFT2 plan layer): plan over (arm, yaw, release height) with straight top-down
canonical orientations, IK-check every segment from current encoders before moving, execute the first feasible plan. If no single arm can do
grasp + release, a two-arm relay is planned: arm A places the apple on the table at a handover point both arms can reach, the head re-observes,
arm B grasps it and releases over the bowl. Observation-only: SAM3 apple and bowl from head RGB-D, encoders, canonical demo priors.'''
import numpy as np
from rsi import perception as P
from rsi import perception_sam3 as S
from rsi import perception_geo as G
from twin.practice.motion import waypoint
import mujoco
from scipy.spatial.transform import Rotation as Rot
from rsi.lift2_suite.common.embodiment import lift2 as ROBOT

UP = np.array([0, 0, 1.])
YAWS = np.radians(np.arange(0, 180, 15))  # closing axis direction; a sphere is symmetric under 180 deg
GYAWS = np.radians(np.arange(0, 360, 15))  # l9 grasp yaws: yaw and yaw+180 give the same closing line but mirrored tilt and a flipped wrist
MAX_CHECKS = 8000  # deterministic bound on plan-search segment checks per episode
PITCH = np.radians([0, 20, 35, 50])  # approach lean about the closing axis (LIFT2 top-down reach ends ~0.55 m forward / ~15 cm up)


CHECKS = 0
RC = None


class RobotCollision:
    '''Robot-only clearance from joint vectors: the static LIFT2 planning model (no task objects, no live sim data). Its table geom is moved to the table plane
    fitted from the head depth. New pairs must keep >= CLEAR m; pairs already within margin at the start pose (base on table) may not get deeper.'''
    CLEAR = 0.0  # l9: any penetration rejected; l8's 2 mm rejected every IK-feasible grasp for far-left apples

    def __init__(self, enc, plane, T_task_base):
        m = self.m = mujoco.MjModel.from_xml_path(str(ROBOT.MJCF)); self.d = mujoco.MjData(m)
        m.geom_margin[:] = .003
        self.adr = [np.array([m.joint(j).qposadr[0] for j in ROBOT.ARMS[a]['joints']]) for a in ('left', 'right')]
        mujoco.mj_forward(m, self.d); b = m.body(ROBOT.BASE_BODY).id
        Tm = np.eye(4); Tm[:3, :3] = self.d.xmat[b].reshape(3, 3); Tm[:3, 3] = self.d.xpos[b]
        M = Tm @ np.linalg.inv(np.asarray(T_task_base, float))  # task world -> planning model frame
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

    def joint_path_clear(self, q0, q1, i, n=25):
        return all(self.clear(q0 + (q1 - q0) * k / n, i) for k in range(1, n + 1))
CARRY_Z = None


def top_down(yaw, pitch=0.):
    y = np.r_[np.cos(yaw), np.sin(yaw), 0.]; x0 = np.cross(y, -UP); z = -np.cos(pitch) * UP + np.sin(pitch) * x0; return np.stack([np.cross(y, z), y, z], 1)


def feasible(io, arm, path, start=None, q=None):
    global CHECKS
    CHECKS += 1
    if CHECKS > MAX_CHECKS:
        raise RuntimeError('plan search bound reached')
    kin = io._w.kinematics; i = ('left', 'right').index(arm)
    q = io._w.encoders()[i] if q is None else q; start = io.tcp(arm) if start is None else start
    for p, R in path:
        for k in range(1, 11):
            wp, wr = waypoint(start, p, R, k / 10)
            try:
                q = io._ik(arm, wp, wr, q)
            except Exception:
                return None
            if RC is not None and not RC.clear(q, i):
                return None
        start = kin.fk(arm, q)
    return q


def perceive(io):
    global RC
    obs = io.observe(); proc = S.get_proc(); plane = G.fit_plane(obs)
    if RC is None:
        RC = RobotCollision(io._w.encoders(), plane, io._w.kinematics.T_task_base)
    RC.enc = io._w.encoders().reshape(2, 6).copy()
    apple = S.locate_object(proc, obs, 'apple', plane=plane, max_height=.12); bowl = S.locate_object(proc, obs, 'bowl', plane=plane, max_height=.12)
    if apple is None or bowl is None:
        raise RuntimeError(f'perception failed apple={apple is not None} bowl={bowl is not None}')
    apple_c = P.fit_sphere(apple['points'], .034)
    bowl_xy, bowl_r = P.fit_upright_cylinder(bowl['points'][bowl['points'][:, 2] > bowl['top'] - .015])
    return np.asarray(apple_c, float), np.asarray(bowl_xy, float), float(bowl['top']), bowl_r


def grasp_path(c, gz, R):
    t = np.r_[c[:2], c[2] + gz]
    return t, [(t + .06 * UP, R), (t, R)], [(np.r_[t[:2], max(CARRY_Z, t[2] + .06)], R)]


def release_path(bxy, btop, rz, R, dz):
    # l10: apple radius ~.035 and hand at the apple centre, so the apple bottom = hand z - .035. Release with the bottom .005 above the observed rim (dz adds
    # height if IK fails); the approach first reaches the point vertically above the release with the bottom >= .015 over the rim, then moves straight down.
    # l8 DEV14 released at rim+2.5 cm hand height (apple bottom ~1 cm BELOW the rim) and pushed the bowl 11.5 mm.
    d = np.r_[bxy, btop + .040 + dz]
    above = np.r_[bxy, max(btop + .050, d[2] + .01)]
    return d, [(above, R), (d, R)], [(d + .04 * UP, R)]


def do(io, arm, path, dur=1.2):
    for p, R in path:
        io.move(arm, p, R, dur)


def run(io):
    ev = io.demo['events']
    close = next(e for e in ev if e['kind'] == 'close')
    opened = next(e for e in ev if e['kind'] == 'open' and e['time'] > close['time'])
    gz = close['position'][2] - close['objects']['apple'][2]  # demo prior: hand height relative to the apple centre at grasp
    rz = opened['position'][2] - opened['objects']['bowl'][2]  # demo prior: hand height above the bowl at release
    apple, bxy, btop, br = perceive(io)
    io.log('perceived', apple=apple, bowl=bxy, bowl_top=btop, bowl_r=br)
    global CARRY_Z
    CARRY_Z = btop + .05  # l10: apple bottom 1.5 cm over the rim for the whole carry; the release pre-point is at the same height
    arms = sorted(('left', 'right'), key=lambda a: np.linalg.norm(io.tcp(a)[:2, 3] - apple[:2]))
    plan = None
    for arm in arms:  # single-arm plan
        for yaw, gp in [(y, p) for p in PITCH for y in GYAWS]:
            R = top_down(yaw, gp); t, g1, g2 = grasp_path(apple, gz, R)
            q = feasible(io, arm, g1 + g2)
            if q is None: continue
            for ryaw, rp in [(y2, p2) for p2 in PITCH for y2 in [yaw] + [y for y in YAWS if y != yaw]] + [(y2 + np.pi, p2) for p2 in PITCH[1:] for y2 in YAWS]:
                R2 = top_down(ryaw, rp)
                for dz in (0., .02, .04):
                    d, r1, r2 = release_path(bxy, btop, rz, R2, dz)
                    if feasible(io, arm, g1 + g2 + r1 + r2) is not None:
                        plan = ('single', arm, R, R2, t, d); break
                if plan: break
            if plan: break
        if plan: break
    if plan:
        _, arm, R, R2, t, d = plan
        io.log('plan', kind='single', arm=arm, grasp=t, drop=d)
        io.grip(arm, 1., .3); do(io, arm, grasp_path(apple, gz, R)[1]); io.grip(arm, 0., .6); io.log('stage_grasped')
        do(io, arm, grasp_path(apple, gz, R)[2]); do(io, arm, [(np.r_[d[:2], max(CARRY_Z, d[2] + .01)], R2), (d, R2)], 1.5)
        io.log('stage_pre_release'); io.grip(arm, 1., .6); io.move(arm, d + .04 * UP, R2, .8)
    else:
        relay(io, apple, bxy, btop, gz, rz, arms)
    for k, a in enumerate(('left', 'right')):
        RC.enc = io._w.encoders().reshape(2, 6).copy()
        if RC.joint_path_clear(RC.enc[k], np.asarray(io.home_q[k], float), k):
            io.home(a, 1.5)
        else:  # l6 DEV2 hit forearm-wrist self-collision here, after a good release: stay at the retreat pose instead
            io.log('home_skipped', arm=a)
    io.wait(2.0)
    io.observe()


def relay(io, apple, bxy, btop, gz, rz, arms):
    '''Two-arm relay: A (reaches the apple) sets it down at a handover point H on the table, B (reaches H and the bowl) re-grasps after a fresh observation.'''
    table_z = apple[2]  # handover apple centre height = the observed resting height
    mid = (apple[:2] + bxy) / 2
    cands = sorted([mid + np.array([dx, dy]) for dx in np.arange(-.09, .091, .03) for dy in np.arange(-.12, .121, .03)], key=lambda h: np.linalg.norm(h - mid))
    cands = [h for h in cands if np.linalg.norm(h - bxy) > br_clear(bxy) and np.linalg.norm(h - apple[:2]) > .05]
    for a, b in ((arms[0], arms[1]), (arms[1], arms[0])):
        for yaw, gp in [(y, p) for p in PITCH[::2] for y in YAWS]:
            R = top_down(yaw, gp); t, g1, g2 = grasp_path(apple, gz, R)
            if feasible(io, a, g1 + g2) is None: continue
            for h in cands:
                hc = np.r_[h, table_z]; th, h1, h2 = grasp_path(hc, gz, R)
                put = [(th + .15 * UP, R), (th + .01 * UP, R)]
                if feasible(io, a, g1 + g2 + put) is None: continue
                for yb, bp in [(y, p) for p in PITCH[::2] for y in YAWS]:
                    Rb = top_down(yb, bp); _, b1, b2 = grasp_path(hc, gz, Rb)
                    for dz in (0., .02, .04):
                        d, r1, r2 = release_path(bxy, btop, rz, Rb, dz)
                        if feasible(io, b, b1 + b2 + r1 + r2) is None: continue
                        io.log('plan', kind='relay', arm_a=a, arm_b=b, handover=hc, drop=d)
                        io.grip(a, 1., .3); do(io, a, g1); io.grip(a, 0., .6); io.log('stage_grasped', arm=a)
                        do(io, a, g2 + put, 1.4); io.grip(a, 1., .6); io.move(a, th + .15 * UP, R, .8)
                        ka = ('left', 'right').index(a); RC.enc = io._w.encoders().reshape(2, 6).copy()
                        if RC.joint_path_clear(RC.enc[ka], np.asarray(io.home_q[ka], float), ka):
                            io.home(a, 1.5)
                        else:
                            raise RuntimeError('relay arm cannot park clear of self-collision')
                        io.wait(.5)
                        apple2, bxy2, btop2, _ = perceive(io); io.log('reperceived', apple=apple2, bowl=bxy2)
                        _, b1, b2 = grasp_path(apple2, gz, Rb); d, r1, r2 = release_path(bxy2, btop2, rz, Rb, dz)
                        if feasible(io, b, b1 + b2 + r1 + r2) is None:  # l9: re-check arm B (IK + RobotCollision) on the re-perceived positions
                            raise RuntimeError('relay arm-B path infeasible after re-perception')
                        io.grip(b, 1., .3); do(io, b, b1); io.grip(b, 0., .6); io.log('stage_grasped', arm=b)
                        do(io, b, b2 + r1, 1.4); io.log('stage_pre_release'); io.grip(b, 1., .6); io.move(b, d + .04 * UP, Rb, .8)
                        return
    raise RuntimeError('no feasible single-arm or relay plan')


def br_clear(bxy):
    return .13  # handover point at least 13 cm from the bowl axis (bowl rim radius ~7 cm + apple + clearance)
