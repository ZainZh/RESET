# Zero-shot controller (Code-as-Policies baseline, no execution feedback).
# Observations only: io.observe() RGB-D + K + camera pose, io.tcp() (encoder FK), io.demo priors, io.instruction.
import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as _Rot
from rsi import perception as P
from rsi import perception_geo as G

ARMS = ('left', 'right')


class Budget(RuntimeError):
    pass


def Rz(a):
    return _Rot.from_euler('z', float(a)).as_matrix()


def rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a, period):
    return (a + period / 2.) % period - period / 2.


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0


def grip_yaw(R):
    """Yaw of the most horizontal TCP axis (defined mod 90 deg for a top-down grasp)."""
    R = np.asarray(R, float)
    i = int(np.argmin([abs(R[2, j]) for j in range(3)]))
    return math.atan2(R[1, i], R[0, i])


# ---------------------------------------------------------------- robot wrapper with a sim-time budget
class Bot:
    def __init__(self, io, cap, reserve=1.8):
        self.io, self.cap, self.reserve, self.t = io, float(cap), float(reserve), 0.

    def remaining(self):
        return self.cap - self.reserve - self.t

    def _need(self, d):
        if self.t + d > self.cap - self.reserve:
            raise Budget('time budget')

    def pos(self, arm):
        return self.io.tcp(arm)[:3, 3].copy()

    def rot(self, arm):
        return self.io.tcp(arm)[:3, :3].copy()

    def dur(self, arm, p, v=.22, lo=.8, hi=3.):
        return float(np.clip(np.linalg.norm(np.asarray(p, float) - self.pos(arm)) / v + .5, lo, hi))

    def move(self, arm, p, R, d=None, v=.22):
        p = np.asarray(p, float); R = np.asarray(R, float)
        d = self.dur(arm, p, v) if d is None else float(d)
        self._need(d)
        try:
            err = self.io.move(arm, p, R, d)
        except Exception as e:
            if type(e).__name__ == 'Limit':
                raise
            self.t += .3 * d
            self.io.log('move_failed', arm=arm, p=p, err=str(e)[:160])
            return False
        self.t += d
        return err < .03

    def grip(self, arm, opening, d=.5):
        self._need(d); self.io.grip(arm, float(opening), d); self.t += d

    def wait(self, d):
        self._need(d); self.io.wait(d); self.t += d

    def home(self, arm, d=2.):
        self._need(d); self.io.home(arm, d); self.t += d

    def lift(self, arm, dz=.08, d=.9):
        p = self.pos(arm); p[2] += dz
        return self.move(arm, p, self.rot(arm), d)

    def pick(self, arm, candidates, approach=.08, lift=.10):
        """candidates: [(p, R)]; tries each until the pre-grasp is reachable. Returns (p, R) used or None."""
        for p, R in candidates:
            p = np.asarray(p, float)
            above = p + [0, 0, approach]
            if not self.move(arm, above, R):
                continue
            self.grip(arm, 1., .3)
            if not self.move(arm, p, R, 1.1):
                self.lift(arm, approach)
                continue
            self.grip(arm, 0., .6)
            self.wait(.25)
            self.move(arm, p + [0, 0, lift], R, 1.0)
            return p, R
        return None

    def place(self, arm, p, R, carry_z, settle=.3):
        p = np.asarray(p, float)
        cur = self.pos(arm)
        if cur[2] < carry_z - .01:
            self.move(arm, [cur[0], cur[1], carry_z], R, .9)
        self.move(arm, [p[0], p[1], carry_z], R)
        self.move(arm, p, R, 1.1)
        self.grip(arm, 1., .5)
        self.wait(settle)
        self.move(arm, [p[0], p[1], max(carry_z, p[2] + .06)], R, 1.0)

    def finish(self, keep_closed=()):
        """Release everything and back off while time remains (scorer needs objects released)."""
        for arm in ARMS:
            if arm in keep_closed:
                continue
            try:
                if self.cap - self.t > 1.2:
                    self.io.grip(arm, 1., .4); self.t += .4
                if self.cap - self.t > 1.6:
                    p = self.pos(arm); p[2] += .06
                    try:
                        self.io.move(arm, p, self.rot(arm), .7)
                    except Exception as e:
                        if type(e).__name__ == 'Limit':
                            raise
                    self.t += .7
            except Exception as e:
                if type(e).__name__ == 'Limit':
                    return


# ---------------------------------------------------------------- perception
class Scene:
    def __init__(self, io, tcp_radius=.06):
        self.obs = io.observe(keep=True)
        self.plane = G.fit_plane(self.obs)
        self.p, self.ys, self.xs = G.safe_points(self.obs)
        self.h = G.height(self.p, self.plane)
        self.hsv = cv2.cvtColor(self.obs['rgb'], cv2.COLOR_RGB2HSV)
        self.tcp = [io.tcp(a)[:3, 3] for a in ARMS]
        self.tcp_radius = tcp_radius

    def colour(self, names):
        if isinstance(names, str):
            names = [names]
        m = np.zeros(self.hsv.shape[:2], bool)
        for n in names:
            for lo, hi in P.HSV[n]:
                m |= cv2.inRange(self.hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0
        return m[self.ys, self.xs]

    def select(self, colour=None, center=None, radius=None, hlo=.004, hhi=.3, exclude=()):
        k = (self.h > hlo) & (self.h < hhi)
        if colour is not None:
            k &= self.colour(colour)
        if center is not None:
            k &= np.linalg.norm(self.p[:, :2] - np.asarray(center, float)[:2], axis=1) < radius
        for xy, r in exclude:
            k &= np.linalg.norm(self.p[:, :2] - np.asarray(xy, float)[:2], axis=1) > r
        for t in self.tcp:
            k &= ~((np.linalg.norm(self.p[:, :2] - t[:2], axis=1) < self.tcp_radius) & (self.p[:, 2] > t[2] - .03))
        return self.p[k], self.h[k]


def blob(q, hq, band=.008):
    top = float(np.percentile(hq, 95))
    t = hq > top - band
    if t.sum() < 5:
        t = np.ones(len(q), bool)
    xy = q[t, :2]
    (cx, cy), (w, l), ang = cv2.minAreaRect((xy * 1000.).astype(np.float32))
    long_yaw = math.radians(ang if w >= l else ang + 90.)
    return {'xy': np.array([cx, cy]) / 1000., 'mean_xy': xy.mean(0), 'top': top, 'n': int(len(q)),
            'yaw': math.radians(ang), 'long_yaw': long_yaw, 'wl': np.array([max(w, l), min(w, l)]) / 1000.,
            'ext': q[:, :2].max(0) - q[:, :2].min(0), 'points': q, 'h': hq}


def clusters(p, h, min_n=25, band=.008):
    if len(p) < min_n:
        return []
    img = P._raster(p)
    n, lab = cv2.connectedComponents(img)
    cell = np.clip(P._cells(p), 0, np.array(img.shape[::-1]) - 1)
    ids = lab[cell[:, 1], cell[:, 0]]
    out = []
    for i in range(1, n):
        k = ids == i
        if k.sum() >= min_n:
            out.append(blob(p[k], h[k], band))
    return sorted(out, key=lambda b: -b['n'])


def split2(p, h, seeds, iters=8):
    """Split one merged point set into two around xy seeds (simple 2-means)."""
    s = np.array([np.asarray(x, float)[:2] for x in seeds])
    for _ in range(iters):
        lab = np.argmin(np.linalg.norm(p[:, None, :2] - s[None], axis=2), 1)
        for j in range(2):
            if np.any(lab == j):
                s[j] = p[lab == j, :2].mean(0)
    return [blob(p[lab == j], h[lab == j]) if np.sum(lab == j) > 10 else None for j in range(2)]


def round_center(q, hq, band=.015):
    """Axis xy and radius of an upright round object from its rim band."""
    top = float(np.percentile(hq, 95))
    rim = q[hq > top - band]
    if len(rim) < 20:
        return q[:, :2].mean(0), None, top
    try:
        xy, r = P.fit_upright_cylinder(rim)
        if not (.015 < r < .16) or np.linalg.norm(xy - rim[:, :2].mean(0)) > .08:
            raise ValueError
        return np.asarray(xy, float), float(r), top
    except Exception:
        return rim[:, :2].mean(0), None, top


# ---------------------------------------------------------------- demonstration priors
def demo_pairs(demo):
    """Pair every gripper close with the next open of the same arm; tag the nearest demo object."""
    ev = demo['events']
    out = []
    for i, e in enumerate(ev):
        if e['kind'] != 'close' or not e.get('objects'):
            continue
        op = next((f for f in ev[i + 1:] if f['arm'] == e['arm'] and f['kind'] == 'open'), None)
        objs = e['objects']
        pos = np.asarray(e['position'], float)
        name = min(objs, key=lambda n: np.linalg.norm(np.asarray(objs[n], float) - pos))
        moved = float(np.linalg.norm(np.asarray(op['objects'][name]) - np.asarray(objs[name]))) if op else 0.
        out.append({'arm': e['arm'], 'close': e, 'open': op, 'object': name, 'moved': moved,
                    'dist': float(np.linalg.norm(np.asarray(objs[name], float) - pos))})
    return out


def pair_for(pairs, name, xy=None, names=None):
    c = [q for q in pairs if q['object'] == name]
    if c:
        return max(c, key=lambda q: q['moved'])
    pool = [q for q in pairs if names is None or q['object'] in names] or pairs
    if not pool:
        return None
    if xy is None:
        return pool[0]
    return min(pool, key=lambda q: np.linalg.norm(np.asarray(q['close']['objects'][q['object']])[:2] - np.asarray(xy)[:2]))


def default_pair(io, xy):
    """No demo grasp available: nearest arm (by demo home TCP), current orientation, grasp at table+2cm."""
    home = io.demo['home']
    arm = min(ARMS, key=lambda a: np.linalg.norm(home[a][:3, 3][:2] - np.asarray(xy)[:2]))
    T = io.tcp(arm)
    e = {'arm': arm, 'position': np.r_[np.asarray(xy)[:2], T[2, 3]], 'rotation': T[:3, :3], 'objects': {'x': np.r_[np.asarray(xy)[:2], 0]}}
    return {'arm': arm, 'close': e, 'open': None, 'object': 'x', 'moved': 0., 'dist': 0.}


def grasp_candidates(pair, obj_xy_now, obj_demo_xy=None, dyaw=0., box_sym=None, use_offset=None):
    """Grasp poses from the demo grasp, translated to the perceived object and rotated by the object yaw change.
    box_sym: rotational symmetry period (rad) to also try equivalent yaws."""
    c = pair['close']
    R0 = np.asarray(c['rotation'], float)
    p0 = np.asarray(c['position'], float)
    od = np.asarray(obj_demo_xy if obj_demo_xy is not None else c['objects'][pair['object']], float)[:2]
    o = p0[:2] - od
    if use_offset is None:
        use_offset = np.linalg.norm(o) > .02
    if not use_offset:
        o = np.zeros(2)
    yaws = [dyaw]
    if box_sym:
        yaws += [dyaw - box_sym if dyaw > 0 else dyaw + box_sym]
    yaws += [0.]
    out, seen = [], []
    for a in yaws:
        if any(abs(a - s) < 1e-3 for s in seen):
            continue
        seen.append(a)
        out.append((np.r_[np.asarray(obj_xy_now, float)[:2] + rot2(a) @ o, p0[2]], Rz(a) @ R0))
    return out


# ================================================================ task: sweep_objects (v8)
# Observation-only. Demo priors used: initial demo object positions (dustpan, broom, cube), the single right-arm
# brush grasp (pose + broom position at grasp). Pan geometry assumption (task asset description): mouth line ~12.5 cm
# from the pan origin, floor centre ~8 cm. Pan frame is perceived from the orange mouth lip (fallback: demo direction
# pan -> demo cube). The brush is perceived after grasp (lifted) to get its actual axis, tip distance and droop.
CUBE_HALF = .016
PAN_LIFT = .008
LEFT_Z = .006


def perp(v):
    return np.array([-v[1], v[0]])


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def find_cube(sc, near, radius=.14, brush_seg=None):
    q, h = sc.select('orange', near, radius, hlo=.012, hhi=.045)
    if brush_seg is not None and len(q):
        k = seg_dist(q[:, :2], *brush_seg) > .022
        q, h = q[k], h[k]
    cl = [b for b in clusters(q, h, 20) if b['wl'][0] < .06]
    if not cl:
        return None
    return min(cl, key=lambda b: np.linalg.norm(b['xy'] - np.asarray(near)[:2]))


def find_brush(sc, demo_xy):
    q, h = sc.select('orange', demo_xy, .28, hlo=.004, hhi=.08)
    cl = [b for b in clusters(q, h, 60, band=.03) if b['wl'][0] > .15]
    if not cl:
        return None
    b = cl[0]
    (cx, cy), (w, l), ang = cv2.minAreaRect((b['points'][:, :2] * 1000.).astype(np.float32))
    yaw = math.radians(ang if w >= l else ang + 90.)
    return {'xy': np.array([cx, cy]) / 1000., 'yaw': yaw, 'len': max(w, l) / 1000., 'points': b['points'], 'h': b['h']}


def find_lip(sc, guess_xy):
    """Orange, nearly flat, elongated strip near the expected mouth."""
    q, h = sc.select('orange', guess_xy, .11, hlo=-.003, hhi=.012)
    best = None
    for b in clusters(q, h, 15, band=.02):
        if .08 < b['wl'][0] < .22 and b['wl'][1] < .035:
            if best is None or b['n'] > best['n']:
                best = b
    return best


def pan_profile(sc, pf):
    """Max/median height along the pan centreline (|local x| < 3 cm), 1 cm bins from local y -0.17 to +0.02."""
    lx = (sc.p[:, :2] - pf.O) @ pf.ex
    ly = (sc.p[:, :2] - pf.O) @ pf.ey
    out = []
    for y in np.arange(-.17, .02, .01):
        k = (np.abs(lx) < .03) & (ly >= y) & (ly < y + .01)
        out.append(round(float(np.median(sc.h[k])), 4) if k.sum() > 5 else None)
    return out


HEAD_DROP = .003


def measure_tip(sc, tcp, a, Lguess):
    """In-hand brush tip along xy direction a from the TCP: (tip distance, TCP z - tip top z); None if not visible."""
    k = sc.colour('orange') & (sc.h > .06) & (np.linalg.norm(sc.p[:, :2] - tcp[:2], axis=1) < .3)
    q = sc.p[k]
    L, drop = Lguess, None
    if len(q) > 30:
        pr = (q[:, :2] - tcp[:2]) @ a
        lat = np.abs((q[:, :2] - tcp[:2]) @ perp(a))
        k2 = (pr > .03) & (lat < .035)
        if k2.sum() > 20:
            L1 = float(np.percentile(pr[k2], 99))
            if .09 < L1 < .2 and abs(L1 - Lguess) < .03:
                L = L1
            tip = q[k2][pr[k2] > L - .04]
            if len(tip) > 5:
                drop = float(tcp[2] - np.percentile(tip[:, 2], 90))
    return L, drop


def measure_axis(sc, tcp, a):
    """Actual in-hand xy direction TCP -> head (the brush can yaw-slip in the grip under push load); None if unclear."""
    k = sc.colour('orange') & (sc.h > .06) & (np.linalg.norm(sc.p[:, :2] - tcp[:2], axis=1) < .3)
    q = sc.p[k][:, :2] - tcp[:2]
    if len(q) < 40:
        return None
    pr = q @ a
    lat = np.abs(q @ perp(a))
    k2 = (pr > .04) & (lat < .07)
    if k2.sum() < 30:
        return None
    w = q[k2]
    c = w.mean(0)
    _, _, vt = np.linalg.svd(w - c)
    ax = vt[0] if vt[0] @ a > 0 else -vt[0]
    # axis through the TCP: use the direction from TCP to the far-end centroid, cross-checked with PCA
    far = w[(w @ ax) > np.percentile(w @ ax, 85)].mean(0)
    d = unit(far)
    if d @ a < math.cos(math.radians(35)) or d @ ax < math.cos(math.radians(15)):
        return None
    return d


def zpush_from(table, thick, drop):
    return float(np.clip(table + .006 + thick + drop, table + .012, table + .08))


class PanFrame:
    def __init__(self, O, u_out):
        self.O, self.uo = np.asarray(O, float)[:2], unit(u_out)
        self.ey = -self.uo
        self.ex = np.array([self.ey[1], -self.ey[0]])

    def local(self, xy):
        d = np.asarray(xy, float)[:2] - self.O
        return np.array([d @ self.ex, d @ self.ey])

    def world(self, lxy):
        return self.O + self.ex * lxy[0] + self.ey * lxy[1]


class Brush:
    """Brush held by the right gripper. a = unit xy direction TCP -> tip, L = tip distance, R = TCP rotation."""

    def __init__(self, bot, R, a, L, z_push, z_safe):
        self.bot, self.R, self.a, self.L, self.zp, self.zs = bot, R, unit(a), float(L), float(z_push), float(z_safe)

    def rot_to(self, a_new):
        dyaw = math.atan2(self.a[0] * a_new[1] - self.a[1] * a_new[0], self.a @ a_new)
        return Rz(dyaw) @ self.R

    def _exec(self, a_new, c0, c1, s, v, io, dz=0.):
        R = self.rot_to(a_new)
        t0 = c0 - a_new * s
        t1 = c1 - a_new * s
        if not self.bot.move('right', np.r_[t0, self.zs], R):
            io.log('push_pre_fail', a=a_new)
            return False
        self.a, self.R = a_new, R
        self.bot.move('right', np.r_[t0, self.zp + dz], R, 1.0)
        length = float(np.linalg.norm(c1 - c0))
        self.bot.move('right', np.r_[t1, self.zp + dz], R, max(1.2, length / .05), v=.05)
        io.log('push_end_err', err=float(np.linalg.norm(self.bot.pos('right') - np.r_[t1, self.zp + dz])))
        back = t1 - v * .025
        self.bot.move('right', np.r_[back, self.zp + dz + .005], R, .6)
        return True

    def tip_push(self, cube, goal, io, dz=0., gap=.025, ang=0.):
        """Head-end push along v; ang tilts the brush axis away from v (different arm configuration, same contact)."""
        v = unit(np.asarray(goal) - np.asarray(cube))
        c0 = np.asarray(cube) - v * (CUBE_HALF + gap)
        c1 = np.asarray(goal) - v * (CUBE_HALF + .002)
        io.log('tip_push', cube=cube, goal=goal, dz=dz, ang=ang)
        return self._exec(rot2(ang) @ v, c0, c1, self.L, v, io, dz)

    def side_push(self, cube, goal, prefer, io):
        """Brush perpendicular to v, contact 5 cm in from the tip; brush body on the side of `prefer` direction."""
        v = unit(np.asarray(goal) - np.asarray(cube))
        cands = sorted([perp(v), -perp(v)], key=lambda c: -(c @ prefer))
        c0 = np.asarray(cube) - v * (CUBE_HALF + .015 + .025)
        c1 = np.asarray(goal) - v * (CUBE_HALF + .015)
        io.log('side_push', cube=cube, goal=goal)
        for a_new in cands:
            if self._exec(a_new, c0, c1, self.L - .025, v, io):
                return True
        return False

    def clear(self, away):
        """Lift the brush well above the cube band before perceiving; fall back to a pure lift."""
        p = self.bot.pos('right')
        p[2] = self.zs + .06
        if not self.bot.move('right', p, self.R, .8):
            self.bot.io.log('clear_lift_fail')
        q = p.copy(); q[:2] += np.asarray(away) * .05; q[2] = self.zs + .10
        self.bot.move('right', q, self.R, .8)

    def seg(self):
        t = self.bot.pos('right')[:2]
        return (t - self.a * .03, t + self.a * (self.L + .03))


def run(io):
    bot = Bot(io, 40., reserve=2.0)
    d = io.demo
    o0 = d['objects0']
    BR = 'broom' if 'broom' in o0 else 'brush'
    pairs = demo_pairs(d)
    rp = [q for q in pairs if q['arm'] == 'right']
    rpr = next((q for q in rp if q['object'] in ('brush', 'broom')), rp[0] if rp else None)
    pan0 = np.asarray(o0['dustpan'], float)
    cube_demo = np.asarray(o0['orange_cube'], float)
    try:
        # 1. keep the dustpan: close the left gripper immediately (it starts on the handle)
        sc = Scene(io)
        table = float(sc.plane[0][2])
        lp, lR = bot.pos('left'), bot.rot('left')
        io.log('left_tcp0', p=lp, table=table)
        # the left TCP starts above the pan's back wall with the gripper open: descend to straddle it, then close
        bot.move('left', np.r_[lp[:2], table + LEFT_Z], lR, .8)
        bot.grip('left', 0., .6)
        bot.wait(.2)
        o = io.observe(keep=False)
        io.log('obs_keys', keys=sorted(o.keys()), grip={k: o[k] for k in o if 'grip' in k or 'open' in k or 'finger' in k})
        bot.move('left', np.r_[lp[:2], table + LEFT_Z + PAN_LIFT], lR, .6)
        bot.wait(.2)
        sc = Scene(io)
        # 2. pan frame from the orange lip
        u0 = unit(cube_demo[:2] - pan0[:2])
        lip = find_lip(sc, pan0[:2] + u0 * .125)
        uo, O = u0, pan0[:2]
        if lip is not None:
            ax = np.array([math.cos(lip['long_yaw']), math.sin(lip['long_yaw'])])
            un = perp(ax)
            un = un if un @ u0 > 0 else -un
            On = lip['xy'] - un * .125
            ok = abs(math.degrees(math.acos(np.clip(un @ u0, -1, 1)))) < 20 and np.linalg.norm(On - pan0[:2]) < .05
            io.log('lip', xy=lip['xy'], wl=lip['wl'], u=un, O=On, ok=ok)
            if ok:
                uo, O = un, On
        pf = PanFrame(O, uo)
        target = pf.world([0., -.068])
        io.log('pan', O=O, u_out=uo, target=target, table=table, lip_h=float(np.median(lip['h'])) if lip is not None else None,
               profile=pan_profile(sc, pf), left=bot.pos('left'))

        # 3. brush grasp (demo grasp transferred to the perceived brush)
        def acquire(sc, near):
            br = find_brush(sc, near)
            if rpr is None:
                raise RuntimeError('no right-arm demo grasp')
            c = rpr['close']
            R0, p0 = np.asarray(c['rotation'], float), np.asarray(c['position'], float)
            demo_brush = np.asarray(c['objects'].get(BR, o0[BR]), float)[:2]
            if br is None:
                br = {'xy': demo_brush, 'yaw': grip_yaw(R0), 'len': .34, 'points': None}
            dyaw = wrap(br['yaw'] - grip_yaw(R0), math.pi / 2)
            if br['points'] is not None:
                ax = np.array([math.cos(br['yaw']), math.sin(br['yaw'])])
                s_ = (br['points'][:, :2] - br['xy']) @ ax
                prof = [round(float(np.percentile(br['h'][(s_ >= a) & (s_ < a + .04)], 95)), 4) if np.sum((s_ >= a) & (s_ < a + .04)) > 5 else None
                        for a in np.arange(-.2, .2, .04)]
                io.log('brush', xy=br['xy'], yaw=br['yaw'], len=br['len'], dyaw=dyaw, profile=prof)
            cand = grasp_candidates(rpr, br['xy'], demo_brush, dyaw=dyaw, use_offset=True)
            got = None
            for p, R in cand:
                if bot.move('right', p + [0, 0, .07], R):
                    bot.grip('right', 1., .2)
                    bot.move('right', p, R, 1.1)
                    bot.grip('right', 0., .7)
                    bot.wait(.3)
                    got = (p, R)
                    break
            if got is None:
                raise RuntimeError('brush unreachable')
            gp, gR = got
            # 4. lift and measure the held brush
            z_cal = gp[2] + .15
            bot.move('right', np.r_[gp[:2], z_cal], gR, 1.2)
            tcp = bot.pos('right')
            sc = Scene(io, tcp_radius=0.)
            # push with the HEAD end (the end nearer the grasp: thick, heavier, hangs below the grip -> stable when pushing);
            # the thin far end tilts up in the grip and pivots further up under axial load (v4 lesson)
            ax = np.array([math.cos(br['yaw']), math.sin(br['yaw'])])
            a = -ax if ax @ (br['xy'] - gp[:2]) > 0 else ax
            L0 = br['len'] / 2 - abs((br['xy'] - gp[:2]) @ ax) - .005
            thick = .033
            if br['points'] is not None:
                pr0 = (br['points'][:, :2] - gp[:2]) @ a
                kt = pr0 > pr0.max() - .05
                if kt.sum() > 10:
                    thick = float(np.clip(np.percentile(br['h'][kt], 95), .015, .045))
            L, drop = measure_tip(sc, tcp, a, L0)
            if drop is None:
                drop = HEAD_DROP
            z_push = zpush_from(table, thick, drop)
            io.log('brush_cal', tcp=tcp, a=a, L=L, L0=L0, thick=thick, drop=drop, z_push=z_push)
            brush = Brush(bot, gR, a, L, z_push, z_push + .06)
            brush.thick, brush.table, brush.zp0 = thick, table, z_push
            return brush

        brush = acquire(sc, o0[BR])

        # 5. push the cube: side push to the staging point if laterally off, then tip push into the pan
        cube_guess = cube_demo[:2]
        stuck = 0
        regrasps = 0
        last = None
        for it in range(6):
            if it:
                brush.clear(-pf.ey)
            sc = Scene(io, tcp_radius=.05)
            if it:
                tcp_now = bot.pos('right')
                held = sc.colour('orange') & (sc.h > .06) & (np.linalg.norm(sc.p[:, :2] - tcp_now[:2], axis=1) < .3)
                if held.sum() < 30 and regrasps < 1:
                    # brush no longer visible in the hand: it was dropped -> find it on the table and grasp again
                    regrasps += 1
                    io.log('brush_lost', n=int(held.sum()))
                    bot.grip('right', 1., .3)
                    p_up = bot.pos('right'); p_up[2] = table + .2
                    bot.move('right', p_up, bot.rot('right'), .8)
                    sc = Scene(io, tcp_radius=0.)
                    brush = acquire(sc, tcp_now[:2])
                    brush.clear(-pf.ey)
                    sc = Scene(io, tcp_radius=.05)
                a2 = measure_axis(sc, bot.pos('right'), brush.a)
                if a2 is not None:
                    da = math.degrees(math.atan2(brush.a[0] * a2[1] - brush.a[1] * a2[0], brush.a @ a2))
                    io.log('axis_recal', da=da)
                    if abs(da) > 5:   # only a clear slip; small differences are perception noise
                        brush.a = a2
                L2, d2 = measure_tip(sc, bot.pos('right'), brush.a, brush.L)
                if d2 is not None:
                    zp2 = zpush_from(table, brush.thick, d2)
                    if abs(zp2 - brush.zp0) < .015:
                        brush.L, brush.zp = L2, zp2
                io.log('recal', L=L2, drop=d2, zp=brush.zp)
            cb = find_cube(sc, cube_guess, .15 if it == 0 else .12, brush.seg())
            if cb is None:
                cb = find_cube(sc, cube_guess, .12)
                io.log('cube_fallback', found=cb is not None)
            if cb is None:
                io.log('cube_lost', it=it)
                break
            cube = cb['xy']; cube_guess = cube
            lc = pf.local(cube)
            io.log('cube', it=it, xy=cube, local=lc)
            if lc[1] > -.094 and abs(lc[0]) < .05:
                break
            if abs(lc[0]) > .022 and lc[1] < -.14:
                stage = pf.world([0., min(lc[1], -.175)])
                # close to the mouth a perpendicular brush would sweep the pan corner: push laterally with the head
                if not (lc[1] > -.185 and brush.tip_push(cube, pf.world([0., lc[1] - .005]), io)):
                    brush.side_push(cube, stage, pf.ey, io)
                cube_guess = stage
                continue
            if last is not None and lc[1] > -.165 and abs(lc[1] - last[1]) < .015:
                stuck += 1
            last = lc
            # cube already at/over the mouth lip: start close behind it with the head raised 8 mm so it clears the lip
            near = lc[1] > -.135
            ang = [0., .35, -.35][stuck % 3]
            if not brush.tip_push(cube, target, io, dz=.008 if (near or stuck) else 0., gap=.012 if near else .025, ang=ang):
                brush.tip_push(cube, target, io, dz=.008 if (near or stuck) else 0., gap=.012 if near else .025)
            cube_guess = target
        # leave the brush up and clear of the cube; both grippers stay closed
        p = bot.pos('right'); p[2] = brush.zs
        bot.move('right', p, brush.R, .8)
    except Budget:
        io.log('budget')
