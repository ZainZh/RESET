"""open_drawer_and_place_object lift2_sd7 = sd6 + knob grasp height from the observed knob top instead of table mode + .066; sd6 = sd5 + block grasp yaw/tilt ranked by finger clearance to the perceived (opened) cabinet points; sd5 = sd4 with the re-grasp pull limited to the missing distance (target opening 13 cm; sd4 re-pulled 17 cm -> 19 cm opening put the drawer side under the block approach); sd4 = sd3 + robot-only self-collision check on every planned waypoint (robot MJCF only) + re-grasp-and-pull when the observed opening is short; sd3 = tool-axis approach/lift, tilt capped by finger-table clearance, longer pull, drop at the centre of the observed opening (LIFT2, smooth_table_v1): xTrainer s1 skill with a LIFT2-specific plan.
Shared with s1: SAM3-grounded cabinet front face (min-x extent, y = extent centre, knob height = table + .066 demo prior),
grasp the knob, pull 13 cm along -x, release; SAM3 block search over learned prompts with cabinet-footprint rejection;
pick the block and drop it at face + (.05..07) inside the opened drawer, release, retreat, dwell.
LIFT2-specific plan:
- table height = lowest well-populated depth mode (on this setup the cabinet top fills most of the head view, so the
  shared modal height P.table_height returns the cabinet top); plane refit around it;
- knob y from the SAM3 'knob' instance on the face when present, else the extent centre;
- knob grasp orientation candidates: the s1 rotation first (R[:,2] = approach), then approach toward +x pitched down
  0..45 deg with the closing axis horizontal or vertical; arm chosen by an IK chain check of the whole pull from the
  current encoders (no motion before a full plan is feasible);
- block grasp: top-down canonical frame, closing yaw from the block's top face, tilts about the closing axis, IK-checked;
- encoder-FK sag correction on the knob and block descend legs.
Demo contact priors: left close rotation (knob grasp, s1), knob height above the table (.066, s1), right close yaw (orders
block yaw candidates). Runtime inputs: head RGB-D, calibration, encoders, robot model. No simulator object state."""
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
from rsi import perception as P
from rsi import perception_sam3 as S
from rsi import perception_geo as G
from twin.practice.motion import KinematicsError, waypoint
import mujoco
from rsi.lift2_suite.common.embodiment import lift2 as ROBOT  # robot model constants only (MJCF path, joint names)

ARMS = ('left', 'right')
UP = np.array([0, 0, 1.])
PULLS = (.17, .15, .13)
MIN_OPEN = .08
TARGET_OPEN = .13
CLEAR = .025
KNOB_DZ = .066
KNOB_TOP_DZ = .008   # grasp this far below the observed knob top (DEV0/1 working height .103 vs observed top ~.111)
GRASP_DZ = .021
CAB_PROMPTS = ('wooden cabinet', 'wooden box', 'cabinet')
PROMPTS = ('orange cube', 'small cube', 'orange block', 'toy block')
CAB_MARGIN = .02
TILTS = np.radians([0., 20., -20., 35., -35., 42., -42.])
TILT_DZ = {42: .028}  # at 50 deg the fingers touch the table at TCP +36 mm (DEV0 diag): 42 deg keeps the pads on the block top


# ---------- perception ----------
def table_plane(obs):
    p, _, _ = G.safe_points(obs)
    p = p[(p[:, 1] < -.25) & (p[:, 1] > -.85) & (np.abs(p[:, 0]) < .4)]
    hist, edges = np.histogram(p[:, 2], bins=np.arange(-.05, .3, .002))
    sm = np.convolve(hist, np.ones(3), 'same')
    i = int(np.argmax(sm >= .15 * sm.max()))
    i = i + int(np.argmax(sm[i:i + 6]))
    z0 = float(edges[i] + .001)
    w = p[np.abs(p[:, 2] - z0) < .01]
    c = w.mean(0); _, _, vt = np.linalg.svd(w[::5] - c); n = vt[-1]
    if n[2] < 0: n = -n
    return (c, n), z0


def plane_z(plane, xy):
    c, n = plane
    return float(c[2] - (n[0] * (xy[0] - c[0]) + n[1] * (xy[1] - c[1])) / n[2])


def cabinet(io, obs, plane, z0):
    proc = S.get_proc(); p, ys, xs = G.safe_points(obs); h = G.height(p, plane)
    st = proc.set_image(Image.fromarray(obs['rgb']))
    best = None
    for pr in CAB_PROMPTS:
        m, s = S._prompt(proc, st, pr)
        for mk, sc in zip(m, s):
            sel = mk[ys, xs] & (h > .03) & (h < .2)
            if sc < .3 or sel.sum() < 2000:
                continue
            if best is None or sc > best[0]:
                best = (float(sc), pr, p[sel])
    if best is None:
        raise RuntimeError('drawer not found')
    sc, pr, q = best
    x = np.percentile(q[:, 0], 1)
    y = .5 * (np.percentile(q[:, 1], 2) + np.percentile(q[:, 1], 98))
    ky, kn, ktop = None, None, None
    m, s = S._prompt(proc, st, 'knob')
    for mk, sk in zip(m, s):
        kq = p[mk[ys, xs]]
        if sk < .3 or len(kq) < 30:
            continue
        kc = np.median(kq, 0)
        if abs(kc[0] - x) < .04 and abs(kc[1] - y) < .12:
            ky, kn, ktop = float(kc[1]), int(len(kq)), float(np.percentile(kq[:, 2], 95)); break
    io.log('cabinet', prompt=pr, score=sc, face_x=x, centre_y=y, knob_y=ky, knob_n=kn, top=float(np.percentile(q[:, 2], 95)))
    kz = z0 + KNOB_DZ if ktop is None else ktop - KNOB_TOP_DZ
    io.log('knob_z', top=ktop, z=kz, table_based=z0 + KNOB_DZ)
    return np.array([x, y if ky is None else ky, kz]), q


def find_block(obs, plane, cab_points):
    proc = S.get_proc(); p, ys, xs = G.safe_points(obs); h = G.height(p, plane)
    st = proc.set_image(Image.fromarray(obs['rgb']))
    lo = cab_points[:, :2].min(0) - CAB_MARGIN; hi = cab_points[:, :2].max(0) + CAB_MARGIN
    diag = []; best = None
    for prompt in PROMPTS:
        masks, scores = S._prompt(proc, st, prompt)
        for m, s in zip(masks, scores):
            if s < .3:
                continue
            sel = m[ys, xs] & (h > .005) & (h < .06)
            n = int(sel.sum())
            if n < 40:
                diag.append({'prompt': prompt, 'score': float(s), 'reject': 'few_points', 'n': n}); continue
            q = p[sel]; hq = h[sel]
            ext = q[:, :2].max(0) - q[:, :2].min(0); top = float(np.percentile(hq, 95))
            c = np.r_[np.median(q[:, :2], 0), np.median(q[:, 2])]
            if np.all(c[:2] > lo) and np.all(c[:2] < hi):
                diag.append({'prompt': prompt, 'score': float(s), 'reject': 'inside_cabinet_footprint', 'center': c}); continue
            if ext.max() > .09 or not (.015 <= top <= .05):
                diag.append({'prompt': prompt, 'score': float(s), 'reject': 'not_block_like', 'extent': ext, 'top': top}); continue
            tp = q[hq > top - .008]
            (_, _), (_, _), ang = cv2.minAreaRect(tp[:, :2].astype(np.float32) * 1000)
            cand = {'xy': tp[:, :2].mean(0), 'yaw': float(np.radians(ang)), 'score': float(s), 'n': n, 'prompt': prompt}
            diag.append({'prompt': prompt, 'score': float(s), 'accept': True, 'center': c, 'n': n})
            if best is None or cand['score'] > best['score']:
                best = cand
        if best is not None and best['score'] >= .7:
            break
    return best, diag


# ---------- robot-only self-collision (robot MJCF, no task world, no objects) ----------
class SelfCheck:
    def __init__(self, home):
        m = self.m = mujoco.MjModel.from_xml_path(str(ROBOT.MJCF)); self.d = mujoco.MjData(m)
        self.adr = [[int(m.joint(j).qposadr[0]) for j in ROBOT.ARMS[a]['joints']] for a in ARMS]
        self.base = set(range(m.nbody))
        self.q = np.asarray(home, float).reshape(2, 6).copy()
        self.home_pairs = self._pairs(self.q)

    def _pairs(self, q, margin=.003):
        d = self.d
        for a in range(2):
            d.qpos[self.adr[a]] = q[a]
        self.m.geom_margin[:] = margin
        mujoco.mj_forward(self.m, d)
        return {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in d.contact[:d.ncon]}

    def ok(self, arm, qa, other):
        q = np.array(other, float).reshape(2, 6).copy(); q[ARMS.index(arm)] = qa
        bad = self._pairs(q) - self.home_pairs
        return not bad


CHK = {}


# ---------- motion ----------
def chain(io, arm, legs):
    kin = io._w.kinematics; enc = io._w.encoders(); q = enc[ARMS.index(arm)]
    chk = CHK.get('c')
    start = kin.fk(arm, q)
    for p, R in legs:
        for k in range(1, 21):
            wp, wr = waypoint(start, p, R, k / 20)
            q = kin.ik(arm, wp, wr, q, 400)
            if chk is not None and not chk.ok(arm, q, enc):
                raise KinematicsError('self-collision on planned path')
        start = kin.fk(arm, q)
    return q


def move(io, arm, p, R, dur, fix=False):
    io.move(arm, p, R, dur)
    if not fix:
        return
    tgt = np.array(p, float)
    for _ in range(2):
        e = np.asarray(p) - io.tcp(arm)[:3, 3]
        if np.linalg.norm(e) < .003:
            break
        tgt = tgt + e
        try:
            io.move(arm, tgt, R, .5)
        except KinematicsError:
            break
    io.log('sag', arm=arm, residual=(np.asarray(p) - io.tcp(arm)[:3, 3]))


def knob_frames(lc_rot):
    out = [('s1', Rot.from_euler('y', -30, degrees=True).as_matrix() @ lc_rot)]
    for pitch in (30, 15, 45, 0):
        a = np.radians(pitch); z = np.array([np.cos(a), 0., -np.sin(a)])
        for name, y in (('h', np.array([0., 1., 0.])), ('v', np.cross(z, [0., 1., 0.]))):
            for sgn in (1, -1):
                yy = sgn * y; out.append((f'p{pitch}{name}{"+" if sgn > 0 else "-"}', np.column_stack([np.cross(yy, z), yy, z])))
    return out


def open_drawer(io, knob, lc_rot, pulls=PULLS):
    g = knob + [-.002, 0, 0]
    for pull in pulls:
        for arm in sorted(ARMS, key=lambda a: abs(io.tcp(a)[0, 3] - (g[0] - pull))):
            for name, R in knob_frames(lc_rot):
                pre = g - R[:, 2] * .06 + [0, 0, .03]
                legs = [(pre, R), (g, R), (g + [-pull, 0, 0], R), (g + [-pull - .05, 0, .08], R)]
                try:
                    chain(io, arm, legs)
                except KinematicsError:
                    continue
                io.log('pull_plan', arm=arm, frame=name, grasp=g, pull=pull)
                io.grip(arm, 1., .3)
                move(io, arm, pre, R, 1.5)
                move(io, arm, g, R, 1.0, fix=True)
                io.grip(arm, 0., .6)
                move(io, arm, g + [-pull, 0, 0], R, 2.5)
                io.grip(arm, 1., .5)
                try:
                    move(io, arm, g + [-pull - .05, 0, .08], R, 1.0)
                except KinematicsError:
                    pass
                io.home(arm, 2.0)
                return arm
    raise RuntimeError('no IK-feasible knob pull')


def lean_order(io, arm, xy, yaws):
    """(yaw, tilt) candidates: approach leaning outward along the shoulder->block radial first (|tilt| ascending), then the rest."""
    sh = io._w.kinematics.T_task_base[:3, 3][:2]
    r = np.asarray(xy) - sh; r /= max(np.linalg.norm(r), 1e-9)
    c = []
    for yaw in yaws:
        x = np.cross([np.cos(yaw), np.sin(yaw), 0.], -UP)
        for t in TILTS:
            out = np.sign(t) * (x[:2] @ r)
            c.append((0 if t == 0 else (1 if out > .3 else 2), abs(t), yaws.index(yaw), yaw, t))
    return [(y, t) for *_, y, t in sorted(c)]


def block_frame(yaw, tilt):
    z = -UP; y = np.array([np.cos(yaw), np.sin(yaw), 0.])
    R = np.column_stack([np.cross(y, z), y, z])
    c, s = np.cos(tilt), np.sin(tilt)
    return R @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def finger_clear(g, R, cab_xy):
    """xy distance from the open finger tips and the wrist footprint to the nearest perceived cabinet/drawer point."""
    pts = [g + s * .055 * R[:, 1] for s in (-1, 1)] + [g - .12 * R[:, 2] + s * .05 * R[:, 1] for s in (-1, 1)]
    return float(min(np.linalg.norm(cab_xy - p[:2], axis=1).min() for p in pts))


def pick_drop(io, plane, blk, face, cab_front_x, z0, demo_yaw, cab=None):
    gz = plane_z(plane, blk['xy'])
    mid = face[0] + .5 * (cab_front_x - face[0])
    drops = [np.r_[mid, face[1], z0 + dz] for dz in (.14, .12, .11)]
    yaws = sorted([blk['yaw'] + k * np.pi / 2 for k in range(4)], key=lambda a: abs(np.angle(np.exp(1j * (a - demo_yaw)))))
    for d in drops:
        for hov, lift in ((.10, .15), (.06, .10)):
            for arm in sorted(ARMS, key=lambda a: abs(io.tcp(a)[0, 3] - blk['xy'][0])):
                cands = lean_order(io, arm, blk['xy'], yaws)
                if cab is not None:
                    gg = np.r_[blk['xy'], gz + GRASP_DZ]; cxy = cab[:, :2][::7]
                    cands = sorted(cands, key=lambda c: finger_clear(gg, block_frame(*c), cxy) < CLEAR)  # stable: clear ones first
                for yaw, tilt in cands:
                    R = block_frame(yaw, tilt)
                    g = np.r_[blk['xy'], gz + TILT_DZ.get(int(round(abs(np.degrees(tilt)))), GRASP_DZ)]
                    pre, post = g - hov * R[:, 2], g - lift * R[:, 2]
                    legs = [(pre, R), (g, R), (post, R), (d + .05 * UP, R), (d, R)]
                    try:
                        chain(io, arm, legs)
                    except KinematicsError:
                        continue
                    io.log('pick_plan', clear=None if cab is None else finger_clear(np.r_[blk['xy'], gz + GRASP_DZ], R, cab[:, :2][::7]), arm=arm, yaw=np.degrees(yaw), tilt=np.degrees(tilt), hover=hov, lift=lift, grasp=g, drop=d)
                    io.grip(arm, 1., .3)
                    move(io, arm, pre, R, 1.5)
                    move(io, arm, g, R, 1.0, fix=True)
                    io.grip(arm, 0., .6)
                    move(io, arm, post, R, 1.0)
                    move(io, arm, d + .05 * UP, R, 1.5)
                    move(io, arm, d, R, .8)
                    io.grip(arm, 1., .6)
                    for off in ([0, 0, .10], [0, 0, .06], [-.08, 0, .03]):
                        try:
                            move(io, arm, d + np.array(off), R, .8); break
                        except KinematicsError:
                            io.log('retreat_unreachable', off=off)
                    io.home(arm, 1.5)
                    return arm
    raise RuntimeError('no IK-feasible block pick/drop')


def run(io):
    ev = io.demo['events']
    lc = next(e for e in ev if e['arm'] == 'left' and e['kind'] == 'close')
    rc = next(e for e in ev if e['arm'] == 'right' and e['kind'] == 'close')
    demo_yaw = float(np.arctan2(np.asarray(rc['rotation'])[1, 1], np.asarray(rc['rotation'])[0, 1]))
    CHK['c'] = SelfCheck(io._w.encoders())
    obs = io.observe()
    plane, z0 = table_plane(obs)
    io.log('table', z0=z0, n=plane[1], modal=P.table_height(obs))
    knob, _ = cabinet(io, obs, plane, z0)
    io.log('knob', knob=knob)
    open_drawer(io, knob, np.asarray(lc['rotation']))
    obs = io.observe()
    face, cab = cabinet(io, obs, plane, z0)
    for attempt in range(2):
        if knob[0] - face[0] >= MIN_OPEN:
            break
        io.log('short_opening_regrasp', attempt=attempt, opened_m=float(knob[0] - face[0]))
        try:
            need = TARGET_OPEN - float(knob[0] - face[0])
            open_drawer(io, face, np.asarray(lc['rotation']), (need + .02, need, need - .02))
        except RuntimeError as e:
            io.log('regrasp_failed', error=str(e)); break
        obs = io.observe()
        face, cab = cabinet(io, obs, plane, z0)
    io.log('opened_face', face=face, opened_m=float(knob[0] - face[0]))
    blk, diag = find_block(obs, plane, cab)
    io.log('block_candidates', candidates=diag)
    if blk is None:
        raise RuntimeError('block not found by SAM3 prompts')
    io.log('block_perceived', block=blk)
    pick_drop(io, plane, blk, face, knob[0], z0, demo_yaw, cab)
    io.wait(1.5)
    io.observe()
