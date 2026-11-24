"""build_tower lift2_st6 = st2 + yellow pre-staging: when the first IK-feasible yellow grasp would lean the wrist toward the bar, the yellow cube is first moved to a staging spot where a non-leaning grasp+stack plan exists (LIFT2, smooth_table_v1): xTrainer t1 skill (SAM3 roles, bar/cube by mask elongation, cube on bar,
yellow on cube) with a LIFT2-specific plan.
Shared with t1: SAM3 prompts 'yellow cube' / 'purple cube', bar = more elongated purple instance, pick height table+.021,
release heights from t1 (+.033 above the support top), dwell.
LIFT2-specific plan:
- top-down canonical grasp frame (approach = -table normal, closing axis aligned to a face of the block from its mask
  points), 4 yaws per block ordered by distance to the demo closing axis; the arm and yaw are chosen by an IK chain check
  from the current encoders before any motion;
- table height at the block's own xy from the fitted (tilted) plane instead of one modal height;
- encoder-FK sag correction on the descend legs (target + measured residual, at most 2 passes);
- head blind zone: the wrist D405 of a parked arm looks at the area in front of the robot; instances from head and wrist are
  merged, and an instance cut by an image border is replaced by a complete view of the same block when one exists;
- the yellow target is re-observed after the purple cube is placed (top of the purple stack).
Runtime inputs: rendered RGB-D (head, wrist), calibration, encoders, robot model, demo contact priors (closing axis of the
left/right close events). No simulator object state."""
import numpy as np
import cv2
from rsi import perception_sam3 as S
from rsi import perception_geo as G
from twin.practice.motion import KinematicsError, waypoint

ARMS = ('left', 'right')
GRASP_DZ = .021      # t1: TCP above the table at pick
REL_DZ = .033        # t1: TCP above the support top at release (.021 + .012)
CUBE = .032
BORDER = 6
HOVERS = ((.10, .15), (.06, .09))
TILTS = np.radians([0., 20., -20., 35., -35.])


def plane_z(plane, xy):
    c, n = plane
    return float(c[2] - (n[0] * (xy[0] - c[0]) + n[1] * (xy[1] - c[1])) / n[2])


def frame(plane, yaw, tilt=0.):
    """Canonical grasp frame: approach z = -plane normal, closing y at yaw; tilt rotates about the closing axis (fingers stay on faces)."""
    z = -plane[1] / np.linalg.norm(plane[1])
    y = np.array([np.cos(yaw), np.sin(yaw), 0.]); y -= z * (y @ z); y /= np.linalg.norm(y)
    R = np.column_stack([np.cross(y, z), y, z])
    if tilt:
        c, s = np.cos(tilt), np.sin(tilt)
        R = R @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return R


def complete(obs, pts):
    K, T = obs['K'], obs['T_world_from_camera']
    c = (pts - T[:3, 3]) @ T[:3, :3]
    u = K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2]; v = K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2]
    h, w = obs['depth'].shape
    return bool(u.min() > BORDER and v.min() > BORDER and u.max() < w - BORDER and v.max() < h - BORDER)


def describe(obs, b, plane, label, src):
    p = b['points']; hq = G.height(p, plane)
    top = p[hq > np.percentile(hq, 95) - .008]
    xy = top[:, :2].mean(0)
    (_, _), (w, h), ang = cv2.minAreaRect(top[:, :2].astype(np.float32) * 1000)
    ev = np.linalg.eigvalsh(np.cov((p[:, :2] - p[:, :2].mean(0)).T))
    return dict(label=label, src=src, xy=xy, top=float(np.percentile(p[:, 2], 95)), yaw=np.radians(ang),
                elong=float(np.sqrt(max(ev[-1], 1e-12) / max(ev[0], 1e-12))), n=int(b['n']), score=float(b['score']),
                complete=complete(obs, p))


def detect(obs, plane, src):
    proc = S.get_proc(); out = []
    for label, pr in (('yellow', 'yellow cube'), ('purple', 'purple cube')):
        bl, _ = S.find_blocks(proc, obs, prompt=(pr,), plane=plane, bowl=None)
        out += [describe(obs, b, plane, label, src) for b in bl]
    return out


def merge(dets):
    """Cluster instances by xy (3 cm); keep one per cluster: complete first, then more points."""
    cl = []
    for d in sorted(dets, key=lambda d: (not d['complete'], -d['n'])):
        if any(np.linalg.norm(d['xy'] - c['xy']) < .03 for c in cl):
            continue
        cl.append(d)
    return cl


def seeds(io, arm):
    return io._w.encoders()[ARMS.index(arm)]


def chain(io, arm, legs, q0=None):
    """IK check of consecutive legs [(p, R)] with the io.move preflight law (20 waypoints), seeded from the encoders."""
    kin = io._w.kinematics; q = seeds(io, arm) if q0 is None else q0
    start = kin.fk(arm, q)
    for p, R in legs:
        for k in range(1, 21):
            wp, wr = waypoint(start, p, R, k / 20)
            q = kin.ik(arm, wp, wr, q, 400)
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


def wrist_look(io, plane, head):
    """Park-view with a wrist D405 over the head's blind zone (area between the head image bottom edge and the robot)."""
    K, T = head['K'], head['T_world_from_camera']
    h, w = head['depth'].shape
    pts = []
    for u in (w * .25, w * .5, w * .75):
        d = T[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (h - K[1, 2]) / K[1, 1], 1.])
        c = T[:3, 3]; s = (plane_z(plane, c[:2]) - c[2]) / d[2]
        pts.append(c + s * d)
    edge = np.mean(pts, 0)
    base = io._w.kinematics.T_task_base[:3, 3]
    target = edge + .25 * (base - edge) * np.r_[1, 1, 0]   # a quarter of the way from the edge toward the base
    target[2] = plane_z(plane, target[:2])
    kin = io._w.kinematics
    best = None
    for arm in ARMS:
        Rh = io.tcp(arm)[:3, :3]
        for back in (.10, .14, .06, .18):
            for z in (.25, .30, .21):
                fwd = Rh[:, 0]
                p = target - back * fwd; p[2] = plane_z(plane, p[:2]) + z
                try:
                    q = chain(io, arm, [(p, Rh)])
                except KinematicsError:
                    continue
                C = kin.camera_fk(arm, q); c = (target - C[:3, 3]) @ C[:3, :3]
                if c[2] <= 0:
                    continue
                Kw = 290.; u, v = Kw * c[0] / c[2], Kw * c[1] / c[2]
                cost = abs(u) + abs(v)
                if best is None or cost < best[0]:
                    best = (cost, arm, p, Rh)
    if best is None:
        io.log('wrist_look_unreachable', target=target)
        return []
    _, arm, p, Rh = best
    io.log('wrist_look', arm=arm, target=target, tcp=p)
    move(io, arm, p, Rh, 1.5)
    io.wait(.3)
    obs = io.observe(camera=arm + '_d405')
    dets = detect(obs, plane, arm + '_d405')
    io.home(arm, 1.5)
    return dets


def roles(io, plane, head_dets, extra):
    dets = merge(head_dets + extra)
    yel = sorted([d for d in dets if d['label'] == 'yellow'], key=lambda d: (not d['complete'], -d['score']))
    pur = sorted([d for d in dets if d['label'] == 'purple'], key=lambda d: (not d['complete'], -d['score']))
    if not yel:
        raise RuntimeError('no yellow cube')
    yel = yel[0]
    pur = [d for d in pur if np.linalg.norm(d['xy'] - yel['xy']) > .04][:2]
    if len(pur) < 2:
        raise RuntimeError('%d purple instances' % len(pur))
    bar = max(pur, key=lambda d: d['elong']); cube = [d for d in pur if d is not bar][0]
    io.log('roles', bar=bar, cube=cube, yellow=yel)
    return bar, cube, yel


def yaws(d, demo_yaw):
    c = [d['yaw'] + k * np.pi / 2 for k in range(4)]
    return sorted(c, key=lambda a: abs(np.angle(np.exp(1j * (a - demo_yaw)))))


def pick_place(io, plane, blk, dest_xy, dest_top, demo_yaw, tag, order=ARMS):
    g = np.r_[blk['xy'], plane_z(plane, blk['xy']) + GRASP_DZ]
    d = np.r_[dest_xy, dest_top + REL_DZ]
    up = plane[1] / np.linalg.norm(plane[1])
    for hov, lift in HOVERS:
        for arm in order:
            for yaw, tilt in [(y, t) for t in TILTS for y in yaws(blk, demo_yaw[arm])]:
                R = frame(plane, yaw, tilt)
                legs = [(g + hov * up, R), (g, R), (g + lift * up, R), (d + .06 * up, R), (d, R), (d + .06 * up, R)]
                try:
                    chain(io, arm, legs)
                except KinematicsError:
                    continue
                io.log('plan', tag=tag, arm=arm, yaw=np.degrees(yaw), tilt=np.degrees(tilt), hover=hov, lift=lift, grasp=g, drop=d)
                io.grip(arm, 1., .3)
                move(io, arm, g + hov * up, R, 1.5)
                move(io, arm, g, R, 1.0, fix=True)
                io.grip(arm, 0., .6)
                move(io, arm, g + lift * up, R, 1.0)
                move(io, arm, d + .06 * up, R, 1.5)
                move(io, arm, d, R, .8, fix=True)
                io.grip(arm, 1., .6)
                try:
                    move(io, arm, d + .06 * up, R, .8)
                except KinematicsError:
                    pass
                io.home(arm, 2.0)
                return arm
    raise RuntimeError(tag + ': no IK-feasible arm/yaw')


def stack_top(obs, plane, bar_xy):
    """Top layer of the purple stack near the bar: SAM3 'purple cube' mask points 45-120 mm above the plane within 10 cm of the bar."""
    from PIL import Image
    proc = S.get_proc(); p, ys, xs = G.safe_points(obs); h = G.height(p, plane)
    st = proc.set_image(Image.fromarray(obs['rgb']))
    masks, scores = S._prompt(proc, st, 'purple cube')
    best = None
    for m, sc in zip(masks, scores):
        if sc < .3:
            continue
        sel = m[ys, xs] & (h > .045) & (h < .12)
        q = p[sel]
        if len(q) < 40:
            continue
        q = q[np.linalg.norm(q[:, :2] - bar_xy, axis=1) < .10]
        if len(q) < 40:
            continue
        hq = G.height(q, plane); t = float(np.percentile(hq, 95)); top = q[hq > t - .008]
        cand = (top[:, :2].mean(0), float(np.percentile(q[:, 2], 95)), len(q))
        if best is None or cand[2] > best[2]:
            best = cand
    return None if best is None else best[:2]


def lean_ok(R, g, obstacles):
    """1 if the wrist (behind the TCP along -approach) leans toward a perceived block within 16 cm, else 0."""
    back = -R[:2, 2]
    if np.linalg.norm(back) < .05:
        return 0
    back = back / np.linalg.norm(back)
    for xy in obstacles:
        v = np.asarray(xy) - g[:2]; dist = np.linalg.norm(v)
        if dist < .16 and back @ (v / dist) > .3:
            return 1
    return 0


def first_plan(io, plane, xy, yaw0, dest_xy, dest_top, demo_yaw, order):
    """First IK-feasible (arm, R) in pick_place order for a block at xy, or None."""
    blk = {'xy': np.asarray(xy), 'yaw': yaw0}
    g = np.r_[blk['xy'], plane_z(plane, blk['xy']) + GRASP_DZ]; d = np.r_[dest_xy, dest_top + REL_DZ]
    up = plane[1] / np.linalg.norm(plane[1])
    for hov, lift in HOVERS:
        for arm in order:
            for yaw, tilt in [(y, t) for t in TILTS for y in yaws(blk, demo_yaw[arm])]:
                R = frame(plane, yaw, tilt)
                try:
                    chain(io, arm, [(g + hov * up, R), (g, R), (g + lift * up, R), (d + .06 * up, R), (d, R), (d + .06 * up, R)])
                    return arm, R, g
                except KinematicsError:
                    continue
    return None


def stage_yellow(io, plane, yel, bar, cube, demo_yaw, xs):
    """Move the yellow cube to a staging spot when its stack plan leans toward the bar. Returns the (possibly new) yellow dict."""
    order = tuple(sorted(ARMS, key=lambda a: abs(xs[a] - yel['xy'][0])))
    fp = first_plan(io, plane, yel['xy'], yel['yaw'], bar['xy'], bar['top'] + CUBE, demo_yaw, order)
    if fp is None or not lean_ok(fp[1], fp[2], [bar['xy']]):
        io.log('stage_skip', reason='no plan' if fp is None else 'not leaning')
        return yel
    spots = []
    for dx in np.arange(-.12, .13, .03):
        for dy in (-.04, -.07, -.10, -.01):
            sxy = np.array([bar['xy'][0] + dx, yel['xy'][1] - dy])
            if min(np.linalg.norm(sxy - bar['xy']), np.linalg.norm(sxy - cube['xy']), np.linalg.norm(sxy - yel['xy'])) < .07:
                continue
            spots.append(sxy)
    spots.sort(key=lambda q: np.linalg.norm(q - yel['xy']))
    for sxy in spots:
        o2 = tuple(sorted(ARMS, key=lambda a: abs(xs[a] - sxy[0])))
        fp2 = first_plan(io, plane, sxy, yel['yaw'], bar['xy'], bar['top'] + CUBE, demo_yaw, o2)
        if fp2 is None or lean_ok(fp2[1], fp2[2], [bar['xy']]):
            continue
        try:  # move yellow to the spot (dest top = table at the spot, so the release height is table + .033)
            pick_place(io, plane, yel, sxy, plane_z(plane, sxy), demo_yaw, 'stage', order)
        except RuntimeError as e:
            io.log('stage_failed', error=str(e)); return yel
        o = io.observe(); dets = [d for d in detect(o, plane, 'stage') if d['label'] == 'yellow']
        near = [d for d in dets if np.linalg.norm(d['xy'] - sxy) < .05]
        ny = max(near, key=lambda d: d['score']) if near else dict(yel, xy=sxy)
        io.log('staged', spot=sxy, observed=bool(near), xy=ny['xy'])
        return ny
    io.log('stage_skip', reason='no staging spot')
    return yel


def run(io):
    ev = io.demo['events']
    demo_yaw = {}
    for arm in ARMS:
        e = next(e for e in ev if e['arm'] == arm and e['kind'] == 'close')
        y = np.asarray(e['rotation'])[:, 1]; demo_yaw[arm] = float(np.arctan2(y[1], y[0]))
    head = io.observe()
    plane = G.fit_plane(head)
    hd = detect(head, plane, 'head')
    io.log('head', dets=[{k: v for k, v in d.items()} for d in hd])
    extra = wrist_look(io, plane, head)
    io.log('wrist', dets=extra)
    bar, cube, yel = roles(io, plane, hd, extra)
    # nearer arm first for the purple cube
    xs = {a: io.tcp(a)[0, 3] for a in ARMS}
    yel = stage_yellow(io, plane, yel, bar, cube, demo_yaw, xs)
    order = tuple(sorted(ARMS, key=lambda a: abs(xs[a] - cube['xy'][0])))
    pick_place(io, plane, cube, bar['xy'], bar['top'], demo_yaw, 'cube', order)
    o2 = io.observe()
    st = stack_top(o2, plane, bar['xy'])
    stack = st is not None
    if stack:
        sxy, stop = st
    else:
        sxy, stop = bar['xy'], bar['top'] + CUBE
    io.log('stack', found=bool(stack), xy=sxy, top=stop)
    order = tuple(sorted(ARMS, key=lambda a: abs(xs[a] - yel['xy'][0])))
    pick_place(io, plane, yel, sxy, stop, demo_yaw, 'yellow', order)
    io.wait(1.5)
    io.observe()
