# mug_r4_ctl (r3 + place points ordered by the PERCEIVED box-top height (plane fit to the box top depth points), highest first, and a 5x5 grid: the source frame is tilted 2.7 deg, so a mug
# released at the low end of the box top ends up to ~10 mm below the box's highest corner; mug_r3 seeds 0 and 2 failed 'on' by -0.1 mm on that margin, diag_mug_r3_on_seed0). r3 = (r2 + real hand surface vertices hand_pts_r2_mug_r1_grasp.npz (r2 used 6 mm voxel centres, up to 5 mm off the surface) and a TCP offset DL back along the approach from the mug axis
# so the palm clears the body; r2 had 0 collision-free candidates on seed 0 because the palm sits 45 mm behind the pinch frame and the body radius is 39-46 mm, diag_mug_coll_r2/r3). r2 = G1 put_mug_on_box controller (single grey-blue mug onto the wooden box). Runtime side: uses only io (rendered RGB-D via SAM3 masks + depth, encoders, demo priors) and the robot model. Never reads object truth, scorer or contacts.
# Shared skill (from the xTrainer m1 controller, not edited in place): SAM3 'mug' and 'wooden box' masks lifted to depth points -> mug axis from the rim, handle direction from depth -> grasp mug -> carry -> place upright on the box top -> release -> retreat.
# G1-specific plan (r1 failed on all 4 smoke seeds: every candidate either put the hand body into the table or the handle into the hand, then hit the wrist-pitch margin; diag_mug_plan_r1, diag_reach_r2):
# - grasp shape 'mug_wide' (mug_r1_grasp.py, trusted-side install, contact prior): thumb abducted, four fingers close around the mug body.
# - candidate grasps over side x tilt phi (0 = horizontal approach, 90 = top-down) x azimuth theta x flip x grasp height GHS; each must pass a ROBOT-MODEL collision check: hand collision points
#   (hand_pts_mug_r1_grasp.npz, from the robot model only) placed at the candidate pose against the PERCEIVED mug primitive (body cylinder with known radii + handle samples along the measured
#   handle direction), the perceived table plane, and later the perceived box top. Then the interpolated IK chain with a joint-limit margin (as cups_r6/r9).
# - one arm does pick and place; place points on a grid inside the perceived box top (nearest the robot first); hand yaw psi at the place is free (mug yaw does not matter for 'on').
# - release -> open -> retreat backwards along the approach -> rise. No joint-space home.
# Contact priors (listed): mug_wide grasp shape (calib_wide_r1.json); mug body radii 38.8 mm (base) to 45.9 mm (rim), height 81 mm, handle radial extent 42-69 mm at 15-70 mm height; code-constant grids.
import math
import os
import numpy as np
from scipy.spatial import cKDTree
from rsi import perception as P
from rsi import perception_sam3 as S
from rsi import perception_geo as G
import g1_common_r2 as C
from rsi.g1_suite.runtime.public_io import waypoint

GRASP = 'mug_r1_grasp'
HERE = os.path.dirname(os.path.abspath(__file__))
HP = np.load(os.path.join(HERE, 'hand_pts_r2_mug_r1_grasp.npz'))
R_BASE, R_RIM, H_MUG = .0388, .0459, .081
H_LO, H_HI, H_R0, H_R1 = .015, .070, .042, .069   # handle band (height above table) and radial extent
RIM_R = .045
HANDLE_R = .052
GHS = (.03, .045)       # pinch frame height above the table
DLS = (.024, .018, .030)  # pinch frame back along the approach from the mug axis
LIFT = .10
BACK = .10
PLACE_GAP = .006
EDGE = .05
STEP = .03
LIM_MARGIN = .08
CLEAR = .004
PHIS = (90, 75, 60, 45, 30, 15, 0)
THETAS = tuple(range(-180, 180, 15))
PSIS = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)


def frame(phi, theta, flip):
    a = np.array([math.cos(theta) * math.cos(phi), math.sin(theta) * math.cos(phi), -math.sin(phi)])
    l = np.array([-math.sin(theta), math.cos(theta), 0.]) * (-1. if flip else 1.)
    return np.column_stack([np.cross(l, a), l, a])


def Rz(p):
    c, s_ = math.cos(p), math.sin(p)
    return np.array([[c, -s_, 0.], [s_, c, 0.], [0., 0., 1.]])


def hand_world(side, opening, R, p):
    return p + HP['%s_%d' % (side, int(opening))] @ R.T


def mug_solid_hit(W, axis, zt, shrink=-.0005):
    h = W[:, 2] - zt
    r = np.linalg.norm(W[:, :2] - axis, axis=1)
    rr = R_BASE + (R_RIM - R_BASE) * np.clip(h / H_MUG, 0, 1) - shrink
    return bool(np.any((h > -.002) & (h < H_MUG + .003) & (r < rr)))


def handle_samples(axis, az, zt):
    u = np.array([math.cos(az), math.sin(az)])
    v = np.array([-u[1], u[0]])
    pts = []
    for rad in np.linspace(H_R0, H_R1, 5):
        for hh in np.linspace(H_LO, H_HI, 6):
            for w in (-.006, 0., .006):
                pts.append(np.r_[axis + rad * u + w * v, zt + hh])
    return np.array(pts)


def free(side, opening, R, p, axis, az, zt, floor):
    W = hand_world(side, opening, R, p)
    if W[:, 2].min() < floor + CLEAR:
        return False
    if mug_solid_hit(W, axis, zt):
        return False
    d, _ = cKDTree(W).query(handle_samples(axis, az, zt))
    return bool(d.min() > .006)


def chain(io, side, pts, start=None):
    if start is None:
        e = io._enc()
        T = io.adapter.tcp(e, side)
    else:
        e, T = dict(start[0]), start[1].copy()
    for p, R in pts:
        p = np.asarray(p, float)
        n = max(1, int(math.ceil(np.linalg.norm(p - T[:3, 3]) / STEP)))
        for k in range(1, n + 1):
            wp, wr = waypoint(T, p, R, k / n)
            try:
                q = io.adapter.plan(e, side, wp, wr)
            except Exception as ex:
                if io.error is not None:
                    raise
                return False, str(ex)[:60], None
            for nm, v in q.items():
                lo, hi = io.adapter.hand.rng[nm]
                if v < lo + LIM_MARGIN or v > hi - LIM_MARGIN:
                    return False, 'margin:' + nm, None
            e.update(q)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
    return True, '', (e, T)


def perceive(io, obs):
    plane = G.fit_plane(obs)
    proc = S.get_proc()
    zt = P.table_height(obs)
    m = S.locate_object(proc, obs, 'mug', plane=plane, max_height=.14)
    b = S.locate_object(proc, obs, 'wooden box', plane=plane, max_height=.16)
    if m is None or b is None:
        raise RuntimeError('perception failed mug=%s box=%s' % (m is not None, b is not None))
    mp = m['points']
    rim = mp[mp[:, 2] > m['top'] - .012]
    axis, _ = P.fit_upright_cylinder(rim, RIM_R)
    axis = np.asarray(axis, float)
    h = mp[:, 2] - zt
    rad = np.linalg.norm(mp[:, :2] - axis, axis=1)
    hp = mp[(h > .02) & (h < .07) & (rad > HANDLE_R)]
    if len(hp) < 15:
        raise RuntimeError('handle not seen (%d points)' % len(hp))
    v = hp[:, :2].mean(0) - axis
    handle_az = math.atan2(v[1], v[0])
    bp = b['points']
    top = bp[bp[:, 2] > b['top'] - .01]
    c0 = top[:, :2].mean(0)
    w, V = np.linalg.eigh(np.cov((top[:, :2] - c0).T))
    loc = (top[:, :2] - c0) @ V
    lo, hi = loc.min(0), loc.max(0)
    A = np.c_[top[:, :2], np.ones(len(top))]
    tilt = np.linalg.lstsq(A, top[:, 2], rcond=None)[0]
    io.log('perceived', mug_axis=axis, mug_top=m['top'], handle_az_deg=math.degrees(handle_az), n_handle=int(len(hp)), box_c=c0, box_top=b['top'],
           box_ext=(hi - lo), box_plane=tilt, table=zt, scores=[m['score'], b['score']])
    return axis, handle_az, zt, dict(c0=c0, V=V, lo=lo, hi=hi, top=float(b['top']), plane=tilt)


def place_grid(box):
    lo, hi = box['lo'] + EDGE, box['hi'] - EDGE
    pts = []
    for u in (np.linspace(lo[0], hi[0], 5) if hi[0] > lo[0] else [0.5 * (box['lo'][0] + box['hi'][0])]):
        for v in (np.linspace(lo[1], hi[1], 5) if hi[1] > lo[1] else [0.5 * (box['lo'][1] + box['hi'][1])]):
            pts.append(box['c0'] + box['V'] @ np.array([u, v]))
    a, b, _ = box['plane']
    return sorted(pts, key=lambda p: -(a * p[0] + b * p[1]))


def plan(io, axis, az, zt, box):
    n = 0
    first = 'left' if axis[1] > C.midline(io) else 'right'
    for side in (first, 'right' if first == 'left' else 'left'):
        for gh in GHS:
            for dl in DLS:
                for phi in PHIS:
                    for th in THETAS:
                        for flip in (0, 1):
                            R = frame(math.radians(phi), math.radians(th), flip)
                            c = np.r_[axis, zt + gh] - dl * R[:, 2]
                            pre = c - BACK * R[:, 2]
                            if not all(free(side, 1, R, c + s_ * (pre - c), axis, az, zt, zt) for s_ in (0., .25, .5, .75, 1.)):
                                continue
                            n += 1
                            lift = c + [0, 0, LIFT]
                            ok, _, st = chain(io, side, [(pre, R), (c, R), (lift, R)])
                            if not ok:
                                continue
                            for dxy in place_grid(box):
                                for psi in PSIS:
                                    R2 = Rz(math.radians(psi)) @ R
                                    d = np.r_[dxy, box['top'] + PLACE_GAP + gh] - dl * R2[:, 2]
                                    if hand_world(side, 0, R2, d)[:, 2].min() < box['top'] + CLEAR:
                                        continue
                                    back = d - BACK * R2[:, 2]
                                    maz = az + math.radians(psi)
                                    if not all(free(side, 1, R2, d + s_ * (back - d), dxy, maz, box['top'], box['top']) for s_ in (.25, .5, .75, 1.)):
                                        continue
                                    ok2, _, _ = chain(io, side, [(d + [0, 0, .05], R2), (d, R2), (back, R2)], start=st)
                                    if ok2:
                                        io.log('plan', side=side, gh=gh, dl=dl, phi=phi, theta=th, flip=flip, psi=psi, dest=d, n_tried=n)
                                        return side, R, R2, pre, c, lift, d, back
    raise RuntimeError('no_reachable_pick_place n_tried=%d' % n)


def controller(io):
    C.park(io)
    obs = io.observe()
    axis, az, zt, box = perceive(io, obs)
    side, R, R2, pre, c, lift, d, back = plan(io, axis, az, zt, box)
    io.grip(side, 1., .3)
    io.move(side, pre, R, 1.5)
    io.move(side, c, R, 1.2)
    io.grip(side, 0., .8)
    io.wait(.3)
    io.move(side, lift, R, 1.2)
    io.move(side, d + [0, 0, .05], R2, 2.0)
    io.move(side, d, R2, 1.0)
    io.wait(.3)
    io.log('released_at', side=side, tcp=io.tcp(side)[:3, 3])
    io.grip(side, 1., .6)
    io.wait(.3)
    try:
        io.move(side, back, R2, 1.2)
        io.move(side, back + [0, 0, .05], R2, .8)
    except Exception as e:
        if io.error is not None:
            raise
        io.log('retreat_failed', err=str(e)[:80])
    io.wait(2.0)
