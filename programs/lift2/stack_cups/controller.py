'''stack_cups s3 (LIFT2, smooth table). Observation-only: head RGB-D + K + camera pose, encoders, robot model (canonical FK/IK and
the mount T_task_base), instruction, listed demo contact prior. Never reads simulator object state, contacts or the scorer.

SHARED SKILL (xTrainer c6 semantics): SAM3 'cup'/'paper cup' masks lifted by depth; rim-axis fit (fit_upright_cylinder on the top 12 mm);
roles purple = grasped cup, blue = target cup (mask median hue, declared colour rule); demo contact prior = grasp the purple cup from above near
its rim, carry, release above the blue cup's axis, open, retreat up.
LIFT2 PLAN: s1 evidence (DEV2 o1): the c6 one-wall rim pinch let the cup swing in the LIFT2 fingers and it wedged tilted on the blue rim.
s3 instead closes across the whole purple rim (open tip gap 83 mm > 62 mm rim): TCP on the cup axis 15 mm below the rim top, closing axis
horizontal; the cup self-centres on the TCP and hangs with its centre of mass under the grip line. Approach = base -z tilted forward by a pitch
toward the heading of the point; closing perpendicular to that heading. A planner replays the facade 20-waypoint preflight over the whole chain
(pre-grasp, grasp, lift, carry, pre-release, release, retreat) per arm/pitch; arm on the blue cup's side first; lowest pitch first.
Release: purple bottom 8 mm above the blue rim.'''
import numpy as np
import cv2
from PIL import Image
from rsi import perception as P
from rsi import perception_sam3 as S
from twin.practice.motion import waypoint

ARMS = ('left', 'right')
PITCHES = (0., .2, .35, .5, .65, .8)
BELOW_RIM = .015
REL_GAP = .008


def base_cloud(obs, Tb):
    p, ys, xs = P.points(obs)
    return (p - Tb[:3, 3]) @ Tb[:3, :3], ys, xs


def table_z(pb):
    z = pb[:, 2]; h = np.histogram(z, bins=np.arange(z.min(), z.max() + .002, .002))
    return float(h[1][np.argmax(h[0])] + .001)


def find_cups(obs, Tb, min_score=.3):
    pb, ys, xs = base_cloud(obs, Tb); tz = table_z(pb); h = pb[:, 2] - tz
    proc = S.get_proc(); st = proc.set_image(Image.fromarray(obs['rgb']))
    hsv = cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2HSV)
    pairs = []
    for pr in ('cup', 'paper cup'):
        mk, sc = S._prompt(proc, st, pr); pairs += list(zip(mk, sc))
    res = []
    for m, s in sorted(pairs, key=lambda x: -x[1]):
        if s < min_score: continue
        sel = m[ys, xs] & (h > .005) & (h < .14)
        if sel.sum() < 100: continue
        q = pb[sel]; c = np.median(q[:, :2], 0)
        if any(np.linalg.norm(c - r['c']) < .03 for r in res): continue
        top = float(np.percentile(q[:, 2], 95)); rim = q[q[:, 2] > top - .012]
        xy, r = P.fit_upright_cylinder(rim)
        res.append({'c': c, 'axis': np.asarray(xy, float), 'r': float(r), 'top': top, 'hue': float(np.median(hsv[ys[sel], xs[sel], 0])),
                    'score': float(s), 'n': int(sel.sum())})
    return res, tz


def pose(Tb, p_base, pitch):
    '''World TCP pose (canonical: +z approach, +y closing). Approach = base -z tilted by pitch toward the base-frame heading of the point;
    closing horizontal, perpendicular to the heading.'''
    p_base = np.asarray(p_base, float); hd = np.arctan2(p_base[1], p_base[0])
    h = np.array([np.cos(hd), np.sin(hd), 0.]); y = np.array([-np.sin(hd), np.cos(hd), 0.])
    a = np.cos(pitch) * np.array([0, 0, -1.]) + np.sin(pitch) * h
    return Tb[:3, :3] @ p_base + Tb[:3, 3], Tb[:3, :3] @ np.stack([np.cross(y, a), y, a], 1)


def chain_ok(io, arm, chain, seed):
    kin = io._w.kinematics; q = np.asarray(seed, float); start = kin.fk(arm, q)
    for p, R in chain:
        try:
            for k in range(1, 21):
                wp, wr = waypoint(start, p, R, k / 20); q = kin.ik(arm, wp, wr, q, 300)
        except Exception:
            return False
        start = kin.fk(arm, q)
    return True


def run(io):
    Tb = np.asarray(io._w.kinematics.T_task_base, float)
    obs = io.observe()
    cups, tz = find_cups(obs, Tb)
    io.log('perceived', cups=cups, table_z=tz)
    if len(cups) < 2: raise RuntimeError('found %d cups' % len(cups))
    purple = min(cups, key=lambda c: abs(c['hue'] - 140)); blue = min((c for c in cups if c is not purple), key=lambda c: abs(c['hue'] - 108))
    io.log('roles', purple=purple['axis'], purple_hue=purple['hue'], purple_r=purple['r'], purple_top=purple['top'],
           blue=blue['axis'], blue_hue=blue['hue'], blue_r=blue['r'], blue_top=blue['top'])
    gz = purple['top'] - BELOW_RIM; below = gz - tz  # TCP height above the purple bottom while held
    g = np.r_[purple['axis'], gz]
    rel = np.r_[blue['axis'], blue['top'] + REL_GAP + below]
    carry = max(rel[2] + .03, gz + .05)
    plans = []
    for arm in ARMS:
        for pitch in PITCHES:
            ch = [pose(Tb, g + [0, 0, .07], pitch), pose(Tb, g, pitch), pose(Tb, np.r_[g[:2], carry], pitch), pose(Tb, np.r_[rel[:2], carry], pitch),
                  pose(Tb, rel, pitch), pose(Tb, rel + [0, 0, .07], pitch)]
            side = 0 if (blue['axis'][1] > 0) == (arm == 'left') else 1
            plans.append((side, pitch, arm, ch))
    plans.sort(key=lambda x: x[:2])
    for side, pitch, arm, ch in plans:
        if chain_ok(io, arm, ch, io.home_q[ARMS.index(arm)]): break
    else:
        raise RuntimeError('no feasible LIFT2 pose chain')
    io.log('plan', arm=arm, pitch=pitch, side=side, grasp=g, release=rel, carry=carry)
    (p0, R0), (p1, R1), (p2, R2), (p3, R3), (p4, R4), (p5, R5) = ch
    io.grip(arm, 1., .3)
    io.move(arm, p0, R0, 1.5); io.move(arm, p1, R1, 1.0)
    io.grip(arm, 0., .8); io.log('stage_grasped')
    io.move(arm, p2, R2, 1.0); io.move(arm, p3, R3, 1.5); io.move(arm, p4, R4, 1.0)
    io.wait(.3)
    io.log('stage_pre_release'); io.grip(arm, 1., .6); io.log('stage_post_open')
    io.move(arm, p5, R5, .8); io.log('stage_post_lift')
    io.home(arm, 1.5)
    io.wait(1.0)
    io.observe()
