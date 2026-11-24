'''collect_cubes k2 = k1 + (a) closing axis = the cube face normal most perpendicular to the pitched approach heading
(bar: its short axis only; k1 DEV1 far cubes missed 3+ times with a tilted finger pair); (b) grasp check from the gripper encoder (holding a cube
reads ~0.014 m, empty 0.000): an empty close re-opens and retries the same cube with the next plan before any carry; (c) grasp TCP height keeps
the tilted fingertips 4 mm above the observed table. k1 docstring follows.
(LIFT2, smooth table). Observation-only: head RGB-D + K + camera pose, encoders, robot model (canonical FK/IK
and the mount T_task_base), instruction, listed demo contact prior. Never reads simulator object state, contacts or the scorer.

SHARED SKILL (xTrainer c30 semantics): SAM3 'bowl' mask -> rim cylinder fit; SAM3 'small cube'/'block' masks lifted by depth, cubes inside the bowl
disc or taller than 60 mm excluded; demo contact prior = top-down two-finger pinch of a 32 mm cube across two faces, TCP 12 mm below the cube top
(c30 grasp height), release above the bowl and open; successive drops on a ring of slots around the drop centre (c30 ring 0/30/30/35/35 mm);
re-observe after every attempt; at most 3 attempts per cube location.
LIFT2 PLAN (new): poses built in the robot base frame; approach = down, tilted forward by a pitch toward the heading of the target from the base;
closing axis = a cube face normal (minAreaRect yaw of the top points), choosing between the two face pairs by how tangential the finger axis is to
the bowl; a planner replays the facade's 20-waypoint preflight over the whole chain (pre-grasp, grasp, lift, carry, release, retreat) per arm /
face pair / pitch, executes the first feasible chain; arm = the side of the cube (y sign in base frame) first; idle arm parked at home; drop
centre = bowl axis pulled 30 mm toward the robot base (reach), release TCP 45 mm above the rim.'''
import numpy as np
import cv2
from PIL import Image
from rsi import perception as P
from rsi import perception_sam3 as S
from twin.practice.motion import waypoint

ARMS = ('left', 'right')
PITCHES = (0., .3, .5, .7, .9)
RING = (0., .03, .03, .035, .035, .02, .02, .02)


def fatal(e):
    return type(e).__name__ in ('NumericalFailure', 'Limit')


def base_cloud(obs, Tb):
    p, ys, xs = P.points(obs)
    return (p - Tb[:3, 3]) @ Tb[:3, :3], ys, xs


def table_z(pb):
    z = pb[:, 2]; h = np.histogram(z, bins=np.arange(z.min(), z.max() + .002, .002))
    return float(h[1][np.argmax(h[0])] + .001)


def perceive(obs, Tb):
    pb, ys, xs = base_cloud(obs, Tb); tz = table_z(pb); h = pb[:, 2] - tz
    proc = S.get_proc(); st = proc.set_image(Image.fromarray(obs['rgb']))
    bowl = None
    mk, sc = S._prompt(proc, st, 'bowl')
    if len(sc) and sc.max() >= .5:
        q = pb[mk[int(sc.argmax())][ys, xs]]
        if len(q) >= 200:
            top = q[q[:, 2] > np.percentile(q[:, 2], 90)]
            xy, r = P.fit_upright_cylinder(top); bowl = (np.asarray(xy, float), float(r), float(np.percentile(q[:, 2], 97)))
    pairs = []
    for pr in ('small cube', 'block'):
        mk, sc = S._prompt(proc, st, pr); pairs += list(zip(mk, sc))
    cubes = []
    for m, s in sorted(pairs, key=lambda x: -x[1]):
        if s < .3: continue
        sel = m[ys, xs] & (h > .005)
        if sel.sum() < 40: continue
        q = pb[sel]; hq = h[sel]; p95 = float(np.percentile(hq, 95))
        q = q[hq < p95 + .015]; hq = hq[hq < p95 + .015]
        top = q[hq > p95 - .010]; c = np.r_[top[:, :2].mean(0), np.median(top[:, 2])]
        if s < .7 and not (sel.sum() >= 100 and .022 <= p95 <= .040): continue
        if p95 > .060: continue
        if bowl is not None and np.linalg.norm(c[:2] - bowl[0]) < bowl[1] + .01: continue
        if any(np.linalg.norm(c[:2] - x['c'][:2]) < .015 for x in cubes): continue
        bar_k = None; yaw = 0.
        if len(top) >= 8:
            (_, _), (rw, rh), ang = cv2.minAreaRect(top[:, :2].astype(np.float32)); yaw = np.radians(ang)
            if max(rw, rh) > .048: bar_k = 0 if rw < rh else 1  # bar 64x32: close across the short side (rect side k*90 deg from yaw)
        cubes.append({'c': c, 'yaw': float(yaw), 'bar_k': bar_k, 'score': float(s), 'n': int(sel.sum()), 'ext': float((q[:, :2].max(0) - q[:, :2].min(0)).max())})
    return bowl, cubes, tz


def pose(Tb, p_base, heading, pitch, close_dir):
    '''World TCP pose (canonical: +z approach, +y closing). Approach = base -z tilted by pitch toward horizontal heading.'''
    hd = np.array([np.cos(heading), np.sin(heading), 0.])
    a = np.cos(pitch) * np.array([0, 0, -1.]) + np.sin(pitch) * hd
    y = np.asarray(close_dir, float); y = y - (y @ a) * a; y /= np.linalg.norm(y)
    return Tb[:3, :3] @ np.asarray(p_base, float) + Tb[:3, 3], Tb[:3, :3] @ np.stack([np.cross(y, a), y, a], 1)


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


def plan_pick(io, Tb, cube, bowl, drop_xy, rel_z, carry_z, tz, skip=0):
    '''Feasible chains in preference order; returns the (skip)-th one (skip > 0 after an empty grasp).'''
    c = cube['c']; hg = np.arctan2(c[1], c[0]); h = np.array([np.cos(hg), np.sin(hg)])
    faces = []
    for k in ((cube['bar_k'],) if cube.get('bar_k') is not None else (0, 1)):
        th = cube['yaw'] + k * np.pi / 2; d = np.array([np.cos(th), np.sin(th)])
        faces.append((abs(d @ h), d))  # closing axis perpendicular to the approach heading first
    faces.sort(key=lambda f: f[0])
    side = 'left' if c[1] > 0 else 'right'
    order = [side] + [a for a in ARMS if a != side]
    n = 0
    for arm in order:
        for _, d in faces:
            for pitch in PITCHES:
                g = np.r_[c[:2], max(c[2] - .012, tz + .004 + .010 * np.cos(pitch) + .016 * np.sin(pitch))]
                hr = np.arctan2(drop_xy[1], drop_xy[0])
                cd = np.r_[d, 0.]
                rel = np.r_[drop_xy, rel_z]
                ch = [pose(Tb, g + [0, 0, .08], hg, pitch, cd), pose(Tb, g, hg, pitch, cd), pose(Tb, np.r_[g[:2], carry_z], hg, pitch, cd),
                      pose(Tb, np.r_[drop_xy, carry_z], hr, pitch, cd), pose(Tb, rel, hr, pitch, cd), pose(Tb, rel + [0, 0, .05], hr, pitch, cd)]
                if chain_ok(io, arm, ch, io.home_q[ARMS.index(arm)]):
                    if n == skip:
                        return arm, pitch, ch
                    n += 1
    return None


def run(io):
    Tb = np.asarray(io._w.kinematics.T_task_base, float)
    tries = {}; empties = {}; k = 0; bowl0 = None
    for rnd in range(14):
        obs = io.observe()
        bowl, cubes, tz = perceive(obs, Tb)
        if bowl is None: bowl = bowl0
        if bowl is None: raise RuntimeError('bowl not found')
        bowl0 = bowl
        io.log('perceived', round=rnd, bowl=list(bowl[0]) + [bowl[1], bowl[2]], cubes=[(x['c'], x['score'], x['ext']) for x in cubes], table_z=tz)
        cands = [x for x in cubes if tries.get((int(round(x['c'][0] / .03)), int(round(x['c'][1] / .03))), 0) < 3]
        if not cands: break
        bxy = bowl[0]; toward = -bxy / np.linalg.norm(bxy)
        ang = k * 2.4; ring = RING[min(k, len(RING) - 1)] * np.array([np.cos(ang), np.sin(ang)])
        drop_xy = bxy + .03 * toward + ring
        rel_z = bowl[2] + .045; carry_z = bowl[2] + .07
        plan = None
        for cube in sorted(cands, key=lambda x: np.linalg.norm(x['c'][:2] - bxy)):
            key = (int(round(cube['c'][0] / .03)), int(round(cube['c'][1] / .03)))
            plan = plan_pick(io, Tb, cube, bowl, drop_xy, rel_z, carry_z, tz, skip=empties.get(key, 0))
            if plan is None and empties.get(key, 0): plan = plan_pick(io, Tb, cube, bowl, drop_xy, rel_z, carry_z, tz)
            if plan is not None: break
            tries[key] = 3; io.log('no_plan', xy=cube['c'])
        if plan is None: break
        tries[key] = tries.get(key, 0) + 1
        arm, pitch, ch = plan
        io.log('pick', round=rnd, k=k, arm=arm, pitch=pitch, xy=cube['c'], drop=drop_xy, bar_k=cube.get('bar_k'), skip=empties.get(key, 0))
        try:
            (p0, R0), (p1, R1), (p2, R2), (p3, R3), (p4, R4), (p5, R5) = ch
            io.grip(arm, 1., .3)
            io.move(arm, p0, R0, 1.3); io.move(arm, p1, R1, .9)
            io.grip(arm, 0., .6)
            gp = float(io._w.gripper_positions()[ARMS.index(arm)])
            io.log('grip_reading', m=gp)
            if gp < .005:  # closed on nothing (encoder); no carry
                empties[key] = empties.get(key, 0) + 1
                io.grip(arm, 1., .3); io.move(arm, p0, R0, .6); io.home(arm, 1.2)
                continue
            io.move(arm, p2, R2, .9); io.move(arm, p3, R3, 1.3); io.move(arm, p4, R4, .7)
            io.grip(arm, 1., .5)
            io.move(arm, p5, R5, .6)
            k += 1
        except Exception as e:
            if fatal(e): raise
            io.log('pick_failed', err=str(e)[:100])
            try: io.grip(arm, 1., .3)
            except Exception as e2:
                if fatal(e2): raise
        io.home(arm, 1.4)
    io.wait(1.5)
    io.observe()
