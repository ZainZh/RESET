# g1_common_tc25 = g1_common_tc24a + pick_place_plan() requires the robot-model hand surface points (hand_pts_r2_towc_grasp_cube.npz; open hand along pregrasp->grasp, closed at grasp) to keep >= HC_MIN = 10 mm from the PERCEIVED obstacle points set by the controller in OBST (SAM3 depth points, extruded down to the table); pass order: clear+chain-continuous, clear only, then the tc24a fallback (logged plan_noclear). Evidence: towc_tc24a_diagbar_b3 seeds 15/12 - left middle/ring fingers pushed the bar during approach/closing (24-31 mm); towc_tchc_DIAG_notscored - chosen plan clearance 0 mm on failing seeds 15/12/16, 28-105 mm on passing seeds 0/2/13, and 20-45 clear alternatives exist on every seed.
# g1_common_tc24a = g1_common_tc23 + ONLY the slow still release (arm holds 0.5 s, fingers open over OPEN_T=2.0 s, 0.3 s wait). CLEAR unchanged (4 mm).
# g1_common_tc21 = g1_common_tc20 + after the joint-space reverse path (q_place -> q_above -> q_carry) the hand goes to the raised park point with the branch-continuous cmove and only then io.home; if that cmove fails the hand stays at q_carry (no joint-space home sweep, logged home_skipped). Evidence: towc_tc20_diag_b3 seed 2 - the purple cube stayed on the bar through reverse_carry (t 14.4) and was knocked to the table by the following io.home sweep (t 16.9).
# g1_common_tc20 = g1_common_tc19 + place(): the commanded arm joints are recorded when the carry starts (q_carry) and when the hand is above the destination (q_above); after release the hand returns in JOINT space along the reverse of its own path, q_place -> q_above -> q_carry, then io.home. The Cartesian retreat/pull-back/raised-park chain is removed. Evidence: towc_tc19_dev0_19_b3 seeds 1,4,6,8,19 - every post-release Cartesian chain hit ik_branch_jump, then io.home swept the stack.
# g1_common_tc19 = g1_common_tc18 + if the branch-continuous move to the raised park point fails after release, the empty hand first rises 10 cm straight up (cmove) and then goes to the raised park point with pmove (io.move fallback allowed, hand empty and >= 26 cm above the support) before io.home. Evidence: towc_tc18_smoke_b3 seed 4 - park_high_failed (ik_branch_jump) then io.home's joint-space sweep from 16 cm knocked the placed purple cube off the bar.
# g1_common_tc18 = g1_common_tc17 + after release (retreat, pull-back, raised park point) only the branch-continuous cmove is used, never the live re-solved io.move fallback (each failure falls through to the next option, last resort io.home). Evidence: towc_tc17_smoke_b3 seed 11 - pmove_fallback on the pull-back after the yellow release, yellow cube ended 9 cm off the stack.
# g1_common_tc15 = g1_common_tc14 + pick_place_plan() prefers (dest, arm, orientation) whose whole chain pregrasp -> grasp -> lift -> carry -> above -> place -> retreat is branch-continuous (IK chained from the commanded targets in <= 1 cm steps, no joint change > JUMP per step); if no candidate passes, the r11 plan (reach only) is used and 'plan_nochain' is logged. Evidence: towc_tc14_smoke_b3 seeds 0/4/9/11 - the planned 50-70 deg orientations have a joint jump on the 6 cm place descent itself (pmove_fallback), so the place phase cannot be made continuous at execution time. tc14 was: baseline g1_common_r11 + cmove()/JOINT_RATE/JUMP copied from g1_common_r13 + pmove(): every place-phase move (carry above, via, descent, retreat, pull-back, raised park point) uses the branch-continuous cmove, falling back to io.move if the joint chain jumps. Planning is unchanged from r11 (no r13 chain_ok / redest). Evidence: towc_r11_diagmove_b3 seed 0 - the live re-solved IK in io.move swung shoulder yaw/wrist roll ~0.6 rad in 0.25 s 4 mm above the place point and carried the cube away (residual 0.155 m); same pattern (place residual 0.10-0.20 m) on DEV seeds 0, 4, 9, 11 of the r11 clean-park run.
# g1_common_r11: identical copy of g1_common_d4 (margin 0 + trusted float-level arm clamp <= 0.002 rad, carry-segment preflight, via-point approach, raised park point, squeeze 0.05), paired with ctl_build_tower_r11. Evidence: diag_margin_plan_r9*.log (margin 0.02 leaves no plan on 6/20 tower layouts).
# g1_common_d4: g1_common_d3 + squeeze-0.05 close in pick() via the public adapter hand_targets (same as g1_common_r10).
# g1_common_d2 (d1 + pick_place_plan(sides=...) restricts the arm; diag_drawer_phys_d1r trace seed 6: a left-arm fallback for the block swept the drawer closed). g1_common_d1 was: g1_common_m6 + open_drawer_and_place_object grasp-profile prior ('cube', 0.032) (orange cube 32 mm; the same pinch closes on the 14x15x15 mm knob, evidence diag_drawer_pull_r1.log). Paired with ctl_open_drawer_and_place_object_d1.
# g1_common_m6 (m5 + place(home=False) skips the final joint-space home and holds at the raised park point; diag_mug_phys_m5 trace seeds 3,5: mug on the box with 3 deg tilt after park_high, swept off only by the joint-space home). g1_common_m5 was: (m4 + pick(): if the straight park->pregrasp move fails IK on the way, retry via raised via-points (pregrasp+8 cm, midpoint+8 cm); diag_mug_phys_m4.log seeds 1-3: grasp_approach_failed ik_invalid:position_residual although the endpoints passed the preflight). g1_common_m4 was: (m3 + MARGIN 0 + trusted-side ARM encoder read clamp only for overshoot <= ARM_CLAMP_MAX = 0.002 rad (larger excursions pass through raw and still fail), logged under CLAMP['arm']; evidence diag_mug_phys_m3.log: mug placement plans need arm joints at their limits (any margin removes all plans) and physics overshoots the wrist-pitch limit at float level, cf. diag_place_enc_r9_seed0.log 3e-5 rad on wrist roll). g1_common_m3 was: (m2 + joint-limit margin 0.02 rad in reach() as in g1_common_r9; pick(sides=...) restricts to the planned arm; after the pull-back, a Cartesian move to a point 15 cm above the recorded park TCP before the joint-space home; m2 diag2 seed 0 evidence: the mug was on the box with 3 deg tilt at pull-back, then the joint-space home swept it off the table). g1_common_m2 was: (m1 + pick_place_plan also preflights 8 samples on the straight carry segment lift->above-dest; place() falls back to a raised via-point; m1 seed 0 evidence: place_unreachable on the carry line after a held grasp). g1_common_m1 was: g1_common_r7 + put_mug_on_box grasp-profile prior ('sphere', 0.084): mug body diameter 84 mm from the source m1 rim-cylinder radius 0.042; physical evidence diag_grasp_mug_r7_seed0.log (45 deg tilted grasp at mug mid-height lifts it upright). Paired with ctl_put_mug_on_box_m1.
# g1_common_r6 (identical copy of g1_common_r5, paired with ctl_*_r6) — g1_common_r5 (r4 + tcands() keeps only pinch frames whose pad axis is near-horizontal (|lateral_z| <= 0.35) and adds tilts 50/55/70; evidence diag_tilt_grasp_r4_*.log: roll 0/180 tilted pinches fail to lift, roll 90/270 lift at 30-60 deg. r4 was: pre-grasp backed off along the approach axis instead of straight above; after release: straight up, elevated pull-back, then park; r3 evidence: tilted vertical descent plowed the cube (seed 0) and the joint-space home sweep knocked a placed cube off the bar (seed 2)): shared helpers for the six non-apple/non-cubes b2 DEV baselines (build_tower, stack_cups, put_mug_on_box, open_drawer_and_place_object, sweep_objects, lift_box).
# Does not edit runtime/robot_adapter.py: the grasp-profile contact priors are installed in-process by install_profiles() on the trusted side and are disclosed per task.
# Contact priors (hand v0.4 virtual planning grasp frame, finger_targets kind/size): build_tower = cube 32 mm (all three blocks are 32 mm wide, the bar is 64x32x32 mm and is gripped across 32 mm).
# Float-level finger encoder undershoot of the 0.0 lower limit is clamped into jnt_range on the trusted side at the read and logged (before and after values). Arm encoders are not clamped. Hand model, tolerance, scorer and criteria are untouched.
# Disclosure: G1 pelvis, legs and waist are fixed (no freejoint); robot-body gravcomp is in the world. This is not walking or balance. DEV seeds only. No VAL, TEST or hardware.
import math
import numpy as np
import mujoco as mj

PARK = (0.0, 0.6, 0.0, 0.3, 0.0, 0.0, 0.0)
OPEN_T = 2.0
HC_MIN = 0.010
OBST = None
import os as _os
from scipy.spatial import cKDTree as _KD
HP = np.load(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'hand_pts_r2_towc_grasp_cube.npz'))


def set_obstacle(points, zt):
    # points: (N,3) perceived obstacle depth points (source frame); the obstacle is their xy footprint extruded from the table up to their 95th-percentile height
    global OBST
    if points is None or not len(points):
        OBST = None
        return
    P = np.asarray(points, float)
    OBST = dict(tree=_KD(P[:, :2]), top=float(np.percentile(P[:, 2], 95)), zt=float(zt))


def hand_clear(side, R, c, hover=.10):
    if OBST is None:
        return 9.0
    pre = pregrasp(c, R, hover)
    path = [pre + (np.asarray(c, float) - pre) * k / 6.0 for k in range(0, 7)] + [np.asarray(c, float)]
    best = 9.0
    for k, p in enumerate(path):
        P = p + HP['%s_%d' % (side, 1 if k < len(path) - 1 else 0)] @ np.asarray(R).T
        P = P[(P[:, 2] < OBST['top'] + .03) & (P[:, 2] > OBST['zt'] - .01)]
        if not len(P):
            continue
        dxy, _ = OBST['tree'].query(P[:, :2])
        best = min(best, float(np.min(np.hypot(dxy, np.maximum(P[:, 2] - OBST['top'], 0)))))
    return best
JN = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']
HAND_KEYS = ('thumb', 'index', 'middle', 'ring', 'pinky')
PROFILE_PRIORS = {'build_tower': ('cube', 0.032), 'put_mug_on_box': ('sphere', 0.084), 'open_drawer_and_place_object': ('cube', 0.032)}
CLAMP = {'n_reads': 0, 'n_changed': 0, 'max_below_limit': {}, 'max_above_limit': {}, 'first_events': [], 'limits': {}}


def install_profiles(ra):
    for k, v in PROFILE_PRIORS.items():
        ra.GRASP_PROFILE[k] = v


def install_clamp(cb, model):
    lim = {}
    for i in range(model.njnt):
        nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if nm and any(k in nm for k in HAND_KEYS) and int(model.jnt_limited[i]):
            lim[nm] = (float(model.jnt_range[i][0]), float(model.jnt_range[i][1]))
    CLAMP['limits'] = {k: list(v) for k, v in lim.items()}
    alim = {}
    for i in range(model.njnt):
        nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if nm and any(nm.endswith('_%s_joint' % j) for j in JN) and int(model.jnt_limited[i]):
            alim[nm] = (float(model.jnt_range[i][0]), float(model.jnt_range[i][1]))
    CLAMP['arm'] = {'max_clamp': ARM_CLAMP_MAX, 'n_changed': 0, 'max_overshoot': {}, 'passed_raw_over_max': {}, 'first_events': []}
    orig = cb.encoders

    def enc():
        v = orig()
        CLAMP['n_reads'] += 1
        for n, (lo, hi) in lim.items():
            if n not in v:
                continue
            x = float(v[n])
            y = min(max(x, lo), hi)
            if y != x:
                CLAMP['n_changed'] += 1
                if x < lo:
                    CLAMP['max_below_limit'][n] = max(CLAMP['max_below_limit'].get(n, 0.0), lo - x)
                else:
                    CLAMP['max_above_limit'][n] = max(CLAMP['max_above_limit'].get(n, 0.0), x - hi)
                if len(CLAMP['first_events']) < 40:
                    CLAMP['first_events'].append({'t': float(cb.time()), 'joint': n, 'before': repr(x), 'after': repr(y), 'lo': lo, 'hi': hi})
                v[n] = y
        A = CLAMP['arm']
        for n, (lo, hi) in alim.items():
            if n not in v:
                continue
            x = float(v[n])
            e = max(lo - x, x - hi)
            if e <= 0:
                continue
            if e > ARM_CLAMP_MAX:
                A['passed_raw_over_max'][n] = max(A['passed_raw_over_max'].get(n, 0.0), e)
                continue
            y = min(max(x, lo), hi)
            A['n_changed'] += 1
            A['max_overshoot'][n] = max(A['max_overshoot'].get(n, 0.0), e)
            if len(A['first_events']) < 40:
                A['first_events'].append({'t': float(cb.time()), 'joint': n, 'before': repr(x), 'after': repr(y), 'lo': lo, 'hi': hi})
            v[n] = y
        return v
    cb.encoders = enc
    return lim


def set_park_pose(h):
    m_, d_ = h.model, h.data
    for side, sg in (('left', 1.0), ('right', -1.0)):
        v = [PARK[0], sg * PARK[1], sg * PARK[2], PARK[3], sg * PARK[4], PARK[5], sg * PARK[6]]
        for n, x in zip(JN, v):
            jn = '%s_%s_joint' % (side, n)
            d_.qpos[m_.jnt_qposadr[mj.mj_name2id(m_, mj.mjtObj.mjOBJ_JOINT, jn)]] = x
            d_.ctrl[mj.mj_name2id(m_, mj.mjtObj.mjOBJ_ACTUATOR, 'a_' + jn)] = x
    mj.mj_forward(m_, d_)


def cands():
    out = []
    for nm, v in (('top_down', (0, 0, -1)), ('fwd_down45', (.7071067812, 0, -.7071067812)), ('fwd_horizontal', (1, 0, 0)), ('from_left_down45', (0, -.7071067812, -.7071067812)), ('from_right_down45', (0, .7071067812, -.7071067812))):
        a = np.array(v, float)
        a /= np.linalg.norm(a)
        ref = np.array([0., 0., 1.]) if abs(a[2]) < .9 else np.array([0., 1., 0.])
        l0 = ref - (ref @ a) * a
        l0 /= np.linalg.norm(l0)
        l1 = np.cross(a, l0)
        for k in range(4):
            th = k * math.pi / 2.0
            l = math.cos(th) * l0 + math.sin(th) * l1
            l /= np.linalg.norm(l)
            out.append(('%s_roll%d' % (nm, 90 * k), np.column_stack([np.cross(l, a), l, a])))
    return out


def park(io, duration=2.5):
    vals = {}
    for side, sg in (('left', 1.0), ('right', -1.0)):
        v = [PARK[0], sg * PARK[1], sg * PARK[2], PARK[3], sg * PARK[4], PARK[5], sg * PARK[6]]
        for n, x in zip(JN, v):
            vals['%s_%s_joint' % (side, n)] = x
    q0 = {k: io.targets[k] for k in vals}
    n = max(1, int(math.ceil(duration / io.dt)))
    for k in range(n):
        for nm, x in vals.items():
            io.targets[nm] = q0[nm] + (x - q0[nm]) * (k + 1) / n
        io._step()
    for nm, x in vals.items():
        io.home_q[nm] = x
    io.wait(0.5)
    for sd in ('left', 'right'):
        try:
            PARK_TCP[sd] = io.tcp(sd)[:3, 3].copy()
        except Exception:
            pass


def rot_tilt(tilt, yaw, roll):
    # pinch frame whose approach axis is tilted forward (+x source) from straight down by tilt deg, swung by yaw deg about vertical, rolled by roll deg
    t, y = math.radians(tilt), math.radians(yaw)
    a = np.array([math.sin(t) * math.cos(y), math.sin(t) * math.sin(y), -math.cos(t)])
    ref = np.array([math.cos(y), math.sin(y), 0.])
    l0 = ref - (ref @ a) * a
    if np.linalg.norm(l0) < 1e-6:
        l0 = np.array([1., 0., 0.])
    l0 /= np.linalg.norm(l0)
    l1 = np.cross(a, l0)
    th = math.radians(roll)
    l = math.cos(th) * l0 + math.sin(th) * l1
    return np.column_stack([np.cross(l, a), l, a])


def tcands(tilts=(0, 15, 30, 45, 50, 55, 60, 70), yaws=(0, 30, -30), rolls=(0, 90, 180, 270), lat_z=.35):
    # code-constant orientation set ordered from most vertical to most tilted; pad (lateral) axis must stay near-horizontal
    out = []
    for t in tilts:
        for y in yaws:
            for r in rolls:
                R = rot_tilt(t, y, r)
                if abs(R[2, 1]) <= lat_z:
                    out.append(('tilt%d_yaw%d_roll%d' % (t, y, r), R))
    return out


def pregrasp(c, R, hover):
    # back off from the grasp point along the approach axis (column 2), never below the grasp height + 2 cm
    p = np.asarray(c, float) - hover * np.asarray(R)[:, 2]
    p[2] = max(p[2], c[2] + .02)
    return p


def _pick_place_plan_r11(io, c, dests, cand_list, hover=.10, lift=.15, above=.06, retreat=.10, prefer=None, sides=None):
    # (r11 original, unused in tc15) kinematic preflight only (no motion): first (dest, arm, orientation) such that the same orientation reaches hover/grasp/lift at c and above/place/retreat at dest.
    # cand_list is ordered by preference; both arms are tried for each candidate before moving to the next, the arm on the object side first.
    c = np.asarray(c, float)
    first = prefer or ('left' if c[1] > midline(io) else 'right')
    order = tuple(sides) if sides else (first, 'right' if first == 'left' else 'left')
    n = 0
    for di, d in enumerate(dests):
        d = np.asarray(d, float)
        for gname, gR in cand_list:
            for side in order:
                n += 1
                a0, a1 = c + [0, 0, lift], d + [0, 0, above]
                seg = [a0 + (a1 - a0) * k / 9.0 for k in range(1, 9)]
                ok, _ = reach(io, side, [pregrasp(c, gR, hover), c, a0, a1, d, d + [0, 0, retreat]] + seg, gR)
                if ok:
                    return side, gname, gR, d, di, n
    return None


def midline(io):
    return 0.5 * (io.demo['home']['left'][1, 3] + io.demo['home']['right'][1, 3])


MARGIN = 0.0
ARM_CLAMP_MAX = 0.002
PARK_TCP = {}


def reach(io, side, pts, R):
    rng = io.adapter.hand.rng
    for p in pts:
        try:
            q = io.adapter.plan(io._enc(), side, np.asarray(p, float), R)
        except Exception as e:
            return False, str(e)[:100]
        for n, v in q.items():
            lo, hi = rng[n]
            if v < lo + MARGIN or v > hi - MARGIN:
                return False, 'near_limit:' + n
    return True, ''


def approach(io, side, p, R, tag):
    # move to p with orientation R; on an IK failure along the straight path, retry through raised via-points
    try:
        io.move(side, p, R, 1.5)
        return
    except Exception as e0:
        if io.error is not None:
            raise
        err = e0
    p0 = io.tcp(side)[:3, 3]
    for via in (p + [0, 0, .08], (p0 + p) / 2 + [0, 0, .08], (p0 + p) / 2 + [0, 0, .15]):
        try:
            io.move(side, via, R, 1.2)
            io.move(side, p, R, 1.0)
            io.log('approach_via', tag=tag, via=via)
            return
        except Exception as e:
            if io.error is not None:
                raise
            err = e
    raise err


SQUEEZE = 0.05


def grip_squeeze(io, side, sq, duration=.6):
    # ramp the six finger drives from the current targets to the adapter's closed+squeeze targets; public robot interface only
    tgt = io.adapter.hand_targets(io.task, side, 0.0, squeeze=sq)
    q0 = {k: io.targets[k] for k in tgt}
    n = max(1, int(np.ceil(duration / io.dt)))
    for k in range(n):
        for nm, v in tgt.items():
            io.targets[nm] = q0[nm] + (v - q0[nm]) * (k + 1) / n
        io._step()
    io.openings[side] = 0.0


def pick(io, c, cand_list, tag, hover=.10, lift=.15, prefer=None, dest=None, above=.06, retreat=.10, sides=None):
    # grasp the object whose pinch-frame target is c (source frame): try both arms and all candidate orientations that pass the kinematic preflight.
    # r2: when dest is given, the same orientation must also pass the preflight at dest+above, dest and dest+retreat, so the held object is never re-oriented between pick and place.
    c = np.asarray(c, float)
    first = prefer or ('left' if c[1] > midline(io) else 'right')
    order = tuple(sides) if sides else (first, 'right' if first == 'left' else 'left')
    skipped = 0
    pts = [c + [0, 0, hover], c, c + [0, 0, lift]]  # used only when no orientation-specific pregrasp; see per-candidate list below
    if dest is not None:
        d = np.asarray(dest, float)
        pts += [d + [0, 0, above], d, d + [0, 0, retreat]]
    for side in order:
        for gname, gR in cand_list:
            pg = pregrasp(c, gR, hover)
            ok, err = reach(io, side, [pg] + pts[1:], gR)
            if not ok:
                skipped += 1
                continue
            try:
                io.grip(side, 1., .3)
                approach(io, side, pg, gR, tag)
                io.move(side, c, gR, 1.2)
            except Exception as e:
                if io.error is not None:
                    raise
                io.log('grasp_approach_failed', tag=tag, side=side, cand=gname, err=str(e)[:80])
                continue
            io.log('at_grasp', tag=tag, side=side, cand=gname, target=c)
            grip_squeeze(io, side, SQUEEZE, .6)
            io.log('closed', tag=tag, side=side, squeeze=SQUEEZE)
            for h in (lift, .10, .06):
                try:
                    io.move(side, c + [0, 0, h], gR, 1.5)
                    io.log('picked', tag=tag, side=side, cand=gname, skipped_ik=skipped, lift=h)
                    return side, gname, gR
                except Exception as e:
                    if io.error is not None:
                        raise
                    io.log('lift_retry', tag=tag, h=h, err=str(e)[:80])
            io.grip(side, 1., .6)
            io.log('lift_failed', tag=tag, side=side, cand=gname)
    raise RuntimeError('no_grasp:' + tag + ' skipped_ik=%d' % skipped)


JOINT_RATE = 2.5
JUMP = 0.3


def cmove(io, side, position, rotation, duration=1.5):
    # public robot interface only (io.tcp, io.adapter.plan, io.targets, io._step); raises before any motion if a waypoint is unreachable
    from rsi.g1_suite.runtime.public_io import waypoint
    names = ['%s_%s_joint' % (side, a) for a in JN]
    start = io.tcp(side)
    p, R = np.asarray(position, float), np.asarray(rotation, float)
    enc = io._enc()
    for a in names:
        enc[a] = io.targets[a]
    chain = []
    prev = [enc[a] for a in names]
    for k in range(1, 21):
        wp, wr = waypoint(start, p, R, k / 20.0)
        q = io.adapter.plan(enc, side, wp, wr)
        enc.update(q)
        cur = [q[a] for a in names]
        if max(abs(x - y) for x, y in zip(cur, prev)) > JUMP:
            raise RuntimeError('ik_branch_jump')
        prev = cur
        chain.append(cur)
    q0 = [io.targets[a] for a in names]
    chain = [q0] + chain
    n = max(1, int(math.ceil(float(duration) / io.dt)))
    for k in range(n):
        f = min(1.0, (k + 1) / n / 0.8) * 20
        i = min(19, int(f))
        u = f - i
        des = [chain[i][j] + (chain[i + 1][j] - chain[i][j]) * u for j in range(len(names))]
        for j, a in enumerate(names):
            io.targets[a] += float(np.clip(des[j] - io.targets[a], -JOINT_RATE * io.dt, JOINT_RATE * io.dt))
        io._step()
    return float(np.linalg.norm(io.tcp(side)[:3, 3] - p))


def pmove(io, side, position, rotation, duration=1.5, tag=''):
    # place-phase move: branch-continuous cmove (joint chain fixed before motion, no live re-solve); if the chain has a joint jump > JUMP it falls back to io.move (r11 behaviour); the fallback is logged
    try:
        return cmove(io, side, position, rotation, duration)
    except RuntimeError as e:
        if io.error is not None or 'ik_branch_jump' not in str(e):
            raise
        io.log('pmove_fallback', tag=tag, target=np.asarray(position, float))
        return io.move(side, position, rotation, duration)


def chain_fine(io, side, pts, R, step=.01):
    # kinematic only: IK chained from the commanded arm targets through pts in <= step metre sub-steps; False on IK failure or any joint change > JUMP in one sub-step
    names = ['%s_%s_joint' % (side, a) for a in JN]
    enc = io._enc()
    for a in names:
        enc[a] = io.targets[a]
    try:
        q = io.adapter.plan(enc, side, np.asarray(pts[0], float), R)
    except Exception:
        return False
    enc.update(q)
    prev = np.array([q[a] for a in names])
    for p0, p1 in zip(pts[:-1], pts[1:]):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        m = max(1, int(math.ceil(np.linalg.norm(p1 - p0) / step)))
        for k in range(1, m + 1):
            try:
                q = io.adapter.plan(enc, side, p0 + (p1 - p0) * k / m, R)
            except Exception:
                return False
            enc.update(q)
            cur = np.array([q[a] for a in names])
            if np.abs(cur - prev).max() > JUMP:
                return False
            prev = cur
    return True


def pick_place_plan(io, c, dests, cand_list, hover=.10, lift=.15, above=.06, retreat=.10, prefer=None, sides=None):
    # tc25: pass 1 hand-clear (>= HC_MIN from OBST) and branch-continuous; pass 2 hand-clear only; pass 3 tc24a order (first reachable), logged plan_noclear
    c = np.asarray(c, float)
    first = prefer or ('left' if c[1] > midline(io) else 'right')
    order = tuple(sides) if sides else (first, 'right' if first == 'left' else 'left')
    n = 0
    first_any = first_clear = None
    for di, d in enumerate(dests):
        d = np.asarray(d, float)
        for gname, gR in cand_list:
            for side in order:
                n += 1
                a0, a1 = c + [0, 0, lift], d + [0, 0, above]
                seg = [a0 + (a1 - a0) * k / 9.0 for k in range(1, 9)]
                ok, _ = reach(io, side, [pregrasp(c, gR, hover), c, a0, a1, d, d + [0, 0, retreat]] + seg, gR)
                if not ok:
                    continue
                cand = (side, gname, gR, d, di, n)
                if first_any is None:
                    first_any = cand
                hc = hand_clear(side, gR, c, hover)
                if hc < HC_MIN:
                    continue
                if first_clear is None:
                    first_clear = cand + (hc,)
                if chain_fine(io, side, [pregrasp(c, gR, hover), c, a0, a1, d, d + [0, 0, retreat]], gR):
                    io.log('plan_chain_ok', cand=gname, side=side, n_checked=n, hand_clear=hc)
                    return cand
    if first_clear is not None:
        io.log('plan_nochain', cand=first_clear[1], side=first_clear[0], n_checked=n, hand_clear=first_clear[6])
        return first_clear[:6]
    if first_any is not None:
        io.log('plan_noclear', cand=first_any[1], side=first_any[0], n_checked=n)
    return first_any


def jmove(io, side, q, duration=1.5):
    # joint-space interpolation of the commanded arm targets to q (dict), with the same slew limit as cmove
    names = ['%s_%s_joint' % (side, a) for a in JN]
    q0 = [io.targets[a] for a in names]
    n = max(1, int(math.ceil(float(duration) / io.dt)))
    for k in range(n):
        f = min(1.0, (k + 1) / n / 0.8)
        for j, a in enumerate(names):
            des = q0[j] + (q[a] - q0[j]) * f
            io.targets[a] += float(np.clip(des - io.targets[a], -JOINT_RATE * io.dt, JOINT_RATE * io.dt))
        io._step()


def place(io, side, rot_list, d, tag, above=.06, retreat=.10, home=True):
    # carry to d (source frame) with the first orientation in rot_list that passes the preflight, release, retreat, return to the fixed park pose
    d = np.asarray(d, float)
    last = None
    _nm = ['%s_%s_joint' % (side, a) for a in JN]
    q_carry = {a: io.targets[a] for a in _nm}
    for rname, rR in rot_list:
        try:
            try:
                res = pmove(io, side, d + [0, 0, above], rR, 2.0, tag)
            except Exception as e0:
                if io.error is not None:
                    raise
                p0 = io.tcp(side)[:3, 3]
                via = (p0 + d + [0, 0, above]) / 2 + [0, 0, .05]
                io.log('carry_via', tag=tag, err=str(e0)[:60], via=via)
                pmove(io, side, via, rR, 1.2, tag)
                res = pmove(io, side, d + [0, 0, above], rR, 1.2, tag)
            io.log('above_dest', tag=tag, residual=res, target=d)
            q_above = {a: io.targets[a] for a in _nm}
            res2 = pmove(io, side, d, rR, 1.2, tag)
            io.wait(.3)
        except Exception as e:
            if io.error is not None:
                raise
            last = e
            continue
        io.log('place_selection', tag=tag, cand=rname, residual_above=res, residual_place=res2)
        io.wait(.5)
        io.grip(side, 1., OPEN_T)
        io.wait(.3)
        jmove(io, side, q_above, 1.2)
        io.log('reverse_above', tag=tag)
        jmove(io, side, q_carry, 2.0)
        io.log('reverse_carry', tag=tag)
        if not home:
            return rname
        try:
            cmove(io, side, PARK_TCP[side] + [0, 0, .15], rR, 1.5)
            io.log('park_high', tag=tag)
        except Exception as e:
            if io.error is not None:
                raise
            io.log('home_skipped', tag=tag, err=str(e)[:60])
            return rname
        try:
            io.home(side, 2.5)
        except Exception as e:
            if io.error is not None:
                raise
            io.log('home_failed', tag=tag, err=str(e)[:80])
        return rname
    raise RuntimeError('place_unreachable:' + tag + ' ' + str(last)[:80])


def J(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)
