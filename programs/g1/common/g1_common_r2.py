# g1_common_r2 (r1 + pick() preflights the same orientation at the place waypoints; place() keeps that orientation): shared helpers for the six non-apple/non-cubes b2 DEV baselines (build_tower, stack_cups, put_mug_on_box, open_drawer_and_place_object, sweep_objects, lift_box).
# Does not edit runtime/robot_adapter.py: the grasp-profile contact priors are installed in-process by install_profiles() on the trusted side and are disclosed per task.
# Contact priors (hand v0.4 virtual planning grasp frame, finger_targets kind/size): build_tower = cube 32 mm (all three blocks are 32 mm wide, the bar is 64x32x32 mm and is gripped across 32 mm).
# Float-level finger encoder undershoot of the 0.0 lower limit is clamped into jnt_range on the trusted side at the read and logged (before and after values). Arm encoders are not clamped. Hand model, tolerance, scorer and criteria are untouched.
# Disclosure: G1 pelvis, legs and waist are fixed (no freejoint); robot-body gravcomp is in the world. This is not walking or balance. DEV seeds only. No VAL, TEST or hardware.
import math
import numpy as np
import mujoco as mj

PARK = (0.0, 0.6, 0.0, 0.3, 0.0, 0.0, 0.0)
JN = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']
HAND_KEYS = ('thumb', 'index', 'middle', 'ring', 'pinky')
PROFILE_PRIORS = {'build_tower': ('cube', 0.032)}
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


def midline(io):
    return 0.5 * (io.demo['home']['left'][1, 3] + io.demo['home']['right'][1, 3])


def reach(io, side, pts, R):
    for p in pts:
        try:
            io.adapter.plan(io._enc(), side, np.asarray(p, float), R)
        except Exception as e:
            return False, str(e)[:100]
    return True, ''


def pick(io, c, cand_list, tag, hover=.10, lift=.15, prefer=None, dest=None, above=.06, retreat=.10):
    # grasp the object whose pinch-frame target is c (source frame): try both arms and all candidate orientations that pass the kinematic preflight.
    # r2: when dest is given, the same orientation must also pass the preflight at dest+above, dest and dest+retreat, so the held object is never re-oriented between pick and place.
    c = np.asarray(c, float)
    first = prefer or ('left' if c[1] > midline(io) else 'right')
    order = (first, 'right' if first == 'left' else 'left')
    skipped = 0
    pts = [c + [0, 0, hover], c, c + [0, 0, lift]]
    if dest is not None:
        d = np.asarray(dest, float)
        pts += [d + [0, 0, above], d, d + [0, 0, retreat]]
    for side in order:
        for gname, gR in cand_list:
            ok, err = reach(io, side, pts, gR)
            if not ok:
                skipped += 1
                continue
            try:
                io.grip(side, 1., .3)
                io.move(side, c + [0, 0, hover], gR, 1.5)
                io.move(side, c, gR, 1.0)
            except Exception as e:
                if io.error is not None:
                    raise
                io.log('grasp_approach_failed', tag=tag, side=side, cand=gname, err=str(e)[:80])
                continue
            io.log('at_grasp', tag=tag, side=side, cand=gname, target=c)
            io.grip(side, 0., .6)
            io.log('closed', tag=tag, side=side)
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


def place(io, side, rot_list, d, tag, above=.06, retreat=.10):
    # carry to d (source frame) with the first orientation in rot_list that passes the preflight, release, retreat, return to the fixed park pose
    d = np.asarray(d, float)
    last = None
    for rname, rR in rot_list:
        try:
            res = io.move(side, d + [0, 0, above], rR, 2.0)
            io.log('above_dest', tag=tag, residual=res, target=d)
            res2 = io.move(side, d, rR, 1.2)
            io.wait(.3)
        except Exception as e:
            if io.error is not None:
                raise
            last = e
            continue
        io.log('place_selection', tag=tag, cand=rname, residual_above=res, residual_place=res2)
        io.grip(side, 1., .6)
        try:
            io.move(side, d + [0, 0, retreat], rR, .8)
        except Exception as e:
            if io.error is not None:
                raise
            io.log('retreat_failed', tag=tag, err=str(e)[:80])
        try:
            io.home(side, 2.0)
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
