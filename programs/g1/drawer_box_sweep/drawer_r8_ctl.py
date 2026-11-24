# drawer_r8_ctl (r8 = r7 + cube orientation: demo/candidate order kept, first with chain margin >= .15 rad (r7 margin-max picked fwdL_down45_lz which did not lift the cube, b3 seed 2); drop x from the opened drawer front (cabinet mask low-x) instead of the re-perceived knob; r7 = r6 + after the pull the CLOSED hand first lifts 6 cm so the thumb clears the drawer front, then opens in the air and backs off (b3 smoke: opening the hand at the knob straightened the fingers and pushed the drawer back from 12.5 cm to ~3 cm); r6 = r5 + cube pick/drop orientation chosen by the largest arm-joint margin over the whole pick-lift-drop chain (b2 DEV0-9: 6/10 failed with right_wrist_roll beyond its limit after the grasp) + recover(): when an encoder reads beyond its range, the public joint target is driven back inside the range and held; r5 = r4 + knob approach from the side/above (yaw x tilt 30 deg, pad axis vertical; physics probe drawer_probe_r1: left yaw -40 opened 12.5 cm, all frontal pinches 0), straight-up release retreat so the arm does not push the drawer back, opened check threshold 4.5 cm perceived; r4 = r3 + knob orientations ranked by worst arm-joint margin from the robot-only model (io.adapter.hand.m joint ranges), min margin .04 rad, planning errors caught per candidate; r3 = r2 + knob target offset so the fixed thumb pad sits on the knob (TCP = knob + pad_axis*(16-6.5) mm), knob re-detected after the pull and up to 3 orientations tried until the knob moved >= 8 cm; r2 = r1 + knob from the mid-height front face: r1 picked the shell top edge at z .155 in seed 0 smoke): G1 open_drawer_and_place_object controller (runtime side: uses only io = rendered RGB-D, calibration, encoders, demo priors, SAM3).
# Shared skill (from xTrainer s1, semantics unchanged): find the cabinet (SAM3 'wooden cabinet'), grasp the drawer knob with the left hand, pull the drawer
# 13 cm toward the robot, release, park; re-observe; find the orange cube by SAM3 prompts outside the cabinet footprint; pick it with the right hand and drop
# it inside the opened drawer; release, retreat, park, wait.
# G1-specific plan (this file): knob position = depth points protruding in front of the SAM3 cabinet front face (no colour rule); knob pinch orientations
# and cube grasp orientations come from a fixed candidate list checked by IK preflight; approach offsets, pull length and drop heights are code constants.
# Contact priors (listed in NOTES.md): demo right close/open rotations are first candidates for the cube pick/drop; knob pinch uses the G1 cube-32 mm pinch frame
# with extra squeeze (trusted-side profile).
import math
import numpy as np
from PIL import Image
from rsi import perception as P, perception_sam3 as S, perception_geo as G

UP = np.array([0., 0., 1.])
PULL = .13
PROMPTS = ('orange cube', 'small cube', 'orange block', 'toy block')
CAB_MARGIN = .02


def Rax(a, l):
    a = np.asarray(a, float); a = a / np.linalg.norm(a)
    l = np.asarray(l, float); l = l - (l @ a) * a; l = l / np.linalg.norm(l)
    return np.column_stack([np.cross(l, a), l, a])


# knob pinch: approach forward and down onto the knob, pad axis across the knob (y); IK-checked in order
KNOB_R = [('fwd_down30_ly', Rax((.866, 0, -.5), (0, 1, 0))), ('fwd_down30_l-y', Rax((.866, 0, -.5), (0, -1, 0))),
          ('fwd_down45_ly', Rax((.707, 0, -.707), (0, 1, 0))), ('fwd_down45_l-y', Rax((.707, 0, -.707), (0, -1, 0))),
          ('fwd_down60_lz', Rax((.5, 0, -.866), (0, 0, 1))), ('fwd_down30_lz', Rax((.866, 0, -.5), (0, 0, 1))),
          ('fwd_ly', Rax((1, 0, 0), (0, 1, 0))), ('fwd_l-y', Rax((1, 0, 0), (0, -1, 0))), ('fwd_lz', Rax((1, 0, 0), (0, 0, 1))),
          ('fwd_down60_lx', Rax((.5, 0, -.866), (1, 0, 0))), ('fwd_down30_lx', Rax((.866, 0, -.5), (1, 0, 0)))]


def cands():
    out = []
    for nm, v in (('down', (0, 0, -1)), ('fwd_down45', (.707, 0, -.707)), ('fwd_down30', (.866, 0, -.5)), ('fwd_down60', (.5, 0, -.866)),
                  ('left_down45', (0, .707, -.707)), ('right_down45', (0, -.707, -.707)), ('fwdL_down45', (.5, .5, -.707)), ('fwdR_down45', (.5, -.5, -.707))):
        for ln, l in (('ly', (0, 1, 0)), ('lx', (1, 0, 0)), ('l-y', (0, -1, 0)), ('l-x', (-1, 0, 0)), ('lz', (0, 0, 1))):
            if abs(np.dot(v, l)) / np.linalg.norm(v) > .9:
                continue
            out.append((nm + '_' + ln, Rax(v, l)))
    return out


ALT = cands()


def reach(io, side, pts, R):
    enc = io._enc()
    for p in pts:
        try:
            io.adapter.plan(enc, side, np.asarray(p, float), R)
        except Exception as e:
            return False, str(e)[:80]
    return True, ''


def cabinet(io, obs, prompt):
    plane = G.fit_plane(obs)
    w = S.locate_object(S.get_proc(), obs, prompt, plane=plane, max_height=.13)
    if w is None:
        w = S.locate_object(S.get_proc(), obs, 'wooden cabinet', plane=plane, max_height=.13)
    if w is None:
        raise RuntimeError('cabinet not found')
    return w, plane


def knob_from_depth(io, obs, cab, plane):
    # front face = robust low-x of the cabinet mask points; knob = all depth points just in front of that face within the cabinet y span, above the table
    p, ys, xs = G.safe_points(obs)
    h = G.height(p, plane)
    cp = cab['points']
    ylo, yhi = np.percentile(cp[:, 1], 3), np.percentile(cp[:, 1], 97)
    ch = G.height(cp, plane)
    ctop = float(np.percentile(ch, 97))
    fp = cp[(ch > .02) & (ch < ctop - .02)]  # front face band, excludes the top surface and its front edge
    face = float(np.percentile(fp[:, 0], 50)) if len(fp) > 50 else float(np.percentile(cp[:, 0], 10))
    sel = (p[:, 1] > ylo + .01) & (p[:, 1] < yhi - .01) & (p[:, 0] < face - .006) & (p[:, 0] > face - .04) & (h > .02) & (h < ctop - .015)
    q = p[sel]
    io.log('knob_points', n=int(len(q)), face_x=face, cab_top_h=ctop, cab_y=[float(ylo), float(yhi)])
    if len(q) < 15:
        return None, face, (ylo, yhi)
    c = np.median(q, 0)
    k = np.linalg.norm(q[:, 1:] - c[1:], axis=1) < .02
    q = q[k]
    knob = np.r_[np.percentile(q[:, 0], 50), np.median(q[:, 1]), np.median(q[:, 2])]
    return knob, face, (ylo, yhi)


def find_block(io, obs, cab_points):
    proc = S.get_proc(); plane = G.fit_plane(obs)
    p, ys, xs = G.safe_points(obs); h = G.height(p, plane)
    state = proc.set_image(Image.fromarray(obs['rgb']))
    lo = cab_points[:, :2].min(0) - CAB_MARGIN; hi = cab_points[:, :2].max(0) + CAB_MARGIN
    diag = []; best = None
    for prompt in PROMPTS:
        masks, scores = S._prompt(proc, state, prompt)
        for m, s in zip(masks, scores):
            if s < .3:
                continue
            sel = m[ys, xs] & (h > .005) & (h < .06)
            n = int(sel.sum())
            if n < 40:
                diag.append({'prompt': prompt, 'score': float(s), 'reject': 'few_points', 'n': n}); continue
            q = p[sel]; hq = h[sel]
            ext = q[:, :2].max(0) - q[:, :2].min(0)
            top = float(np.percentile(hq, 95))
            c = np.r_[np.median(q[:, :2], 0), np.median(q[:, 2])]
            if np.all(c[:2] > lo) and np.all(c[:2] < hi):
                diag.append({'prompt': prompt, 'score': float(s), 'reject': 'inside_cabinet_footprint'}); continue
            if ext.max() > .09 or not (.015 <= top <= .05):
                diag.append({'prompt': prompt, 'score': float(s), 'reject': 'not_block_like', 'extent': ext, 'top': top}); continue
            cand = {'points': q, 'top': float(np.percentile(q[:, 2], 95)), 'center': c, 'score': float(s), 'n': n, 'prompt': prompt}
            diag.append({'prompt': prompt, 'score': float(s), 'accept': True, 'center': c, 'n': n})
            if best is None or cand['score'] > best['score']:
                best = cand
        if best is not None and best['score'] >= .7:
            break
    return best, diag


THUMB_OFF = .016 - .0065  # pinch frame: thumb pad 16 mm from the virtual TCP along -pad axis; knob half width ~6.5 mm (depth extent, code constant)


def recover(io, side, err):
    # an encoder beyond its range makes FK/IK refuse; drive that joint target .06 rad inside its range (public targets, actuators only) and wait
    msg = str(err)
    if 'encoder_out_of_limit:' not in msg:
        return False
    jn = msg.split('encoder_out_of_limit:')[1].split()[0].strip("')\"")
    m = io.adapter.hand.m
    try:
        lo, hi = m.jnt_range[m.joint(jn).id]
    except Exception:
        return False
    v = io.targets.get(jn)
    if v is None:
        return False
    io.targets[jn] = float(min(max(v, lo + .06), hi - .06))
    io.wait(.4)
    io.log('recovered', joint=jn, target=io.targets[jn])
    return True


def safe_move(io, side, p, R, dur, tries=2):
    for k in range(tries):
        try:
            return io.move(side, p, R, dur)
        except Exception as e:
            if io.error is not None:
                raise
            if k + 1 < tries and recover(io, side, e):
                continue
            raise


def ytl(yaw_deg, tilt_deg, l='lz'):
    yaw, tilt = math.radians(yaw_deg), math.radians(tilt_deg)
    a = np.array([math.cos(tilt) * math.cos(yaw), math.cos(tilt) * math.sin(yaw), -math.sin(tilt)])
    perp = np.array([-math.sin(yaw), math.cos(yaw), 0.])
    lv = {'lz': np.array([0., 0., 1.]), 'l-z': np.array([0., 0., -1.]), 'lp': perp, 'l-p': -perp}[l]
    return Rax(a, lv)


# (side, name, R): side approaches with the pad axis vertical; the thumb hooks the drawer front behind the knob (probe evidence)
KNOB_SIDE = [('left', 'L_Y-40_T30_lz', ytl(-40, 30)), ('left', 'L_Y-30_T30_lz', ytl(-30, 30)), ('left', 'L_Y-50_T30_lz', ytl(-50, 30)),
             ('right', 'R_Y40_T30_lz', ytl(40, 30)), ('right', 'R_Y30_T30_lz', ytl(30, 30)), ('left', 'L_Y-30_T45_lz', ytl(-30, 45)),
             ('right', 'R_Y30_T45_lz', ytl(30, 45)), ('left', 'L_Y-40_T15_lz', ytl(-40, 15))]
OPEN_MIN = .045


def pull_once(io, knob, nm, R, side):
    a = R[:, 2]
    g = knob + R[:, 1] * THUMB_OFF
    io.log('knob_grasp', cand=nm, side=side, knob=knob, target=g)
    io.grip(side, 1., .3)
    io.move(side, g - a * .07, R, 1.5)
    io.move(side, g - a * .004, R, 1.0)
    io.grip(side, 0., .7)
    io.wait(.2)
    io.log('knob_closed', cand=nm)
    io.move(side, g + [-PULL, 0, 0], R, 2.5)
    io.log('knob_pulled', cand=nm)
    lifted = False
    for dz in (.06, .045, .03):
        try:
            safe_move(io, side, g + [-PULL, 0, dz], R, 1.0)
            lifted = True
            break
        except Exception as e:
            if io.error is not None:
                raise
            io.log('knob_lift_failed', dz=dz, err=str(e)[:80])
    io.grip(side, 1., .5)
    for off in ([-PULL - .05, 0, .08], [-PULL - .05, 0, .05]):
        try:
            safe_move(io, side, g + off, R, .9)
            break
        except Exception as e:
            if io.error is not None:
                raise
            io.log('knob_retreat_failed', off=off, err=str(e)[:80])
    io.home(side, 2.0)


MIN_MARGIN = .04


def margin_plan(io, side, pts, R):
    # IK through pts from the current encoders; returns the worst distance (rad) of any arm joint to its range (robot-only model), or None if unreachable
    m = io.adapter.hand.m
    enc = io._enc()
    worst = 9.
    for p in pts:
        try:
            q = io.adapter.plan(enc, side, np.asarray(p, float), R)
        except Exception:
            return None
        enc.update(q)
        for n, v in q.items():
            lo, hi = m.jnt_range[m.joint(n).id]
            worst = min(worst, v - lo, hi - v)
    return worst


def open_drawer(io, knob, cab_prompt='wooden cabinet'):
    k0 = knob.copy()
    tried = 0
    for attempt in range(3):
        ranked = []
        for i, (side, nm, R) in enumerate(KNOB_SIDE):
            g = knob + R[:, 1] * THUMB_OFF
            a = R[:, 2]
            w = margin_plan(io, side, [g - a * .07, g, g + [-PULL, 0, 0], g + [-PULL, 0, .07]], R)
            if w is not None and w >= MIN_MARGIN:
                ranked.append((i, w, side, nm, R))
        io.log('knob_ranked', cands=[(x[3], round(float(x[1]), 3)) for x in ranked])
        if len(ranked) <= attempt:
            break
        i, w, side, nm, R = ranked[attempt]  # list order = probe evidence order, margin only filters
        try:
            pull_once(io, knob, nm, R, side)
        except Exception as e:
            if io.error is not None:
                raise
            io.log('pull_failed', cand=nm, err=str(e)[:80])
            try:
                io.grip(side, 1., .3)
                io.home(side, 2.0)
            except Exception:
                if io.error is not None:
                    raise
        tried += 1
        obs = io.observe()
        cab, plane = cabinet(io, obs, cab_prompt)
        k1, face1, _ = knob_from_depth(io, obs, cab, plane)
        moved = None if k1 is None else float(k0[0] - k1[0])
        io.log('drawer_check', cand=nm, knob_now=k1, moved_m=moved)
        if moved is not None and moved >= OPEN_MIN:
            return k1
        if k1 is not None:
            knob = k1
    raise RuntimeError('drawer not opened after %d pulls' % tried)


def run(io):
    ev = io.demo['events']
    rc = next(e for e in ev if e['arm'] == 'right' and e['kind'] == 'close')
    ro = next(e for e in ev if e['arm'] == 'right' and e['kind'] == 'open')
    obs = io.observe()
    cab, plane = cabinet(io, obs, 'wooden cabinet')
    knob, face, ys = knob_from_depth(io, obs, cab, plane)
    if knob is None:
        raise RuntimeError('knob not found in depth')
    io.log('knob', knob=knob, face_x=face)
    knob1 = open_drawer(io, knob)
    obs = io.observe()
    cab2, plane = cabinet(io, obs, 'wooden cabinet with open drawer')
    face2 = float(np.percentile(cab2['points'][:, 0], 3))
    io.log('opened_face', face=face2, opened_m=float(face - face2))
    c, diag = find_block(io, obs, cab['points'])
    io.log('block_candidates', candidates=diag)
    if c is None:
        raise RuntimeError('block not found by SAM3 prompts')
    tz = P.table_height(obs)
    top = c['points'][c['points'][:, 2] > c['top'] - .008]
    t = np.r_[top[:, :2].mean(0), tz + .016]
    io.log('block_perceived', center=c['center'], target=t, prompt=c['prompt'], score=c['score'], table=tz)
    # drop point: centre of the newly opened part of the drawer (between the pulled front and the closed face), at drawer y
    dy = float(knob[1])
    drop_xy = np.r_[0.5 * (max(face2, face - .17) + .015 + face) - .005, dy]  # between the opened drawer front wall and the shell face  # centre of the opened part: between the pulled front wall (behind the knob) and the shell face
    gcands = [('demo_close', rc['rotation'])] + ALT
    dcands = [('demo_open', ro['rotation'])] + ALT
    side = 'right' if t[1] < dy + .05 else 'left'
    ranked = []
    for gn, gR in gcands:
        for hz in (.10, .08, .12):
            d = np.r_[drop_xy, tz + hz]
            w = margin_plan(io, side, [t + .10 * UP, t, t + .15 * UP, d + .05 * UP, d], gR)
            if w is not None:
                ranked.append((w, gn, gR, d))
                break
    good = [x for x in ranked if x[0] >= .15]
    ranked = good + sorted([x for x in ranked if x[0] < .15], key=lambda x: -x[0])
    io.log('cube_ranked', cands=[(x[1], round(float(x[0]), 3)) for x in ranked[:6]])
    if ranked and ranked[0][0] >= .05:
        picked = ranked[0][1:]
    else:
        picked = None
        for gn, gR in gcands:
            w = margin_plan(io, side, [t + .10 * UP, t, t + .12 * UP], gR)
            if w is not None and w >= .05:
                picked = (gn, gR, None); break
    if picked is None:
        raise RuntimeError('cube unreachable')
    gn, gR, d = picked
    io.log('cube_grasp', side=side, cand=gn, drop=d)
    io.grip(side, 1., .3)
    safe_move(io, side, t + .10 * UP, gR, 1.5)
    safe_move(io, side, t, gR, 1.0)
    io.grip(side, 0., .6)
    io.wait(.2)
    for hz in (.15, .12, .08):
        try:
            safe_move(io, side, t + hz * UP, gR, 1.0); break
        except Exception as e:
            if io.error is not None:
                raise
            io.log('lift_retry', h=hz, err=str(e)[:60])
    rel = None
    seq = ([(gn, gR, d)] if d is not None else []) + [(n, R, np.r_[drop_xy, tz + hz]) for hz in (.10, .08, .12) for n, R in dcands]
    for n, R, dd in seq:
        try:
            safe_move(io, side, dd + .05 * UP, R, 1.5)
            safe_move(io, side, dd, R, .8)
            rel = (n, R, dd); break
        except Exception as e:
            if io.error is not None:
                raise
            io.log('drop_retry', cand=n, err=str(e)[:60])
    if rel is None:
        raise RuntimeError('drop unreachable')
    io.log('drop', cand=rel[0], at=rel[2])
    io.grip(side, 1., .6)
    for off in ([0, 0, .08], [0, 0, .05], [-.05, 0, .05]):
        try:
            io.move(side, rel[2] + np.array(off), rel[1], .8); break
        except Exception as e:
            if io.error is not None:
                raise
            io.log('retreat_retry', off=off)
    io.home(side, 2.0)
    io.wait(1.0)
    io.observe()
