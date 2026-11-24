# ctl_build_tower_tc27 = tc26 but the stack re-measure uses the raw SAM3 'purple cube' masks (rsi.perception_sam3._prompt, read-only) with depth points 45-75 mm above the fitted table plane and within 30 mm of the purple release point, because rsi.perception_sam3.find_blocks drops any block whose top is > 0.060 m above the table (so the stacked cube never appears in nb3; towc_tc26_smoke_b3: stack_measure_few n=0 on all 4 seeds). tc26 was: tc25 + yellow destination from a geometric re-measure of the placed purple cube on a fresh head-camera frame: purple SAM3 depth points more than 20 mm above the perceived bar top; top = their 95th-percentile z, xy = mean of points within 8 mm of that top; first yellow dest = (xy, top + HALF + CLEAR), then the tc25 list. Evidence: stack_observed.used was False in every tc21/tc23/tc24a/tc25 record (95th percentile of mixed bar+cube points = bar top 0.080 m), and towc_tc25_diaggrip_b3 seeds 12/16 - the purple cube settled 10-15 mm from its release point, yellow aimed at the release point fell off.
# ctl_build_tower_tc25 = ctl_build_tower_tc24a on g1_common_tc25 + the planner obstacle: perceived bar points for the purple-cube transfer; for the yellow transfer, the perceived purple blocks of the fresh stack observation (bar + placed cube), or the first-frame bar if that observation fails.
# ctl_build_tower_tc24a = ctl_build_tower_tc23 on g1_common_tc24a (slow still release only; CLEAR stays 4 mm).
# ctl_build_tower_tc23 = ctl_build_tower_tc21 with destinations ordered toward the robot first: bar/stack dests -6, -12 mm (x), then the observed centre; the yellow target is the purple release point shifted -6 mm, then unshifted. Evidence: towc_tc21_dev0_19_b3 - every +x slide-off failure (seeds 1, 16, 19 yellow/purple) was released at the centre dest (true -4 mm), most successes at the -6 mm dest (true -10 mm); towc_tc22_diaggrip_b3 seed 16: cube slid +x off the bar after release with no hand contact.
# ctl_build_tower_tc21 = ctl_build_tower_tc20 on g1_common_tc21 (no joint-space home sweep unless the raised park point was reached branch-continuously).
# ctl_build_tower_tc20 = ctl_build_tower_tc19 on g1_common_tc20 (post-release joint-space reverse path).
# ctl_build_tower_tc19 = ctl_build_tower_tc18 on g1_common_tc19 (raised-park recovery after release).
# ctl_build_tower_tc18 = ctl_build_tower_tc17 + (a) the yellow destination is the purple cube's actual release point (after redest) raised by one block (32 mm), followed by the r11 stack/bar fallbacks; (b) no support re-perception for the yellow cube (r12 'stack' redest picked the bar top on towc_tc17_smoke_b3 seed 4 and lowered the target 32 mm). Evidence: towc_tc17_smoke_b3 seed 9 - cube re-aimed 25 mm by redest but yellow went to the original plan point and fell beside the stack.
# ctl_build_tower_tc17 = ctl_build_tower_tc15 with the controller's own footprint margin EDGE 10 -> 6 mm, so the -6 mm (toward the robot) bar destination is admitted (dest order unchanged: centre, -6, -12 mm). Evidence: towc_tcdp_DIAG_notscored seeds 0/4 - at the bar centre 0-5 branch-continuous chains, at -6 mm 8-16. Scorer and criterion unchanged; the cube centre stays >= 6 mm inside the observed bar top footprint.
# ctl_build_tower_tc15 (tasks_tower_clean) = baseline ctl_build_tower_r12 (r11 + support re-perception after a held grasp) unchanged except it imports g1_common_tc15. Evidence for adding r12's redest: towc_r11_diagmove_b3 seed 15 and towc_r11_diag_b3 seed 3 - closing on the purple cube knocked the bar 2-5 cm and the cube was released over the old bar position.
# ctl_build_tower_r12 (r11 + after a held grasp the support (bar, or the purple stack top for the yellow cube) is re-perceived with SAM3 and the destination re-chosen from the new observation, keeping the grasp orientation (reach preflight incl. carry segment); build_tower_r11_diag_seed3 evidence: closing on the purple cube knocked the bar 5.7 cm (tilt 27 deg), the cube was then released at the old bar position). r11 was: (r10 controller on g1_common_r11: joint-limit margin 0 with trusted float-level arm encoder clamp <= 0.002 rad (logged), carry-segment preflight, via-point approach, raised park point; diag_margin_plan_r11.log: plans on 18/20 layouts vs 14/20 with margin 0.02). r10 was: (r9 on g1_common_r10 (squeeze 0.05 grasp) + a missed grasp opens and rises 6 cm instead of a joint-space home, saving ~2 s per retry; r9 DEV seed 3 evidence: 4 missed grasps then sim time cap). r9 was: (= r8 controller on g1_common_r9: plans keep 0.02 rad from arm joint limits; diag_place_enc_r9_seed0.log: the r8 seed-0 IK sat at the wrist-roll limit and physics overshot by 3e-5 rad; diag_margin_plan_r9*.log: margin 0.02 leaves joint plans on 15/20 DEV layouts (none for the purple cube on 0,1,8,11,18)). r8 was: (r7 + table height from the robust fitted plane (perception_geo.fit_plane) instead of perception.table_height, and grasp height = observed block top - 16 mm; evidence: r7 records show table_height 0.033 m on every seed while the world table top is 0.0444 m, so every grasp aimed ~10 mm low; diag_tilt_grasp_r7_seed1_high.log: +12 mm error already fails). r7 was: (r6b + up to 5 grasp attempts while sim time allows (reserve per block), park move 1.5 s; r6b DEV 0-3 evidence: seed 3 and seed 0 failed after 3 observed-missed grasps, diag_tilt_grasp_r4_seed3_right.log shows grasp outcome varies per orientation). r6b was: (r6 + a pick() that raises no_grasp counts as a failed attempt instead of ending the episode; r6 seed 0 evidence). r6 was: (r5 + observed grasp check after lift: if SAM3 still sees the block on the table near the pick point, open, re-perceive, retry with the next feasible orientation, max 3 attempts; r5 seed 0 evidence: 55 deg pinch nudged the cube without lifting): G1 build_tower controller, r2 + one joint kinematic plan per block (arm, least-tilted pinch orientation, destination) chosen before any motion; the orientation is kept from grasp to release.
# Runtime side: uses only io (rendered RGB-D via SAM3 masks, encoders, demo priors). Never reads object truth, scorer or contacts.
# r2 evidence (seed 0 diag): top-down pinches reach only x <= ~0.03 m from the park pose; the bar sits at x 0.03-0.11 m, so r1 re-oriented the held cube to fwd_horizontal at the bar and dropped it, and r2's horizontal grasp pushed the cube instead of lifting it.
# Destination: observed bar top centre, then points shifted toward the robot (-x) by 6 and 12 mm, kept >= 10 mm inside the observed bar top footprint (scorer needs the child centre inside the bar xy box; not read here, only the observed footprint is used).
# Contact priors: cube 32 mm grasp profile (all blocks 32 mm wide); demo close rotation first, then code-constant tilted orientations; fixed park pose.
import numpy as np
from rsi import perception as P
from rsi import perception_sam3 as S
from rsi import perception_geo as G
import g1_common_tc27 as C

TIE_RATIO = 1.15
HALF = 0.016
CLEAR = 0.004
EDGE = 0.006


def table_z(obs):
    return float(G.fit_plane(obs)[0][2])


def top_pts(c):
    p = c['points']
    return p[p[:, 2] > c['top'] - .008]


def top_xy(c):
    return top_pts(c)[:, :2].mean(0)


def elong(c):
    xy = c['points'][:, :2]
    w = np.linalg.eigvalsh(np.cov((xy - xy.mean(0)).T))
    return float(np.sqrt(max(w[-1], 1e-12) / max(w[0], 1e-12)))


def annotate(bl, zt):
    for b in bl:
        b['top'] = float(np.percentile(b['points'][:, 2], 95))
        b['table'] = zt
    return bl


def dest_list(blk, z):
    # observed top centre first, then -x shifted points that stay EDGE inside the observed top footprint along x at that y band
    t = top_pts(blk)
    cxy = t[:, :2].mean(0)
    out = []
    band = t[np.abs(t[:, 1] - cxy[1]) < .01]
    xmin = float(np.percentile(band[:, 0], 3)) if len(band) > 10 else float(np.percentile(t[:, 0], 3))
    for dx in (.006, .012):
        x = cxy[0] - dx
        if x >= xmin + EDGE:
            out.append(np.r_[x, cxy[1], z])
    out.append(np.r_[cxy, z])
    return out, xmin


def sam_roles(io, obs):
    proc = S.get_proc()
    zt = table_z(obs)
    ylw, _ = S.find_blocks(proc, obs, prompt=('yellow cube',), bowl=None)
    pur, _ = S.find_blocks(proc, obs, prompt=('purple cube',), bowl=None)
    if not ylw:
        raise RuntimeError('sam3: no yellow cube instance')
    yel = ylw[0]
    pur = [b for b in pur if np.linalg.norm(b['center'][:2] - yel['center'][:2]) > .04]
    pur = sorted(pur, key=lambda b: -b['score'])[:2]
    if len(pur) < 2:
        raise RuntimeError('sam3: %d purple instances after removing yellow' % len(pur))
    annotate(pur + [yel], zt)
    e = [elong(b) for b in pur]
    ratio = max(e) / max(min(e), 1e-9)
    if ratio >= TIE_RATIO:
        bar = pur[int(np.argmax(e))]
        basis = 'elongation'
    else:
        bar = max(pur, key=lambda b: b['n'])
        basis = 'mask_points_tiebreak'
    cube = [b for b in pur if b is not bar][0]
    io.log('roles_sam', basis=basis, elong=[round(x, 2) for x in e], ratio=round(ratio, 2), n=dict(yellow=yel['n'], purple=[b['n'] for b in pur]), bar=bar['center'][:2], cube=cube['center'][:2], yellow=yel['center'][:2])
    return bar, cube, yel, zt


def left_behind(io, prompt, c, avoid):
    # observed check: a block of this colour still resting on the table within 5 cm of the pick point (avoid = xy of other same-colour blocks, e.g. the bar)
    o = io.observe()
    zt = table_z(o)
    bl, _ = S.find_blocks(S.get_proc(), o, prompt=(prompt,), bowl=None)
    annotate(bl, zt)
    for b in bl:
        xy = top_xy(b)
        if any(np.linalg.norm(xy - np.asarray(a)) < .03 for a in avoid):
            continue
        if np.linalg.norm(xy - c[:2]) < .05 and b['top'] < zt + .045:
            return np.r_[xy, b['top'] - HALF]
    return None


CAP = 60.0


def support_dests(io, kind, ref_xy, base_top):
    # re-perceive the support on a fresh frame: kind 'bar' = elongated purple block near ref_xy resting on the table; 'stack' = highest purple top near ref_xy
    o = io.observe()
    zt = table_z(o)
    bl, _ = S.find_blocks(S.get_proc(), o, prompt=('purple cube',), bowl=None)
    annotate(bl, zt)
    bl = [b for b in bl if np.linalg.norm(top_xy(b) - ref_xy) < .10 and b['top'] < zt + .11]
    if kind == 'bar':
        bl = [b for b in bl if elong(b) >= 1.5 and b['top'] < zt + .05]
        if not bl:
            return None, None
        b = max(bl, key=elong)
        ds, _ = dest_list(b, b['top'] + HALF + CLEAR)
        return ds, b
    if not bl:
        return None, None
    b = max(bl, key=lambda b: b['top'])
    ds, _ = dest_list(b, b['top'] + HALF + CLEAR)
    return ds, b


def redest(io, side, gR, d, support, tag):
    kind, ref_xy, base_top = support
    try:
        ds, b = support_dests(io, kind, ref_xy, base_top)
    except Exception as e:
        if io.error is not None:
            raise
        io.log('redest_failed', tag=tag, err=str(e)[:80])
        return d
    if not ds:
        io.log('redest_none', tag=tag)
        return d
    p0 = io.tcp(side)[:3, 3]
    for nd in ds:
        a1 = nd + [0, 0, .06]
        seg = [p0 + (a1 - p0) * k / 6.0 for k in range(1, 6)]
        ok, _ = C.reach(io, side, seg + [a1, nd, nd + [0, 0, .10]], gR)
        if ok:
            io.log('redest', tag=tag, old=d, new=nd, moved=float(np.linalg.norm(nd[:2] - d[:2])), top=b['top'])
            return nd
    io.log('redest_unreachable', tag=tag, n=len(ds))
    return d


def transfer(io, c, dests, cl, tag, prompt, avoid, reserve, support):
    tried = set()
    for attempt in range(5):
        if attempt and io.now() + 7.5 + reserve > CAP:
            io.log('attempt_budget_exhausted', tag=tag, attempt=attempt, t=io.now())
            break
        cands = [x for x in cl if x[0] not in tried]
        plan = C.pick_place_plan(io, c, dests, cands)
        if plan is None:
            io.log('no_joint_plan', tag=tag, attempt=attempt, n_dests=len(dests), n_cands=len(cands))
            raise RuntimeError('no_joint_plan:' + tag)
        side, gname, gR, d, di, n = plan
        tried.add(gname)
        io.log('joint_plan', tag=tag, attempt=attempt, side=side, cand=gname, dest=d, dest_index=di, n_checked=n)
        try:
            C.pick(io, c, [(gname, gR)], tag, prefer=side)
        except RuntimeError as e:
            if io.error is not None or not str(e).startswith('no_grasp'):
                raise
            io.log('pick_failed', tag=tag, attempt=attempt, err=str(e)[:80])
            io.grip(side, 1., .5)
            io.home(side, 2.0)
            io.home('left' if side == 'right' else 'right', 2.0)
            continue
        lb = left_behind(io, prompt, c, avoid)
        if lb is None:
            io.log('grasp_check', tag=tag, attempt=attempt, held=True)
            if support is not None:
                d = redest(io, side, gR, d, support, tag)
            C.place(io, side, [(gname, gR)], d, tag + '_place')
            return side, d
        io.log('grasp_check', tag=tag, attempt=attempt, held=False, seen=lb)
        io.grip(side, 1., .4)
        try:
            io.move(side, io.tcp(side)[:3, 3] + [0, 0, .06], io.tcp(side)[:3, :3], .6)
        except Exception as e:
            if io.error is not None:
                raise
            io.home(side, 2.0)
        c = lb
    raise RuntimeError('grasp_failed:' + tag)


def controller(io):
    C.park(io, 1.5)
    ev = io.demo['events']
    lc = next(e for e in ev if e['arm'] == 'left' and e['kind'] == 'close')
    rc = next(e for e in ev if e['arm'] == 'right' and e['kind'] == 'close')
    tc = C.tcands()
    obs = io.observe()
    bar, cube, yel, zt = sam_roles(io, obs)
    bar_top = bar['top']
    io.log('perceived', bar=top_xy(bar), cube=top_xy(cube), yellow=top_xy(yel), bar_top=bar_top, table=zt)
    # 1) purple cube onto the bar
    c1 = np.r_[top_xy(cube), cube['top'] - HALF]
    d1s, xmin = dest_list(bar, bar_top + HALF + CLEAR)
    io.log('bar_dests', dests=d1s, xmin=xmin)
    C.set_obstacle(bar['points'], zt)
    _s1, dc = transfer(io, c1, d1s, [('demo_close_left', lc['rotation'])] + tc, 'cube', 'purple cube', [top_xy(bar)], 20.0, ('bar', top_xy(bar), bar_top))
    io.log('cube_release_point', d=dc)
    # 2) re-observe the stack; aim at the highest purple top surface if the cube is there, else the bar
    d2s = [np.r_[d[:2], bar_top + .032 + HALF + CLEAR] for d in d1s]
    try:
        o3 = io.observe()
        zt3 = table_z(o3)
        nb3, _ = S.find_blocks(S.get_proc(), o3, prompt=('purple cube',), bowl=None)
        annotate(nb3, zt3)
        nb3 = [b for b in nb3 if np.linalg.norm(b['center'][:2] - yel['center'][:2]) > .04]
        if nb3:
            C.set_obstacle(np.vstack([b['points'] for b in nb3]), zt3)
            hb = max(nb3, key=lambda b: b['top'])
            used = bool(hb['top'] > bar_top + .02)
            if used:
                d2s, _ = dest_list(hb, hb['top'] + HALF + CLEAR)
            io.log('stack_observed', top=hb['top'], bar_top=bar_top, xy=top_xy(hb), used=used)
    except Exception as e:
        if io.error is not None:
            raise
        io.log('stack_observe_failed', error=str(e)[:100])
    c2 = np.r_[top_xy(yel), yel['top'] - HALF]
    d2s = [np.r_[dc[0] - .006, dc[1], dc[2] + .032], np.r_[dc[:2], dc[2] + .032]] + list(d2s)
    try:
        o4 = io.observe()
        pl4 = G.fit_plane(o4)
        p4, ys4, xs4 = G.safe_points(o4)
        h4 = G.height(p4, pl4)
        st4 = S.get_proc().set_image(S.Image.fromarray(o4['rgb']))
        mk, sc = S._prompt(S.get_proc(), st4, 'purple cube')
        sel = np.zeros(len(p4), bool)
        for m_, s_ in zip(mk, sc):
            if s_ >= .3:
                sel |= m_[ys4, xs4]
        sel &= (h4 > .045) & (h4 < .075) & (np.linalg.norm(p4[:, :2] - dc[:2], axis=1) < .03)
        hp, hh = p4[sel], h4[sel]
        if len(hp) >= 30:
            hmax = float(np.percentile(hh, 95))
            sxy = hp[hh > hmax - .008][:, :2].mean(0)
            stop = float(np.percentile(hp[:, 2], 95))
            io.log('stack_measured', xy=sxy, top=stop, height=hmax, n=int(len(hp)), from_release=float(np.linalg.norm(sxy - dc[:2])))
            if abs(stop - (dc[2] + HALF - CLEAR)) < .015:
                d2s = [np.r_[sxy, stop + HALF + CLEAR]] + d2s
        else:
            io.log('stack_measure_few', n=int(len(hp)))
    except Exception as e:
        if io.error is not None:
            raise
        io.log('stack_measure_failed', error=str(e)[:80])
    transfer(io, c2, d2s, [('demo_close_right', rc['rotation'])] + tc, 'yellow', 'yellow cube', [], 8.0, None)
    io.wait(2.5)
    io.observe()
