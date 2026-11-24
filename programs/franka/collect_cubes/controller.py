# collect_cubes c30_drop_memory = c29_observed_drop + bowl-relative memory of executed release targets and previously seen
# in-bowl blocks, merged with current SAM-visible occupancy, so occluded earlier releases are not treated as free slots.
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from rsi import perception as P
from rsi import perception_geo as G
from rsi import perception_sam3 as S

YAWS = (0, 25, -25, 50, -50)
UP = np.array([0, 0, 1.])
HALF_SPAN = .0625  # finger-endpoint half span used by the c24 clearance rule
RIM_MARGIN = .02
DEADBAND = .02  # DEV control hypothesis from c27 3-8 mm fit/settling centre changes on DEV55/60, not a camera-noise calibration
OCC_H = (.015, .12)  # plausible top height above the table for a visible in-bowl cube (single or stacked); bowl floor ~5 mm, rim ~90 mm
MERGE = .025  # memory/visible points closer than this are treated as the same block (cube side 32 mm); declared rule, not tuned


def fit_plane_reduced(obs, iters=4, band=.008):
    """Inline copy of rsi.perception_geo.fit_plane; the only change is full_matrices=False in the SVD (only Vt is used).
    The shared full SVD built an N x N U (N ~ 5000, ~200 MB) per iteration: ~1.4-1.9 s per call vs ~0.025 s here,
    identical plane on DEV 17,28,43,52,55,60,65."""
    p, _, _ = G.safe_points(obs)
    z0 = P.table_height(obs)
    w = p[(p[:, 0] > G.FIT_X[0]) & (p[:, 0] < G.FIT_X[1]) & (p[:, 1] > G.FIT_Y[0]) & (p[:, 1] < G.FIT_Y[1])]
    c = np.array([0., -.5, z0]); n = np.array([0., 0., 1.])
    sel = np.abs(w[:, 2] - z0) < .015
    for _ in range(iters):
        near = w[sel]
        if len(near) < 200:
            break
        c = near.mean(0)
        _, _, vt = np.linalg.svd(near[::5] - c, full_matrices=False)
        n = vt[-1] * np.sign(vt[-1][2])
        sel = np.abs((w - c) @ n) < band
    return c, n


def yawed(R, deg):
    return Rotation.from_euler('z', deg, degrees=True).as_matrix() @ np.asarray(R)


def fatal(e):
    return type(e).__name__ in ('NumericalFailure', 'Limit')


def finger_angle(R):
    return float(np.degrees(np.arctan2(R[1, 1], R[0, 1])))


def clearance(xy, th_deg, bxy, rim_r):
    # c24 rule: finger endpoints at xy +/- HALF_SPAN along the finger axis stay RIM_MARGIN outside the rim; a point check, not a swept-path proof
    th = np.radians(th_deg)
    a = np.array([np.cos(th), np.sin(th)])
    return float(min(np.linalg.norm(np.asarray(xy) + s * HALF_SPAN * a - np.asarray(bxy)) for s in (1, -1)) - (rim_r + RIM_MARGIN))


def approach(io, arm, t, Rbase, above, down_s, label, ok=None):
    last = None
    for deg in YAWS:
        R = yawed(Rbase, deg)
        if ok is not None and not ok(deg, R):
            io.log('yaw_skip_clearance', what=label, arm=arm, yaw=deg)
            continue
        try:
            io.move(arm, t + above * UP, R, 1.1)
            io.move(arm, t, R, down_s)
            return R
        except Exception as e:
            if fatal(e):
                raise
            last = e
            io.log('yaw_retry', what=label, arm=arm, yaw=deg, err=str(e)[:60])
    raise last if last is not None else RuntimeError('no clearance-safe yaw for ' + label)


def place_block(io, arm, t, Rg, Ro, bxy, btop, k, home_p, color, pick_ok, ring):
    """Pick at t, drop near bowl centre (ring offset), lift; return the commanded release TCP xy offset from bxy (not a landed centre)."""
    io.grip(arm, 1., .3)
    approach(io, arm, t, Rg, .10, 1.0, 'pick_' + color, ok=pick_ok)
    io.grip(arm, 0., .6)
    io.move(arm, t + .14 * UP, io.tcp(arm)[:3, :3], .8)
    io.log("drop_ring", k=k, radius=float(np.linalg.norm(ring)), color=color)
    v = home_p[arm][:2] - np.asarray(bxy); v = v / max(np.linalg.norm(v), 1e-6)
    Rd = None
    for shift in (0., .015):
        for hh in ((.03, .05, .08) if color == 'bar' else (.04, .07, .10)):
            dd = np.r_[np.asarray(bxy) + ring + shift * v, btop + hh]
            try:
                Rd = approach(io, arm, dd, Ro, .05, .8, 'drop_' + color); d = dd
                break
            except Exception as e2:
                if fatal(e2): raise
                io.log('drop_variant_failed', color=color, shift=shift, h=hh)
        if Rd is not None: break
    if Rd is None: raise RuntimeError('drop unreachable')
    io.grip(arm, 1., .6)
    for h in (.10, .06, .03):
        try:
            io.move(arm, d + h * UP, Rd, .8)
            break
        except Exception as e3:
            if fatal(e3): raise
            io.log('lift_retry', color=color, h=h, err=str(e3)[:60])
    return np.asarray(d[:2], float) - np.asarray(bxy, float)


def bowl_occupancy(obs, plane, bxy, rim_r):
    """Visible in-bowl blocks from the main camera: same SAM prompts as S.find_blocks, depth-lifted, kept if the top-point centre lies inside
    the bowl disc (rim_r - .03, the find_blocks exclusion disc) and its top is OCC_H above the table plane. Separate from the outside-bowl
    pick selector because find_blocks excludes the disc and rejects tops > 60 mm. Visibility only: cubes hidden by the wall, the rim or
    other cubes are missed, so this is not complete occupancy."""
    proc = S.get_proc()
    p, ys, xs = G.safe_points(obs)
    h = G.height(p, plane)
    state = proc.set_image(Image.fromarray(obs['rgb']))
    occ = []
    for pr in ('small cube', 'block'):
        mk, sc = S._prompt(proc, state, pr)
        for m, s in zip(mk, sc):
            if s < .3:
                continue
            sel = m[ys, xs] & (h > OCC_H[0]) & (h < OCC_H[1])
            if not 20 <= sel.sum() <= 3000:
                continue
            q, hq = p[sel], h[sel]
            top = q[hq > np.percentile(hq, 95) - .010]
            c = top[:, :2].mean(0)
            if float((q[:, :2].max(0) - q[:, :2].min(0)).max()) > .06 or np.linalg.norm(c - bxy) > rim_r - .03:
                continue
            if any(np.linalg.norm(c - o['xy']) < .015 for o in occ):
                continue
            occ.append({'xy': c, 'top_h': float(np.percentile(hq, 95)), 'score': float(s), 'n': int(sel.sum())})
    return occ


def choose_drop(k, bxy, occ):
    """Drop TCP offset from the bowl centre. The TCP target is not the landed object centre; the released cube can slide or bounce.
    No visible occupancy: c24 ring slot. Otherwise the slot inside the c24 ring envelope (r <= 35 mm) whose minimum distance to the
    visible in-bowl centres is largest; ties go to the c24 ring slot."""
    ang = k * 2.4
    rr = (0., .03, .03, .035, .035)[min(k, 4)]
    ring = rr * np.array([np.cos(ang), np.sin(ang)]) if k else np.zeros(2)
    if not occ:
        return ring, {'reason': 'no_visible_occupancy', 'objective': None}
    cen = [o['xy'] - np.asarray(bxy) for o in occ]
    score = lambda s: min(np.linalg.norm(s - c) for c in cen)
    slots = [np.zeros(2), ring] + [r * np.array([np.cos(a), np.sin(a)]) for r in (.02, .03, .035) for a in np.radians(np.arange(0, 360, 30))]
    best = max(slots, key=lambda s: (round(score(s), 3), -np.linalg.norm(s - ring)))
    return best, {'reason': 'observed', 'objective': float(score(best)), 'ring_objective': float(score(ring))}


def remember(mem, pt, src, rnd):
    """Bowl-relative memory: a new point replaces older entries within MERGE (latest wins). Never forgotten otherwise (conservative)."""
    pt = np.asarray(pt, float)
    mem[:] = [m for m in mem if np.linalg.norm(m["rel"] - pt) >= MERGE] + [{"rel": pt, "src": src, "round": rnd}]


def planning_occupancy(occ, mem, bxy):
    """Current visible blocks plus remembered points not within MERGE of any visible one. Returned as world xy for choose_drop."""
    vis = [o["xy"] - np.asarray(bxy) for o in occ]
    extra = [m for m in mem if all(np.linalg.norm(m["rel"] - v) >= MERGE for v in vis)]
    pts = list(occ) + [{"xy": np.asarray(bxy) + m["rel"], "memory": True} for m in extra]
    return pts, extra


def go_home(io, arm, home_p, home_R):
    # lift straight up first: a joint-space sweep from inside the bowl can drag fingers through the blocks
    T = io.tcp(arm)
    for h in (.10, .06, .03):
        try:
            io.move(arm, T[:3, 3] + h * UP, T[:3, :3], .6)
            break
        except Exception as e:
            if fatal(e): raise
    try:
        io.home(arm, 1.8)
    except Exception as e:
        if fatal(e): raise
        io.log('home_failed', arm=arm, err=str(e)[:60])


def clear_view(io, home_p, home_R):
    # idle arms at home hide cubes from the camera (probe_c23_dev52_lookpose): put both TCPs at a fixed look pose
    # = demo home + h up with home rotation, independent of the current TCP so repeated looks do not climb (c24 drift)
    for arm in ('left', 'right'):
        for h in (.15, .10, .06):
            tgt = home_p[arm] + h * UP
            try:
                io.log('clear_view_target', arm=arm, h=h, p=tgt)
                io.move(arm, tgt, home_R[arm], 1.0)
                break
            except Exception as e:
                if fatal(e): raise
                io.log('clear_view_retry', arm=arm, h=h, err=str(e)[:60])


def sam_bowl(obs, plane):
    # SAM3 "bowl" mask + cylinder fit of its top points (rsi.perception_sam3.locate_bowl), main camera RGB-D only
    try:
        proc = S.get_proc()
        p, ys, xs = G.safe_points(obs)
        b = S.locate_bowl(proc, proc.set_image(Image.fromarray(obs['rgb'])), p, ys, xs, plane)
    except Exception as e:
        return None, str(e)[:60]
    if b is None or not (np.all(np.isfinite(b[0])) and np.isfinite(b[1])):
        return None, 'none'
    return b, None


def colour_bowl(obs, tcps):
    # legacy beige colour locate (as c18/c24 initial bowl)
    nb = P.locate(obs, 'beige', tcps, max_height=.12)
    if nb is None:
        return None
    xy, r = P.fit_upright_cylinder(nb['points'][nb['points'][:, 2] > nb['top'] - .015])
    return (np.asarray(xy, float), float(r)) if np.all(np.isfinite(xy)) and np.isfinite(r) else None


def run(io):
    ev = io.demo['events']
    rot = {}
    for arm in ('left', 'right'):
        c = next((e for e in ev if e['arm'] == arm and e['kind'] == 'close'), None)
        o = next((e for e in ev if e['arm'] == arm and e['kind'] == 'open'), None)
        hr = io.demo['home'][arm][:3, :3]
        rot[arm] = (c['rotation'] if c else hr, o['rotation'] if o else hr)
    home_p = {a: io.demo['home'][a][:3, 3] for a in ('left', 'right')}
    home_R = {a: io.demo['home'][a][:3, :3] for a in ('left', 'right')}
    obs = io.observe()
    tcps = [io.tcp(a)[:3, 3] for a in ('left', 'right')]
    bowl = P.locate(obs, 'beige', tcps, max_height=.12)
    if bowl is None:
        raise RuntimeError('bowl not found')
    bxy, rim_r = P.fit_upright_cylinder(bowl['points'][bowl['points'][:, 2] > bowl['top'] - .015])
    bxy = np.asarray(bxy, float); bxy0 = bxy.copy()
    btop = bowl['top']
    io.log('bowl', xy=bxy, top=btop, r=rim_r)
    # later rounds track the bowl's displacement from t0 (SAM3 first, colour fallback) and add it to the t0 colour centre,
    # so a constant SAM-vs-colour centre bias does not move drops; radius and top stay at the t0 colour fit
    sam0, why = sam_bowl(obs, fit_plane_reduced(obs))
    io.log('bowl_sam_t0', ok=sam0 is not None, err=why, xy=None if sam0 is None else sam0[0], r=None if sam0 is None else sam0[1])
    k = 0
    tries = {}
    mem = []  # bowl-relative occupancy hypotheses from own releases and earlier observations (no simulator state)
    for rnd in range(9):
        clear_view(io, home_p, home_R)
        obs = io.observe()
        tcps = [io.tcp(a)[:3, 3] for a in ('left', 'right')]
        plane = fit_plane_reduced(obs)
        if rnd:
            new, src = None, None
            sb, why = sam_bowl(obs, plane) if sam0 is not None else (None, 'no_t0')
            if sb is not None and abs(sb[1] - sam0[1]) < .03 and np.linalg.norm(sb[0] - sam0[0]) < .15:
                new, src = bxy0 + (sb[0] - sam0[0]), 'sam'
            else:
                try:
                    cb = colour_bowl(obs, tcps)
                except Exception as e:
                    cb = None; io.log('bowl_relocate_failed', err=str(e)[:60])
                if cb is not None and np.linalg.norm(cb[0] - bxy0) < .15:
                    new, src = cb[0], 'colour'
            if new is None:
                io.log('bowl_update', round=rnd, source='kept', sam_err=why)
            else:
                step = float(np.linalg.norm(new - bxy))
                io.log('bowl_update', round=rnd, source=src if step > DEADBAND else src + '_held', xy=new, step=step, from_t0=float(np.linalg.norm(new - bxy0)))
                if step > DEADBAND:
                    bxy = new
        blocks, _ = S.find_blocks(S.get_proc(), obs, plane=plane, bowl=(bxy, rim_r - .03))
        blocks = [x for x in blocks if np.all(np.isfinite(x['center']))]
        cands = [x for x in blocks if tries.get((int(round(x['center'][0] / .03)), int(round(x['center'][1] / .03))), 0) < 3]
        if not cands:
            break
        c = min(cands, key=lambda x: min(np.linalg.norm(home_p[a][:2] - x['center'][:2]) for a in home_p))
        pxy = c['points'][:, :2]; ext = float((pxy.max(0) - pxy.min(0)).max()); color = 'bar' if ext > .055 else 'block'; io.log('extent', ext=ext, color=color)
        key = (int(round(c['center'][0] / .03)), int(round(c['center'][1] / .03)))
        tries[key] = tries.get(key, 0) + 1
        xy = c['center'][:2]
        arm = min(('left', 'right'), key=lambda a: np.linalg.norm(home_p[a][:2] - xy))
        t = np.r_[xy, max(P.table_height(obs) + .021, c['center'][2] - .012)]
        Rg, Ro = rot[arm]
        if color == 'block':
            top = c['points'][c['points'][:, 2] > c['points'][:, 2].max() - .012][:, :2]
            if len(top) >= 8:
                cube_yaw = cv2.minAreaRect(top.astype(np.float32))[2]
                fa = finger_angle(Rg)
                delta = (cube_yaw - fa + 45) % 90 - 45
                ycands = []
                for dd in (delta, delta - 90 if delta > 0 else delta + 90):
                    for off in (0, 10, -10, 20, -20):
                        d = dd + off
                        ycands.append((abs(off), abs(d), d, clearance(xy, fa + d, bxy, rim_r)))
                # tangential finger axis (perpendicular to bowl->cube) maximises endpoint distance from the bowl centre
                rad = np.degrees(np.arctan2(xy[1] - bxy[1], xy[0] - bxy[0]))
                dt = (rad + 90 - fa + 90) % 180 - 90
                ot = abs((dt - delta + 45) % 90 - 45)
                tang = (ot, abs(dt), dt, clearance(xy, fa + dt, bxy, rim_r))
                # cube-axis candidates that meet the clearance rule keep priority; tangential only if none of them does
                good = [q for q in ycands if q[3] >= 0] or ([tang] if tang[3] >= 0 else [])
                ycands.append(tang)
                if not good:
                    tries[key] = 3
                    io.log('pick_skipped_no_clearance', xy=xy, round=rnd, best=max(q[3] for q in ycands))
                    continue
                best = min(good)
                delta = float(best[2])
                Rg = yawed(Rg, delta)
                io.log('grasp_yaw', cube_yaw=float(cube_yaw), finger_axis=float(fa), delta=float(delta), off=float(best[0]), clearance=float(best[3]), tangential=bool(best[2] == dt))
        Rg0 = Rg
        io.log('pick_base_clearance', color=color, clearance=clearance(xy, finger_angle(Rg0), bxy, rim_r))

        def pick_ok(deg, R, xy=xy):
            # every executed pick yaw (block or bar, base or retry) keeps clearance >= 0
            return clearance(xy, finger_angle(R), bxy, rim_r) >= 0
        occ = bowl_occupancy(obs, plane, bxy, rim_r) if k else []
        pts, extra = planning_occupancy(occ, mem, bxy)
        ring, why = choose_drop(k, bxy, pts)
        io.log('drop_choice', round=rnd, k=k, visible_in_bowl=len(occ), occupied=[o['xy'] for o in occ], tops=[o['top_h'] for o in occ],
            memory_used=[{'rel': m['rel'], 'src': m['src'], 'round': m['round']} for m in extra], memory_size=len(mem), slot=ring, **why)
        for o in occ:
            remember(mem, o['xy'] - np.asarray(bxy), 'seen', rnd)
        io.log('pick', color=color, arm=arm, xy=xy, round=rnd)
        try:
            rel = place_block(io, arm, t, Rg, Ro, bxy, btop, k, home_p, color, pick_ok, ring)
            remember(mem, rel, 'release', rnd)  # commanded TCP release, approximate; not proof of placement
            k += 1
        except Exception as e:
            if fatal(e): raise
            io.log('pick_failed', color=color, err=str(e)[:80])
            try:
                io.grip(arm, 1., .3)
            except Exception as e2:
                if fatal(e2): raise
        go_home(io, arm, home_p[arm], home_R[arm])
    io.wait(2.0)
    io.observe()
