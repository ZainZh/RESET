"""build_tower t1: SAM-first role assignment (new candidate; b5 is untouched).
b5 motion, IK retries and re-observe are kept. Differences from b5:
- yellow cube = top instance of the SAM3 prompt "yellow cube" (learned mask, no HSV threshold);
- the other two blocks = SAM3 "purple cube" instances; bar vs cube from geometry inside the learned masks
  (XY elongation of the lifted mask points; tie -> larger mask point count). No hue value is used for any decision;
  hue is only logged as a diagnostic (hue_diag) and never read back.
- no silent fallback: missing roles raise and the episode counts as a failure.
Runtime inputs: rendered RGB-D, calibration, encoders, robot model, demo contact priors (grasp rotations at close/open events).
"""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from rsi import perception as P
from rsi import perception_sam3 as S
from twin.practice.motion import KinematicsError
UP = np.array([0, 0, 1.])
TIE_RATIO = 1.15


def top_xy(c):
    p = c["points"]
    t = p[p[:, 2] > c["top"] - .008]
    return t[:, :2].mean(0)


def elong(c):
    xy = c["points"][:, :2]
    w = np.linalg.eigvalsh(np.cov((xy - xy.mean(0)).T))
    return float(np.sqrt(max(w[-1], 1e-12) / max(w[0], 1e-12)))


def _hue_diag(b):
    return float(cv2.cvtColor(np.uint8(b["mean_rgb"]).reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0, 0])


def _annotate(bl, zt):
    for b in bl:
        b["top"] = float(np.percentile(b["points"][:, 2], 95)); b["table"] = zt
    return bl


def sam_roles(io, obs):
    proc = S.get_proc()
    zt = P.table_height(obs)
    ylw, _ = S.find_blocks(proc, obs, prompt=("yellow cube",), bowl=None)
    pur, _ = S.find_blocks(proc, obs, prompt=("purple cube",), bowl=None)
    if not ylw:
        raise RuntimeError("sam3: no yellow cube instance")
    yel = ylw[0]
    pur = [b for b in pur if np.linalg.norm(b["center"][:2] - yel["center"][:2]) > .04]
    pur = sorted(pur, key=lambda b: -b["score"])[:2]
    if len(pur) < 2:
        raise RuntimeError("sam3: %d purple instances after removing yellow" % len(pur))
    _annotate(pur + [yel], zt)
    e = [elong(b) for b in pur]
    ratio = max(e) / max(min(e), 1e-9)
    if ratio >= TIE_RATIO:
        bar = pur[int(np.argmax(e))]; basis = "elongation"
    else:
        bar = max(pur, key=lambda b: b["n"]); basis = "mask_points_tiebreak"
    cube = [b for b in pur if b is not bar][0]
    io.log("roles_sam", basis=basis, elong=[round(x, 2) for x in e], ratio=round(ratio, 2),
           scores=dict(yellow=round(yel["score"], 2), purple=[round(b["score"], 2) for b in pur]),
           n=dict(yellow=yel["n"], purple=[b["n"] for b in pur]),
           hue_diag=dict(yellow=round(_hue_diag(yel)), purple=[round(_hue_diag(b)) for b in pur]),
           bar=bar["center"][:2], cube=cube["center"][:2], yellow=yel["center"][:2])
    if basis != "elongation":
        io.log("role_ambiguous", alternative_bar=cube["center"][:2], chosen_bar=bar["center"][:2])
    return bar, cube, yel, zt


def safe_move(io, arm, p, R, dur, tag):
    p = np.asarray(p, float)
    tries = [(0., 0.)] + [(dz, yaw) for yaw in (0., 12., -12., 25., -25.) for dz in (0., .02, -.01)][1:]
    last = None
    for k, (dz, yaw) in enumerate(tries):
        try:
            Rk = Rotation.from_euler("z", np.radians(yaw)).as_matrix() @ np.asarray(R, float)
            err = io.move(arm, p + dz * UP, Rk, dur)
            if k: io.log("ik_retry", step=tag, attempt=k, dz=dz, yaw=yaw)
            return err
        except KinematicsError as e:
            last = e
    io.log("ik_failed", step=tag, error=str(last))
    raise last


def pick_place(io, arm, grasp, xy, table, tag):
    t = np.r_[xy, table + .021]
    io.grip(arm, 1., .3)
    safe_move(io, arm, t + .10 * UP, grasp["rotation"], 1.5, tag + ":above")
    safe_move(io, arm, t, grasp["rotation"], 1.0, tag + ":down")
    io.grip(arm, 0., .6)
    safe_move(io, arm, t + .15 * UP, grasp["rotation"], 1.0, tag + ":lift")


def place(io, arm, rel, d, tag):
    safe_move(io, arm, d + .06 * UP, rel["rotation"], 1.5, tag + ":above")
    safe_move(io, arm, d, rel["rotation"], .8, tag + ":down")
    io.grip(arm, 1., .6)
    try:
        safe_move(io, arm, d + .10 * UP, rel["rotation"], .8, tag + ":retreat")
    except KinematicsError:
        pass
    io.home(arm, 2.0)


def run(io):
    ev = io.demo["events"]
    lc = next(e for e in ev if e["arm"] == "left" and e["kind"] == "close")
    lo = next(e for e in ev if e["arm"] == "left" and e["kind"] == "open")
    rc = next(e for e in ev if e["arm"] == "right" and e["kind"] == "close")
    ro = next(e for e in ev if e["arm"] == "right" and e["kind"] == "open")
    obs = io.observe()
    bar, cube, yel, zt = sam_roles(io, obs)
    bxy = top_xy(bar)
    io.log("perceived", bar=bxy, cube=top_xy(cube), yellow=top_xy(yel), bar_top=bar["top"])
    pick_place(io, "left", lc, top_xy(cube), cube["table"], "L")
    try:
        o2 = io.observe()
        zt2 = P.table_height(o2)
        nb, _ = S.find_blocks(S.get_proc(), o2, prompt=("purple cube",), bowl=None)
        _annotate(nb, zt2)
        cand = [b for b in nb if np.linalg.norm(b["center"][:2] - cube["center"][:2]) > .04 and np.linalg.norm(b["center"][:2] - yel["center"][:2]) > .04 and elong(b) >= 1.5]
        if cand:
            bar = max(cand, key=elong); bxy = top_xy(bar)
        io.log("reobserve", n_candidates=len(cand), bar=bxy)
    except Exception as e:
        io.log("reobserve_failed", error=str(e))
    place(io, "left", lo, np.r_[bxy, bar["top"] + .021 + .012], "L")
    pick_place(io, "right", rc, top_xy(yel), yel["table"], "R")
    d2 = np.r_[bxy, bar["top"] + .032 + .021 + .012]
    safe_move(io, "right", d2 + .06 * UP, ro["rotation"], 1.5, "R2:above")
    safe_move(io, "right", d2, ro["rotation"], .8, "R2:down")
    io.grip("right", 1., .6)
    try:
        safe_move(io, "right", d2 + .10 * UP, ro["rotation"], .8, "R2:retreat")
    except KinematicsError:
        pass
    io.wait(2.0)
    io.observe()
