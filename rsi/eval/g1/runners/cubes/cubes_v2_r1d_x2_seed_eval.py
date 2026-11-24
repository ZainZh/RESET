# cubes_v2_r1d_x2_seed: cubes_v2_r1d_seed.py unchanged except WORLD = worlds/b3_cubes_v2_x2 (option A: layout +0.02 m in x along the table plane) and TAG.
# cubes_v2_r1d_seed (= cubes_v2_r1c_seed + r1d items, same rules for every seed):
#  (1) PULL-IN shared skill (reverse of sweep pushing): a block with no grasp plan at any approach height is pulled toward the robot along the table: closed hand above the block's far side,
#      down to block-centre height BEHIND beyond it, drag until the TCP is at x = PULL_TO + BEHIND, lift, home; then the table is re-perceived (probe_drag_v2_r1: 5/5 unreachable blocks pulled to x ~ -0.03 and lifted).
#      Skipped if the drag path would come within bowl r + PULL_BOWL_CLEAR of the perceived bowl centre; at most PULL_MAX per block location.
#  (2) per-block attempt cap: after MAX_ATT failed attempts (empty grasp, chain failure, no plan) a block is deferred; deferred blocks get one more attempt after all others; per-attempt time budget
#      (direct 12 s, via home 16 s, pull-in 9 s, + 2.5 s for the final wait and scorer dwell) instead of a fixed cap - 15 s.
#  [smoke2 changes; smoke1 version kept as cubes_v2_r1d_seed_smoke1_snapshot.py] (a) a remembered block is restored only if no seen block is within MEM_R = 6 cm (smoke1 seeds 4/12: a block nudged 3.1 cm by a
#      neighbour's grasp came back as a phantom); (b) a remembered (unseen) block gets one attempt, an empty grasp retires its location; (c) a chain failure after a positive hold check retires the location (the block left with
#      the hand; smoke1 seeds 3/9); (d) blocks perceived beyond PULL_FIRST_X are evaluated first, so a needed pull-in happens early (smoke1 seed 9 ran out of time before pulling).
#  [smoke3; smoke2 version kept as cubes_v2_r1d_seed_smoke2_snapshot.py] (e) when only near-rim blocks remain, candidates with finger clearance in [LATE_CLEAR, FINGER_CLEAR) are allowed for them too (smoke2 seeds 1/3: a near-rim block had no candidate with >= 1.5 cm rim clearance).
#  [smoke4; smoke3 version kept as cubes_v2_r1d_seed_smoke3_snapshot.py] (f) a held block whose direct drop move fails (wrist_yaw limit, smoke1-3 seeds 3/9) is carried home and dropped from home; (g) final wait 3.0 s (smoke3 seed 1: last block still settling during the scorer dwell).
#  (3) end-of-task check: an empty detection is confirmed by a second full-table perception after 0.5 s before stopping; blocks within r + RIM_KEEP of the bowl are no longer excluded but deferred and
#      grasped only with a rim-clear candidate (finger clearance >= FINGER_CLEAR) (r1c seed 1: a block 3 cm outside the rim was never attempted).
# cubes_v2_r1c_seed (= cubes_v2_r1b_seed + far-block fallback: if no candidate plans with the normal c+.15 approach/lift, the same selection is repeated with approach and lift height .10 then .08, always dropped via home, no extra lift before home (diag_select_v2_r3.log: stranded far blocks have 0-3 pick plans at +.15, 3-7 at +.10, 0-1 lifts to +.20, and a home drop always plans)).
# cubes_v2_r1b_seed (= cubes_v2_r1_seed frozen for cubes_v2_r1_full_dev0_19 + drop via home: when the chained pick->drop IK is not plannable, the drop is planned from the home (PARK_CUBES) arm state and executed as lift to c+.20, joint-space home holding the block, then Cartesian drop (full run: blocks at the far right corner had pick plans but no chained drop plan, diag_select_v2_r2.log)).
# cubes_v2_r1_seed (branch programs/g1/collect_cubes, see NOTES.md): = cubes_v1_r7_seed on world b3_cubes_v2 (table 0.92, worlds/b3_cubes_v2/README.md) from the cubes-specific clean start pose PARK_CUBES,
# with (1) grasp orientations restricted to the set proven to lift on this world (b3_cubes_v2_tasks/lift_map_r2, lift_candidate_stats_z0.92.log), ordered by lift precision, the cube-yaw-aligned top-down roll tried first;
# (2) bowl re-located every round; (5) [smoke2] encoder-only empty-grasp check after the lift (finger tracking residual < HOLD_RES -> reopen and retry the same block with the next candidate),
# (10) [smoke7] blocks are tried in order of distance to the bowl, the first block with a clear candidate is grasped; perceived yaw kept in memory for remembered blocks (smoke6 seed 12: remembered block got yaw 0, 3 empty grasps);
# time guard: no new grasp attempt after CAP - ATTEMPT_T (smoke6 seed 12 vetoed dwell_exceeds_cap); smoke6 version kept as cubes_v2_r1_seed_smoke6_snapshot.py;
# (9) [smoke6] finger-bowl clearance: each candidate's finger collision segments (robot model segment_fk at the planned grasp q with pre-shape and closed finger targets) must stay >= FINGER_CLEAR outside the perceived bowl rim, else the candidate goes to the back of the list
# (diag_bowl_push_v2_r1: pinky/ring/middle tilted the bowl while grasping blocks next to it); block memory: blocks seen earlier but missing from the current detection are kept as targets at their last perceived pose (smoke5 seed 0: a far block vanished from SAM after round 0); smoke5 version kept as cubes_v2_r1_seed_smoke5_snapshot.py;
# (8) [smoke5] blocks whose perceived centre is within bowl r + RIM_KEEP of the bowl centre are not grasped (smoke4 seed 9: an empty grasp next to the rim tipped the bowl); smoke4 version kept as cubes_v2_r1_seed_smoke4_snapshot.py;
# (7) [smoke4] after release: Cartesian retreat straight up (+.12, else +.08/+.05) before the joint-space home, and home over 2.0 s (smoke3 seed 9: the joint-space home from the drop pose swept the bowl 0.29 m); smoke3 version kept as cubes_v2_r1_seed_smoke3_snapshot.py;
# (6) [smoke3] drop heights .06/.08/.10 above the rim (smoke2 seed 9: bowl tipped after a .10 drop onto a filled bowl), yaw-aligned candidates ordered roll r0, r0+90 before tilts, RETRIES 2 (smoke2 version kept as cubes_v2_r1_seed_smoke2_snapshot.py);
# no yaw-diagonal candidate filter, wider drop orientation set (tilted top-down rolls); smoke1 version kept as cubes_v2_r1_seed_smoke1_snapshot.py; (3) pre-approach at c+.15 (above the bowl rim) before descending to c+.10 and c; (4) drop orientation may differ from the grasp one if the grasp one cannot reach the bowl.
# r7 header follows: cubes_v1_r7_seed: = cubes_v1_r6_seed + chain-margin grasp/drop selection and recover() from the frozen G1 drawer r8 (r6 smoke: lifts near the body hit wrist_yaw limits, drop unreachable) + drop offset .25 r, drop height .10 first, keep the grasp orientation through the drop. r6 header follows: cubes_v1_r6_seed: = cubes_r5_seed (controller unchanged) on world b3_cubes_v1 (worlds/b3_cubes_v1/README.md) from PARK_CLEAN, rendered on NVIDIA EGL device 2.
# cubes_r5_seed: ONE b2 collect_cubes DEV episode in its own process (called by cubes_r5_parent.py). r5 = r4 runtime (render_guard_cubes_r3, EGL device 1) + new G1 grasp: target = perceived cube centre on the calibrated cube_32mm pinch frame (demo offset logged, not applied), pre-shape opening 0.5, close with hand-module squeeze 0.3, arm homes after each release (probe_cubes_grasp_r18). r4 note: rendered on EGL device 1: EGL device 0 has broken readback (diag_egl_device_r16_all.log: 399/400 mismatching frames of a static scene; devices 1 and 2: 0/400) (see collect_cubes r1 NOTES: corrupted head-cam readbacks on shared GPU 0) + private SAM3 on GPU 1. Same frozen G1 runtime, park pose, finger-encoder clamp and orientation candidates as apple_r4_seed.py.
# Controller (runtime side) uses only io: rendered RGB-D, calibration, encoders, demo priors. It never reads object truth, scorer or contacts.
# Perception: SAM3 find_blocks (no cv2) + SAM3 bowl. Each round re-observes, picks the best visible block outside the bowl, grasps it, drops it over the bowl at a per-round offset.
# Contact prior (listed per task): demo close/open rotations (tried first); code-constant alternate orientations. The demo close-event offset is logged but not applied (G1 uses its calibrated pinch frame).
# Trusted side: builds the world, applies the DEV layout once, clamps float-level finger encoder undershoot at the read and logs it, scores with the independent G1 scorer. Perception error vs layout truth is post-hoc only.
# Disclosure: G1 pelvis, legs and waist are fixed (no freejoint); robot-body gravcomp is in the world. This is not walking or balance. DEV seeds only. No VAL, TEST or hardware.
RESET_ROOT = __import__('os').environ.get('RESET_ROOT') or next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'eval.py').is_file()).as_posix()
import sys, os, json, math, time, types, argparse, traceback
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--outdir', required=True)
ap.add_argument('--video', action='store_true')
ap.add_argument('--layout-file', required=True)
ap.add_argument('--layout-key', required=True, choices=['dev', 'test'])
args = ap.parse_args()
ROOT = RESET_ROOT
R0 = ROOT + '/programs/g1'
WORLD = RESET_ROOT + '/worlds/g1/b3_cubes_v2_x2'
SCENE = WORLD + '/collect_cubes/scene.xml'
TASK = 'collect_cubes'
TAG = 'cubes_v2_r1d_x2'
CAP = 90.0
PARK = (-0.9, 0.5, 0.0, 0.3, 0.0, -0.5, 0.0)  # PARK_CUBES: cubes-specific clean start pose, code constant (b3_cubes_v2_build/cubes_v2_check_r2_final.log: zero robot-table/robot-object penetration on DEV0-19 + demo)
JN = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']
HAND_KEYS = ('thumb', 'index', 'middle', 'ring', 'pinky')
CUBE_H = 0.032
MAX_ROUNDS = 24
DISC = 'b3_cubes_v2 world (b3 with the demo table top 0.08 m higher relative to the fixed G1, z 0.92, and 0.07 m closer, near edge 0.08) + rigid layout version (yaw -15 deg, dx -0.14 m, dy +0.02 m); start pose PARK_CUBES (cubes-specific code constant); NVIDIA EGL device per parent; fixed pelvis, legs, waist; robot-body gravcomp; not walking or balance; cubes passive dynamic bodies; park pose and orientation candidates are code constants; finger encoder reads clamped into jnt_range on the trusted side and logged; rendered on EGL device 1 (device 0 readback broken); pre-shape 0.5 and squeeze 0.3 are code constants; trusted-side render integrity guard (>=6 renders, last 3 identical and non-blank); DEV only'
sys.path.insert(0, ROOT)
os.environ['RSI_SAM3_REMOTE'] = '1'
os.environ.setdefault('RSI_SAM3_ADDR', __import__('tempfile').gettempdir() + '/sam3.sock')
sys.path.insert(0, RESET_ROOT + '/programs/g1/collect_cubes')  # original dir
import render_guard_b3_r1 as RG


class _NoCV2(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        raise RuntimeError('cv2 unavailable in G1 venv: cv2.' + k)


sys.modules['cv2'] = _NoCV2('cv2')
import mujoco as mj
from rsi import perception as P, perception_sam3 as S, perception_geo as G
from rsi.g1_suite.frame import TaskFrame
from rsi.g1_suite.hand.hand_arm import HandArmModel
from rsi.g1_suite.runtime.mujoco_backend import build_trusted
from rsi.g1_suite.runtime.robot_adapter import RobotAdapter, T_source_from_g1
from rsi.g1_suite.runtime.public_io import PublicIO
from rsi.g1_suite.runtime.trusted_runner import run_episode, vetoes
from rsi.g1_suite.runtime.demo_priors import load_demo_priors
from rsi.g1_suite.runtime.scoring_wire import Scoring

CLAMP = {'n_reads': 0, 'n_changed': 0, 'max_below_limit': {}, 'max_above_limit': {}, 'first_events': [], 'limits': {}}


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


def _cands():
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


ALT = _cands()


def _td(ty, roll_deg):
    # top-down approach tilted by ty (rad) about source y, rolled about the approach; lateral (closing) axis = (cos r, -sin r, 0) for ty = 0 (same parameterisation as probe_lift_v2_r2)
    c, s_ = math.cos(ty), math.sin(ty)
    a = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]) @ np.array([0, 0, -1.0])
    ref = np.array([1.0, 0, 0])
    l0 = ref - (ref @ a) * a
    l0 /= np.linalg.norm(l0)
    l1 = np.cross(a, l0)
    r = math.radians(roll_deg)
    l = math.cos(r) * l0 + math.sin(r) * l1
    return np.column_stack([np.cross(l, a), l, a])


# lift-proven on b3_cubes_v2 (lift_candidate_stats_z0.92.log: precision >= .9 given the chain ran), in order of the number of lifted cells
LIFT_SET = ('demo_close', 'demo_open', 'td:0:0', 'td:0:22.5', 'td:0.2:0', 'td:0.2:22.5', 'td:-0.2:67.5', 'td:0:67.5', 'td:0.2:67.5', 'td:-0.2:90', 'td:0:90', 'td:-0.2:112.5', 'td:0.2:90', 'td:0:112.5', 'td:0.2:112.5')
TILTS = (0.0, 0.2, -0.2)


def block_yaw(pts):
    # yaw (deg, mod 90) of a block's top face from its perceived points: minimum-area bounding square over 2 deg steps
    q = np.asarray(pts, float)[:, :2]
    q = q - q.mean(0)
    best = None
    for a in range(0, 90, 2):
        t = math.radians(a)
        R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
        u = q @ R.T
        area = np.ptp(u[:, 0]) * np.ptp(u[:, 1])
        if best is None or area < best[0]:
            best = (area, a)
    return float(best[1])


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


PRESHAPE = 0.5   # G1 pre-shape opening during descent (probe_cubes_grasp_r18: fully open fingers hit the table / miss)
SQUEEZE = 0.3    # closure beyond the calibrated 32 mm pad gap so the position servos load the cube (squeeze 0 -> no lift in r18)


def sgrip(io, side, opening, squeeze, duration):
    # same as io.grip but passes the hand module's squeeze through io.adapter.hand_targets (public actuator targets only)
    s = io.openings[side]
    n = max(1, int(math.ceil(duration / io.dt)))
    for k in range(n):
        o = s + (opening - s) * (k + 1) / n
        io.targets.update(io.adapter.hand_targets(io.task, side, o, squeeze=squeeze * (k + 1) / n))
        io.openings[side] = o
        io._step()


MIN_GOOD = .10   # (r7 .15) drawer r8 lesson: keep the prior order, accept the first chain whose worst arm-joint margin >= .15 rad
MIN_OK = .03
DROP_H = (.06, .08, .10)
DROP_R = .25     # drop offset as a fraction of the perceived bowl radius (r5 used .35)


def recover(io, err):
    # drawer r8 recover(): an arm encoder beyond its range makes FK/IK refuse; drive that joint's public target .06 rad inside its range and wait
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


def safe_move(io, side, p, R, dur):
    for k in range(3):
        try:
            io.move(side, p, R, dur)
            return
        except Exception as e:
            if io.error is not None:
                raise
            if k == 2 or not recover(io, e):
                raise


def margin_plan(io, side, pts, R):
    # IK chained through pts from the current encoders; worst arm-joint distance to its range (robot-only model) or None
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


def roll_of(R):
    l = np.asarray(R, float)[:, 1]
    return math.degrees(math.atan2(-l[1], l[0]))


def locate_bowl(io, proc):
    obs = io.observe()
    plane = G.fit_plane(obs)
    bowl = S.locate_object(proc, obs, 'bowl', plane=plane, max_height=.12)
    if bowl is None:
        return None, obs, plane
    bp = bowl['points']
    xy, r = P.fit_upright_cylinder(bp[bp[:, 2] > bowl['top'] - .015])
    return (np.asarray(xy, float), float(r), float(bowl['top'])), obs, plane


HOLD_RES = .008  # finger tracking residual (rad, thumb/index/middle joints of the side) after the lift: empty hand <= .001, held 32 mm block >= .015 (probe_lift_v2_r3, lift_res_r3/)
DROP_SET = ((-0.4, 45), (-0.2, 45), (-0.4, 0), (-0.2, 0), (0.0, 0), (0.0, 45), (-0.4, 90))  # drop-only orientations (diag_drop_v2_r1.log: largest margins over the bowl)
RETRIES = 2
ATTEMPT_T = 15.0  # s: a full pick-drop-home attempt takes about 11.6 s, then 1.5 s final wait and the 0.5 s scorer dwell
RIM_KEEP = .035  # m beyond the perceived bowl radius: such blocks are deferred and need a rim-clear candidate
MAX_ATT = 2  # failed attempts per block before it is deferred
T_DIRECT, T_HOME, T_PULL, T_END = 12.0, 16.0, 9.0, 4.0  # s budgets
BEHIND, PULL_TO, PULL_MAX, PULL_BOWL_CLEAR = .035, -0.04, 1, .05
PULL_FIRST_X = .03  # G1 on b3_cubes_v2: far reach limit at grasp height x ~ .00-.02 (kin_grid_r1)
MEM_R = .06
PULL_ORDER = ((-0.8, 0), (-0.6, 0), (-1.0, 0), (-0.8, 30), (-0.6, 30), (-1.0, 30), (-1.2, 0), (-1.2, 30))  # (approach tilt rad, roll deg): kin_drag_v2_r1
LATE_CLEAR = 0.0  # [smoke7] candidates with finger segment centres inside the rim are never used (smoke6 seed 9: clearance -0.015 tipped the bowl)
FINGER_CLEAR = .015  # m, finger segment centre outside the perceived bowl rim (horizontal), for segments below rim top + .02


def grasp_cands(named, yaw):
    # cube-yaw-aligned top-down rolls first, then the lift-proven set in precision order (smoke1 filtered near-diagonal ones; diagonal grasps still lift, lift_map_r1 yaw .6)
    r0 = (-yaw) % 90.0
    rolls = [r0, r0 + 90.0 if r0 <= 45.0 else r0 - 90.0]  # the two face-aligned closing directions
    out = [('yaw_td:%g:%.0f' % (t, r), _td(t, r)) for t in TILTS for r in rolls]  # TILTS outer loop: both rolls at tilt 0 first
    return out + [(n, named[n]) for n in LIFT_SET]


def finger_res(io, side):
    enc = io._enc()
    return float(sum(abs(io.targets[k] - enc[k]) for k in enc if k.startswith(side + '_') and any(h in k for h in ('thumb', 'index', 'middle'))))


def finger_clearance(io, side, c, gR, bowl_xy, bowl_r, top, up=.10):
    # robot-model-only: worst horizontal distance of the side's finger collision segments to the perceived bowl rim, over (c+.10 pre-shape, c pre-shape, c closed); None if not plannable
    enc = io._enc()
    worst = 9.
    for p, op in ((c + [0, 0, min(up, .10)], PRESHAPE), (c, PRESHAPE), (c, 0.0)):
        try:
            q = io.adapter.plan(enc, side, np.asarray(p, float), gR)
        except Exception:
            return None
        enc.update(q)
        e2 = dict(enc)
        e2.update(io.adapter.hand_targets(io.task, side, op))
        sf = io.adapter.hand.segment_fk(e2, side, io.adapter.T)
        if not sf.valid:
            return None
        for v in sf.segments.values():
            gp = np.asarray(v['geom_pos'], float)
            if gp[2] < top + .02:
                worst = min(worst, float(np.linalg.norm(gp[:2] - bowl_xy)) - bowl_r)
    return worst


UPS = (.15, .10, .08)  # approach and lift heights above the grasp point, normal first


def select(io, named, drop_alt, tried, c, yaw, mid, dxy, top, bowl_xy, bowl_r):
    # feasible (pick chain margin, drop plan, finger clearance >= FINGER_CLEAR) and late (clearance in [LATE_CLEAR, FINGER_CLEAR)) candidates for one block, at the highest approach height that has any
    first = 'left' if c[1] > mid else 'right'
    skip = {n for xy, n in tried if np.linalg.norm(xy - c[:2]) < .03}
    for up in UPS:
        feas, late = [], []
        for side in (first, 'right' if first == 'left' else 'left'):
            for gname, gR in grasp_cands(named, yaw):
                if gname in skip:
                    continue
                w = margin_plan(io, side, [c + [0, 0, up], c + [0, 0, min(up, .10)], c, c + [0, 0, up]], gR)
                if w is None or w < MIN_OK:
                    continue
                dp = plan_drop(io, side, c, gR, dxy, top, drop_alt) if up == UPS[0] else None
                mode = 'direct'
                if dp is None:
                    dp = plan_drop_home(io, side, dxy, top, gR, drop_alt)
                    mode = 'via_home'
                if dp is None:
                    continue
                fc = finger_clearance(io, side, c, gR, bowl_xy, bowl_r, top, up)
                if fc is None or fc < LATE_CLEAR:
                    continue
                (late if fc < FINGER_CLEAR else feas).append((min(w, dp[2]), side, gname, gR, dp[0], dp[1], mode, up))
                if len(feas) >= RETRIES:
                    return feas, late
        if feas or late:
            return feas, late
    return [], []


def margin_from(io, side, enc, pts, R):
    # as margin_plan but chained from the given encoder dict (robot model only)
    m = io.adapter.hand.m
    enc = dict(enc)
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


def plan_drop_home(io, side, dxy, top, gR, drop_alt):
    # drop planned from the home arm state (other joints from the current encoders)
    enc = io._enc()
    enc.update({k: v for k, v in io.home_q.items() if k.startswith(side + '_')})
    for hz in DROP_H:
        d = np.r_[dxy, top + hz]
        best = None
        for dR in [gR] + drop_alt:
            wd = margin_from(io, side, enc, [d + [0, 0, .04], d], dR)
            if wd is not None and (best is None or wd > best[2]):
                best = (d, dR, wd)
            if best is not None and best[2] >= MIN_GOOD:
                break
        if best is not None and best[2] >= MIN_OK:
            return best
    return None


def plan_drop(io, side, c, gR, dxy, top, drop_alt):
    for hz in DROP_H:
        d = np.r_[dxy, top + hz]
        best = None
        for dR in [gR] + drop_alt:
            wd = margin_plan(io, side, [c + [0, 0, .15], d + [0, 0, .04], d], dR)
            if wd is not None and (best is None or wd > best[2]):
                best = (d, dR, wd)
            if best is not None and best[2] >= MIN_GOOD:
                break
        if best is not None and best[2] >= MIN_OK:
            return best
    return None


def near_count(lst, xy, r=.03):
    return sum(1 for q in lst if np.linalg.norm(np.asarray(q) - np.asarray(xy)[:2]) < r)


def plan_pull(io, side, c, plane, bowl_xy, bowl_r):
    # chain above far side -> down -> drag -> lift; heights follow the perceived table plane; None if not plannable or too close to the bowl
    if c[0] <= PULL_TO + .03:
        return None
    pc, n = plane
    tz = lambda x, y: float(pc[2] - (n[0] * (x - pc[0]) + n[1] * (y - pc[1])) / n[2])
    zc = c[2] - tz(c[0], c[1])
    p = np.array([c[0] + BEHIND, c[1], tz(c[0] + BEHIND, c[1]) + zc])
    q = np.array([PULL_TO + BEHIND, c[1], tz(PULL_TO + BEHIND, c[1]) + zc])
    for t in np.linspace(0, 1, 9):
        x = (1 - t) * c[0] + t * PULL_TO
        if np.linalg.norm(np.array([x, c[1]]) - bowl_xy) < bowl_r + PULL_BOWL_CLEAR:
            return None
    for t, r in PULL_ORDER:
        R = _td(t, r)
        w = margin_plan(io, side, [p + [0, 0, .08], p, q, q + [0, 0, .08]], R)
        if w is not None and w >= MIN_OK:
            return p, q, R, (t, r)
    return None


def controller(io):
    park(io)
    proc = S.get_proc()
    close = next(e for e in io.demo['events'] if e['kind'] == 'close')
    opened = next(e for e in io.demo['events'] if e['kind'] == 'open' and e['time'] > close['time'])
    named = {'demo_close': close['rotation'], 'demo_open': opened['rotation']}
    for n in LIFT_SET:
        if n.startswith('td:'):
            _, t, r = n.split(':')
            named[n] = _td(float(t), float(r))
    drop_alt = [_td(t, r) for t, r in DROP_SET] + [named['demo_open']]
    mid = 0.5 * (io.demo['home']['left'][1, 3] + io.demo['home']['right'][1, 3])
    done = 0
    att = []  # xy of failed attempts (empty grasp, chain failure, no plan)
    pulls = []  # xy of pull-in starts and ends
    tried = []  # (xy, candidate name) of empty grasps; skipped for a block within 3 cm
    memory = []  # last perceived block centres (runtime perception only)
    yaw_mem = []  # (xy, perceived yaw) of blocks seen with enough points
    moved = []  # xy of blocks this controller dropped (now in or near the bowl)
    gone = []  # xy of retired locations (unseen block grasped empty, or block carried off before a chain failure)
    bowl_last = None
    empty_confirm = 0
    for rnd in range(MAX_ROUNDS):
        if io.now() > CAP - T_END - T_PULL:
            io.log('stop_time', t=io.now())
            break
        bw, obs, plane = locate_bowl(io, proc)
        if bw is None:
            if bowl_last is None:
                raise RuntimeError('perception failed bowl')
            bw = bowl_last
        bowl_last = bw
        bowl_xy, bowl_r, top = bw
        io.log('bowl', rnd=rnd, xy=bowl_xy, r=bowl_r, top=top)
        blocks, _ = S.find_blocks(proc, obs, plane=plane, bowl=(bowl_xy, bowl_r))
        seen = [np.asarray(b['center'], float) for b in blocks]
        for mc in memory:
            if all(np.linalg.norm(mc[:2] - q[:2]) > MEM_R for q in seen) and all(np.linalg.norm(mc[:2] - f) > .03 for f in moved + gone) and np.linalg.norm(mc[:2] - bowl_xy) > bowl_r + RIM_KEEP:
                ym = [q[1] for q in yaw_mem if np.linalg.norm(q[0] - mc[:2]) < .03]
                blocks.append(dict({'center': mc, 'points': np.zeros((0, 3)), 'remembered': True}, **({'yaw': ym[0]} if ym else {})))
                io.log('remembered_block', rnd=rnd, center=mc)
        memory = [np.asarray(b['center'], float) for b in blocks]
        for b in blocks:
            if 'yaw' not in b and len(b['points']) >= 20:
                b['yaw'] = block_yaw(b['points'][b['points'][:, 2] > float(b['center'][2]) - .006])
        yaw_mem = [(np.asarray(b['center'][:2], float), b['yaw']) for b in blocks if 'yaw' in b] + [q for q in yaw_mem if all(np.linalg.norm(q[0] - np.asarray(b['center'][:2])) > .03 for b in blocks if 'yaw' in b)]
        for b in blocks:
            b['near_rim'] = bool(np.linalg.norm(b['center'][:2] - bowl_xy) < bowl_r + RIM_KEEP)
            b['n_att'] = near_count(att, b['center'])
        blocks = [b for b in blocks if b['n_att'] <= MAX_ATT]
        io.log('round', rnd=rnd, n_blocks=len(blocks), centers=[b['center'] for b in blocks][:6], n_att=[b['n_att'] for b in blocks][:6], near_rim=[b['near_rim'] for b in blocks][:6])
        if not blocks:
            if empty_confirm == 0:
                empty_confirm = 1
                io.log('final_check', rnd=rnd)
                io.wait(.5)
                continue
            break
        empty_confirm = 0
        ang = rnd * 2.399963
        dxy = bowl_xy + DROP_R * bowl_r * np.array([math.cos(ang), math.sin(ang)])
        # normal blocks first (nearest to bowl), deferred (near rim or >= MAX_ATT failures) last
        order = sorted(blocks, key=lambda q: (q['center'][0] <= PULL_FIRST_X, q['near_rim'] or q['n_att'] >= MAX_ATT, float(np.linalg.norm(q['center'][:2] - bowl_xy))))
        pick = None
        pull = None
        for b in order:
            c = np.array([b['center'][0], b['center'][1], float(b['center'][2]) - CUBE_H / 2.0])
            yaw = float(b['yaw']) if 'yaw' in b else 0.0
            b['yaw'] = yaw
            feas, late = select(io, named, drop_alt, tried, c, yaw, mid, dxy, top, bowl_xy, bowl_r)
            if b['near_rim'] and not all(q['near_rim'] for q in blocks):
                late = []
            if feas:
                pick = (b, c, yaw, feas, late)
                break
            if late and pick is None:
                pick = (b, c, yaw, feas, late)
                continue
            if not feas and not late and pull is None and near_count(pulls, c) < PULL_MAX:
                side = 'left' if c[1] > mid else 'right'
                pp = plan_pull(io, side, c, plane, bowl_xy, bowl_r)
                if pp is not None:
                    pull = (b, c, side, pp)
                    if c[0] > PULL_FIRST_X:
                        break
                    continue
            if not feas and not late:
                att.append(np.asarray(c[:2], float))
                io.log('no_plan', rnd=rnd, target=c)
        if pull is not None and (pick is None or pull[1][0] > PULL_FIRST_X):
            b, c, side, (p, q, R, tr) = pull
            if io.now() + T_PULL + T_END > CAP:
                io.log('stop_time_attempt', t=io.now())
                break
            pulls += [np.asarray(c[:2], float), np.asarray(q[:2] - [BEHIND, 0], float)]
            io.log('pull_in', rnd=rnd, side=side, target=c, to=q, tilt_roll=tr)
            try:
                sgrip(io, side, 0.0, 0.0, .3)
                safe_move(io, side, p + [0, 0, .08], R, 1.5)
                safe_move(io, side, p, R, 1.0)
                safe_move(io, side, q, R, 1.5)
                safe_move(io, side, q + [0, 0, .08], R, .8)
            except Exception as e:
                if io.error is not None:
                    raise
                io.log('pull_failed', rnd=rnd, err=str(e)[:90])
                att.append(np.asarray(c[:2], float))
            sgrip(io, side, PRESHAPE, 0.0, .3)
            io.home(side, 2.0)
            memory = [m for m in memory if np.linalg.norm(m[:2] - c[:2]) > .03]
            continue
        if pick is None:
            continue
        b, c, yaw, feas, late = pick
        if len(feas) < RETRIES:
            feas += late[:RETRIES - len(feas)]
        feas = feas[:1 if b.get('remembered') else max(1, MAX_ATT + 1 - b['n_att'])]
        io.log('grasp_selection', rnd=rnd, yaw=yaw, n_late=len(late), deferred=bool(b['near_rim'] or b['n_att'] >= MAX_ATT), chosen=[[f[1], f[2], round(float(f[0]), 3), f[6], f[7]] for f in feas], target=c)
        held = False
        side = feas[0][1]
        for k, (w, side, gname, gR, d, dR, mode, up) in enumerate(feas):
            if io.now() + (T_HOME if mode == 'via_home' else T_DIRECT) + T_END > CAP:
                io.log('stop_time_attempt', t=io.now())
                break
            res_ok = False
            try:
                sgrip(io, side, PRESHAPE, 0.0, .3)
                safe_move(io, side, c + [0, 0, up], gR, 1.5 if k == 0 else .8)
                if up > .10:
                    safe_move(io, side, c + [0, 0, .10], gR, .6)
                safe_move(io, side, c, gR, 1.0)
                sgrip(io, side, 0.0, SQUEEZE, .8)
                io.wait(.3)
                safe_move(io, side, c + [0, 0, up], gR, 1.0)
                io.wait(.2)
                res = finger_res(io, side)
                io.log('hold_check', rnd=rnd, cand=gname, side=side, res=res)
                if res < HOLD_RES:
                    tried.append((np.asarray(c[:2], float), gname))
                    att.append(np.asarray(c[:2], float))
                    if b.get('remembered'):
                        gone.append(np.asarray(c[:2], float))
                        memory = [m for m in memory if np.linalg.norm(m[:2] - c[:2]) > .03]
                    sgrip(io, side, PRESHAPE, 0.0, .3)
                    continue
                res_ok = True
                if mode == 'via_home':
                    io.home(side, 2.0)
                    io.log('via_home', rnd=rnd, res=finger_res(io, side))
                safe_move(io, side, d + [0, 0, .04], dR, 1.5)
                safe_move(io, side, d, dR, .8)
                held = True
                break
            except Exception as e:
                if io.error is not None:
                    raise
                io.log('chain_failed', rnd=rnd, cand=gname, err=str(e)[:90])
                if res_ok and mode == 'direct':
                    try:
                        io.home(side, 2.0)
                        if finger_res(io, side) >= HOLD_RES:
                            dp = plan_drop_home(io, side, dxy, top, gR, drop_alt)
                            if dp is not None:
                                d, dR = dp[0], dp[1]
                                safe_move(io, side, d + [0, 0, .04], dR, 1.5)
                                safe_move(io, side, d, dR, .8)
                                io.log('rescued_via_home', rnd=rnd)
                                held = True
                                break
                    except Exception as e2:
                        if io.error is not None:
                            raise
                        io.log('rescue_failed', rnd=rnd, err=str(e2)[:80])
                tried.append((np.asarray(c[:2], float), gname))
                att.append(np.asarray(c[:2], float))
                if res_ok:
                    gone.append(np.asarray(c[:2], float))
                    memory = [m for m in memory if np.linalg.norm(m[:2] - c[:2]) > .03]
                sgrip(io, side, PRESHAPE, 0.0, .4)
                io.home(side, 1.5)
                break
        if not held:
            io.home(side, 1.2)
            continue
        io.log('release', rnd=rnd, drop=d)
        moved.append(np.asarray(c[:2], float))
        sgrip(io, side, 1.0, 0.0, .5)
        io.wait(.3)
        done += 1
        up = None
        for hz in (.12, .08, .05):
            try:
                safe_move(io, side, d + [0, 0, hz], dR, .8)
                up = hz
                break
            except Exception as e:
                if io.error is not None:
                    raise
        io.log('retreat', rnd=rnd, up=up)
        io.home(side, 2.0)
    io.log('rounds_done', released=done)
    io.wait(3.0)
    io.observe()


def J(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def main():
    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    outp = '%s/seed_%02d.json' % (args.outdir, args.seed)
    rec = {'seed': args.seed, 'task': TASK, 'tag': TAG, 'mujoco': mj.__version__, 'disclosure': DISC, 'park_pose': list(PARK), 'success': False, 'error': None}
    try:
        lay = json.load(open(args.layout_file))['collect_cubes'][args.layout_key][str(args.seed)]['poses']
        fr = json.load(open(WORLD + '/collect_cubes/build.json'))['frame']
        tf = TaskFrame(fr['top_center_source'], fr['gravity_source'], fr['mount_g1'])
        RG.install_render_ring()
        cb, h = build_trusted(SCENE, lay, 640, 480, family=None)
        RG.guard_observe(cb)
        m_, d_ = h.model, h.data
        for side, sg in (('left', 1.0), ('right', -1.0)):
            v = [PARK[0], sg * PARK[1], sg * PARK[2], PARK[3], sg * PARK[4], PARK[5], sg * PARK[6]]
            for n, x in zip(JN, v):
                jn = '%s_%s_joint' % (side, n)
                d_.qpos[m_.jnt_qposadr[mj.mj_name2id(m_, mj.mjtObj.mjOBJ_JOINT, jn)]] = x
                d_.ctrl[mj.mj_name2id(m_, mj.mjtObj.mjOBJ_ACTUATOR, 'a_' + jn)] = x
        mj.mj_forward(m_, d_)
        install_clamp(cb, m_)
        adapter = RobotAdapter(HandArmModel.from_robot_xml(), T_source_from_g1(tf), TASK)
        demo, prov = load_demo_priors(TASK)
        io = PublicIO(TASK, cb, adapter, demo, CAP, instruction='Put all the cubes into the bowl.')
        sc = Scoring(TASK, h)
        frames = []
        if args.video:
            rr = mj.Renderer(h.model, height=480, width=640)
            oc = mj.MjvCamera()
            oc.type = mj.mjtCamera.mjCAMERA_FREE
            oc.lookat[:] = [0.03, -0.5, 0.2]
            oc.distance = 1.3
            oc.azimuth = 180.0
            oc.elevation = -35.0
            orig = cb.step
            cnt = [0]

            def vstep(t):
                orig(t)
                cnt[0] += 1
                if cnt[0] % 50 == 0:
                    rr.update_scene(h.data, camera='head_cam')
                    a = rr.render().copy()
                    rr.update_scene(h.data, camera=oc)
                    b = rr.render().copy()
                    frames.append(np.concatenate([a, b], axis=1))
            cb.step = vstep
        res = run_episode(io, controller, sc.snapshot, CAP)
        ev = sc.evaluate(res)
        sr = ev['scorer_result']
        rec.update(vetoes=vetoes(res), controller_error=res['controller_error'], latched_error=res['latched_error'], time_limit=res['time_limit'], numerical_invalid=res['numerical_invalid'],
                   dwell_ok=res['dwell_ok'], dwell_reason=res['dwell_reason'], sim_time=io.now(), scorer_vetoed=ev['vetoed'], scorer_result=sr, records=io.records)
        rec['success'] = bool(sr is not None and sr.get('success') and not ev['vetoed'])
        pr = next((r for r in io.records if r['event'] == 'bowl'), None)
        if pr is not None:
            rec['posthoc_bowl_xy_error_m'] = float(np.linalg.norm(np.array(pr['xy']) - np.array(lay['object_bowl'][:2])))
        if args.video and frames:
            vp = '%s/frames_seed_%02d.npy' % (args.outdir, args.seed)
            np.save(vp, np.stack(frames))
            rec['frames_file'] = vp
            rec['n_frames'] = len(frames)
    except Exception as e:
        rec['error'] = repr(e)
        rec['traceback'] = traceback.format_exc()
    rec['encoder_clamp'] = CLAMP
    rec['render_guard'] = dict(RG.STATS)
    rec['wall_sec'] = round(time.time() - t0, 1)
    json.dump(rec, open(outp, 'w'), indent=1, default=J)
    print('SEED_DONE', args.seed, 'success', rec['success'], 'error', rec['error'], 'vetoes', rec.get('vetoes'), 'clamped_reads', CLAMP['n_changed'], 'guard', RG.STATS, 'wall', rec['wall_sec'], flush=True)


main()
