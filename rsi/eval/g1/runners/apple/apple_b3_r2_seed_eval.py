# apple_b3_r2_seed: = apple_b3_r1_seed with start pose PARK_CLEAN (old PARK penetrated the b3 table, PARK_PENETRATION_NOTICE.md). apple_b3_r1 note: ONE b3 (demo-table world) collect_apple DEV episode (called by apple_b3_r1_parent.py). Controller = apple_r4 (frozen on b2) unchanged; only the world (b3), the trusted-side render_guard_b3_r1 and the SAM3 socket differ. apple_r4 note: Same frozen G1 runtime, park pose and candidate set as apple_r1_seed.py.
# r4 change (trusted side only): hand-finger encoder reads are clamped into each joint's own jnt_range before they reach the planner; every clamp is recorded (value before and after).
# Reason: r1 reported ik_invalid:encoder_out_of_limit:right_thumb_metacarpal_joint for all 44 candidates; the cause (float-level undershoot of the 0.0 lower limit) is an inference that this run verifies by logging the raw values.
# The frozen hand model, its 1e-9 tolerance, the scorer and the success criteria are untouched. Arm encoders are not clamped.
# Runtime side (controller function) uses only io: rendered RGB-D, calibration, encoders, demo priors. It never reads object truth, scorer or contacts. Park pose and candidate orientations are code constants, not per-seed.
# Disclosure: G1 pelvis, legs and waist are fixed (no freejoint); robot-body gravcomp is in the world. This is not walking or balance. Apple uses the source explicit inertia in the world. DEV seeds only. No VAL, TEST or hardware.
# Note: cv2 is absent in the G1 venv; a stub that raises on any use lets rsi.perception import. The apple path never calls cv2.
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
SCENE = RESET_ROOT + '/worlds/g1/b3/collect_apple/scene.xml'
TASK = 'collect_apple'
TAG = 'apple_b3_r2'
CAP = 30.0
PARK = (0.0, 0.6, 0.0, 0.3, 0.0, -0.4, 0.0)  # PARK_CLEAN: old PARK with wrist_pitch -0.4; 0 robot-table penetration on b3 (a21), 0 robot-object on DEV0-19
JN = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']
HAND_KEYS = ('thumb', 'index', 'middle', 'ring', 'pinky')
DISC = 'b3 demo-table world (mount 0.42,0,0.84); rendered on NVIDIA EGL device 2 with trusted-side render guard; fixed pelvis, legs, waist; robot-body gravcomp; not walking or balance; apple passive dynamic body with source explicit inertia; park pose PARK_CLEAN and orientation candidates are code constants; finger encoder reads clamped into jnt_range on the trusted side and logged; DEV only'
sys.path.insert(0, ROOT)
os.environ['RSI_SAM3_REMOTE'] = '1'
os.environ.setdefault('RSI_SAM3_ADDR', __import__('tempfile').gettempdir() + '/sam3_g1_cubes_r2.sock')
sys.path.insert(0, RESET_ROOT + '/programs/g1/collect_apple')  # original dir
import render_guard_b3_r1 as RG  # private rsi SAM3 service


class _NoCV2(types.ModuleType):
    def __getattr__(self, k):
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


def controller(io):
    park(io)
    obs = io.observe()
    plane = G.fit_plane(obs)
    proc = S.get_proc()
    apple = S.locate_object(proc, obs, 'apple', plane=plane, max_height=.12)
    bowl = S.locate_object(proc, obs, 'bowl', plane=plane, max_height=.12)
    if apple is None or bowl is None:
        raise RuntimeError('perception failed apple=%s bowl=%s' % (apple is not None, bowl is not None))
    apple_c = P.fit_sphere(apple['points'], .034)
    bp = bowl['points']
    bowl_xy, bowl_r = P.fit_upright_cylinder(bp[bp[:, 2] > bowl['top'] - .015])
    io.log('perceived', apple=apple_c, bowl=bowl_xy, bowl_r=bowl_r, bowl_top=bowl['top'])
    close = next(e for e in io.demo['events'] if e['kind'] == 'close')
    opened = next(e for e in io.demo['events'] if e['kind'] == 'open' and e['time'] > close['time'])
    grasp_off = close['position'] - close['objects']['apple']
    cands = [('demo_close', close['rotation']), ('demo_open', opened['rotation'])] + ALT
    mid = 0.5 * (io.demo['home']['left'][1, 3] + io.demo['home']['right'][1, 3])
    first = 'left' if apple_c[1] > mid else 'right'
    order = (first, 'right' if first == 'left' else 'left')
    io.log('arm_order', order=list(order), midline=float(mid))
    chosen = None
    nskip = 0
    for side in order:
        for gname, gR in cands:
            tgt = apple_c + (grasp_off if gname == 'demo_close' else np.zeros(3))
            ok = True
            for p in (tgt + [0, 0, .10], tgt, tgt + [0, 0, .15]):
                try:
                    io.adapter.plan(io._enc(), side, np.asarray(p, float), gR)
                except Exception as e_plan:
                    ok = False
                    io.log('plan_fail', side=side, cand=gname, err=str(e_plan)[:120])
                    break
            if not ok:
                nskip += 1
                continue
            try:
                io.grip(side, 1., .3)
                io.move(side, tgt + [0, 0, .10], gR, 1.5)
                io.move(side, tgt, gR, 1.0)
            except Exception as e:
                if io.error is not None:
                    raise
                io.log('grasp_approach_failed', side=side, cand=gname, err=str(e)[:80])
                continue
            chosen = (side, gname, gR, tgt)
            break
        if chosen:
            break
    io.log('grasp_selection', skipped_ik=nskip, chosen=None if chosen is None else [chosen[0], chosen[1]])
    if chosen is None:
        raise RuntimeError('no grasp candidate executed')
    side, gname, gR, tgt = chosen
    io.grip(side, 0., .6)
    lifted = False
    for h in (.15, .10, .06):
        try:
            io.move(side, tgt + [0, 0, h], gR, 1.0)
            lifted = True
            break
        except Exception as e:
            if io.error is not None:
                raise
            io.log('lift_retry', h=h, err=str(e)[:80])
    if not lifted:
        raise RuntimeError('lift unreachable')
    top = float(bowl['top'])
    rel = None
    for hrel in (.08, .12):
        drop = np.r_[bowl_xy, top + hrel]
        for rname, rR in [(gname, gR), ('demo_open', opened['rotation'])] + cands:
            try:
                io.move(side, drop + [0, 0, .03], rR, 1.5)
                io.move(side, drop, rR, .8)
                rel = (rname, rR, drop)
                break
            except Exception as e:
                if io.error is not None:
                    raise
                io.log('drop_retry', cand=rname, h=hrel, err=str(e)[:80])
        if rel:
            break
    if rel is None:
        raise RuntimeError('drop unreachable')
    io.log('release_selection', cand=rel[0])
    io.grip(side, 1., .6)
    for h in (.10, .06, .03):
        try:
            io.move(side, rel[2] + [0, 0, h], rel[1], .8)
            break
        except Exception as e:
            if io.error is not None:
                raise
            io.log('retreat_retry', h=h, err=str(e)[:80])
    io.wait(2.5)
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
        lay = json.load(open(args.layout_file))['collect_apple'][args.layout_key][str(args.seed)]['poses']
        fr = json.load(open(RESET_ROOT + '/worlds/g1/b3/collect_apple/build.json'))['frame']
        tf = TaskFrame(fr['top_center_source'], fr['gravity_source'], fr['mount_g1'])
        RG.install_render_ring()
        cb, h = build_trusted(SCENE, lay, 640, 480, family=None)
        RG.guard_observe(cb)
        # trusted side, before any step: start the robot AT the fixed park pose (code constant, not per-seed)
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
        io = PublicIO(TASK, cb, adapter, demo, CAP, instruction='Put the apple into the bowl.')
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
        pr = next((r for r in io.records if r['event'] == 'perceived'), None)
        if pr is not None:
            rec['posthoc_apple_perception_error_m'] = float(np.linalg.norm(np.array(pr['apple']) - np.array(lay['object_apple'][:3])))
            rec['posthoc_bowl_xy_error_m'] = float(np.linalg.norm(np.array(pr['bowl']) - np.array(lay['object_bowl'][:2])))
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
    print('SEED_DONE', args.seed, 'success', rec['success'], 'error', rec['error'], 'vetoes', rec.get('vetoes'), 'clamped_reads', CLAMP['n_changed'], 'wall', rec['wall_sec'], flush=True)


main()
