# tdbs_common_r1: trusted-side harness shared by drawer_r1_*, box_r1_* and sweep_r1_* (G1, DEV only). Not imported by any controller function.
# What it adds on the trusted side WITHOUT editing the frozen shared runtime (rsi/g1_suite/runtime/*, hand/*, scoring/*):
#  1. G1 grasp profiles for the three tasks (contact priors, disclosed): a RobotAdapter subclass ProfiledAdapter that declares (kind, size) for FK/IK
#     and adds a per-task squeeze to the finger closure in hand_targets (hand_arm.finger_targets(squeeze=...) is the frozen hand API; squeeze scales
#     with closure so opening 1 stays fully open). PublicIO accepts it (isinstance RobotAdapter).
#  2. Fixed park pose (g1_common_r2.PARK, code constant, not per-seed) applied before any step; float-level finger encoder undershoot clamp (g1_common_r2).
#  3. World + layout selection for b2 or b3 (b3 = demo table world, built by another sub-task; read only).
#  4. Sweep: build_trusted(family='sweep') puts the v3 Monitor in the step path (CompositeGuard, BACKEND_READY Addendum r3). Scoring() builds its own
#     Monitor that never sees a step, so here the sweep is scored with the stepped monitor h.monitor through the frozen g1_scorer functions
#     (sweep_snapshot / sweep_evaluate / sweep_final_success); thresholds and semantics untouched.
# Disclosure: G1 pelvis, legs and waist fixed (no freejoint); robot-body gravcomp in the world; not walking or balance. Objects are passive dynamic bodies.
RESET_ROOT = __import__('os').environ.get('RESET_ROOT') or next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'eval.py').is_file()).as_posix()
import sys, os, json, time, types, traceback
import numpy as np

ROOT = RESET_ROOT
R0 = ROOT + '/programs/g1'
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, R0 + '/common', HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ['RSI_SAM3_REMOTE'] = '1'
os.environ.setdefault('RSI_SAM3_ADDR', __import__('tempfile').gettempdir() + '/sam3_g1_r4.sock')


class _NoCV2(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        raise RuntimeError('cv2 unavailable in G1 venv: cv2.' + k)


if 'cv2' not in sys.modules:
    try:
        import cv2  # noqa: F401
    except Exception:
        sys.modules['cv2'] = _NoCV2('cv2')
import mujoco as mj
import g1_common_r2 as C  # read-only import of the frozen apple-era helpers (PARK, clamp, park pose)
from rsi.g1_suite.frame import TaskFrame
from rsi.g1_suite.hand.hand_arm import HandArmModel
from rsi.g1_suite.runtime import robot_adapter as RA
from rsi.g1_suite.runtime.mujoco_backend import build_trusted
from rsi.g1_suite.runtime.robot_adapter import RobotAdapter, T_source_from_g1, UnsupportedGraspProfile
from rsi.g1_suite.runtime.public_io import PublicIO
from rsi.g1_suite.runtime.trusted_runner import run_episode, vetoes, CAPS
from rsi.g1_suite.runtime.demo_priors import load_demo_priors
from rsi.g1_suite.runtime.scoring_wire import Scoring

INSTR = {'open_drawer_and_place_object': 'Open the drawer and put the orange cube inside.',
         'sweep_objects': 'Sweep the orange cube into the dustpan with the brush.',
         'lift_box': 'Put the carton upright on top of the wooden box.'}


class ProfiledAdapter(RobotAdapter):
    # profiles: {side: (kind, size, squeeze)}; kind/size select the frozen hand calibration (cube 32 mm or sphere 68 mm pinch frame) for FK, IK and fingers.
    # squeeze s: closure f = (1 - opening) * (F_MAX + s) via finger_targets(opening=None is not used); implemented as finger_targets(opening, squeeze=s*(1-opening)).
    def __init__(self, hand, T, task, profiles):
        super().__init__(hand, T, task)
        self.profiles = dict(profiles)
        self.active = 'right'

    def profile(self):
        return self.profiles[self.active][:2]

    def tcp(self, named, side, passive=None):
        self.active = side
        return super().tcp(named, side, passive)

    def plan(self, named, side, position, rotation):
        self.active = side
        return super().plan(named, side, position, rotation)

    def hand_targets(self, task_unused, side, opening, squeeze=0.0):
        kind, size, s = self.profiles[side]
        r = self.hand.finger_targets(side, opening, kind, size, squeeze=float(s) * (1.0 - float(opening)))
        if not r.valid:
            raise UnsupportedGraspProfile('finger_targets_invalid:' + str(r.reason))
        return {(k[2:] if k.startswith('a_') else k): float(v) for k, v in r.ctrl.items()}


def world_paths(world, task):
    return '%s/worlds/g1/%s/%s/scene.xml' % (RESET_ROOT, world, task), '%s/worlds/g1/%s/%s/build.json' % (RESET_ROOT, world, task)


def layout(world, task, seed):
    if not 0 <= seed <= 19:
        raise SystemExit('only DEV seeds 0-19 are allowed')
    wl = '%s/worlds/g1/%s/layouts.json' % (RESET_ROOT, world)  # a new world may ship its own frozen DEV layouts
    if os.path.exists(wl):
        d = json.load(open(wl))
        if task in d:
            return d[task]['dev'][str(seed)]['poses'], wl
    if task == 'sweep_objects':
        src = R0 + '/scene_alignment/a02_sweep_layouts/sweep_layouts.json'
        return json.load(open(src))['dev'][str(seed)]['poses'], src
    src = R0 + '/probe01_dev_layouts/layouts.json'
    return json.load(open(src))[task]['dev'][str(seed)]['poses'], src


def run_seed(task, seed, outdir, controller, profiles, tag, world='b2', video=False, diag=False, extra_disc=''):
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    outp = '%s/seed_%02d.json' % (outdir, seed)
    if os.path.exists(outp):
        raise SystemExit('refusing to overwrite ' + outp)
    CAP = CAPS[task]
    disc = ('fixed pelvis, legs, waist; robot-body gravcomp; not walking or balance; objects passive dynamic bodies; park pose code constant; '
            'G1 grasp profiles %s installed trusted-side (ProfiledAdapter, contact prior); finger encoder reads clamped into jnt_range and logged; DEV only. %s') % (profiles, extra_disc)
    rec = {'seed': seed, 'task': task, 'tag': tag, 'world': world, 'mujoco': mj.__version__, 'disclosure': disc, 'park_pose': list(C.PARK),
           'profiles': profiles, 'success': False, 'error': None, 'diag_used': bool(diag)}
    try:
        lay, lay_src = layout(world, task, seed)
        rec['layout_source'] = lay_src
        scene, bj = world_paths(world, task)
        fr = json.load(open(bj))['frame']
        tf = TaskFrame(fr['top_center_source'], fr['gravity_source'], fr['mount_g1'])
        fam = 'sweep' if task == 'sweep_objects' else None
        cb, h = build_trusted(scene, lay, 640, 480, family=fam)
        C.set_park_pose(h)
        C.install_clamp(cb, h.model)
        adapter = ProfiledAdapter(HandArmModel.from_robot_xml(), T_source_from_g1(tf), task, profiles)
        demo, prov = load_demo_priors(task)
        rec['demo_provenance'] = prov
        io = PublicIO(task, cb, adapter, demo, CAP, instruction=INSTR[task])
        sc = Scoring(task, h)
        m_, d_ = h.model, h.data
        frames = []
        if video:
            rr = mj.Renderer(m_, height=480, width=640)
            oc = mj.MjvCamera()
            oc.type = mj.mjtCamera.mjCAMERA_FREE
            tb = d_.geom_xpos[mj.mj_name2id(m_, mj.mjtObj.mjOBJ_GEOM, 'table')]
            oc.lookat[:] = [tb[0] - 0.05, tb[1], tb[2] + 0.12]
            oc.distance = 1.45
            oc.azimuth = 200.0
            oc.elevation = -38.0
            orig = cb.step
            cnt = [0]

            def vstep(t):
                orig(t)
                cnt[0] += 1
                if cnt[0] % 50 == 0:
                    rr.update_scene(d_, camera='head_cam')
                    a = rr.render().copy()
                    rr.update_scene(d_, camera=oc)
                    b = rr.render().copy()
                    frames.append(np.concatenate([a, b], axis=1))
            cb.step = vstep
        diagrows = []
        if diag:  # trusted-side diagnosis only (object truth + TCP at each controller log event); never visible to the controller; diag runs are not scored batches
            fb = [i for i in range(m_.nbody) if m_.body(i).name.startswith('object_') and m_.body_jntnum[i]]
            olog = io.log

            def dlog(event, **k):
                olog(event, **k)
                row = {'t': io.now(), 'event': event, 'obj': {m_.body(i).name: d_.xpos[i].tolist() for i in fb}}
                for j in range(m_.njnt):
                    if m_.jnt_type[j] == 2 and not m_.body(m_.jnt_bodyid[j]).name.startswith(('left_', 'right_')):
                        row['slide_' + m_.joint(j).name] = float(d_.qpos[m_.jnt_qposadr[j]])
                for sd in ('left', 'right'):
                    try:
                        row['tcp_' + sd] = adapter.tcp(cb.encoders(), sd)[:3, 3].tolist()
                    except Exception as e:
                        row['tcp_' + sd] = repr(e)[:60]
                cons = set()
                for c in d_.contact[:d_.ncon]:
                    b1, b2 = m_.body(m_.geom_bodyid[c.geom1]).name, m_.body(m_.geom_bodyid[c.geom2]).name
                    if (b1.startswith('object_') or b2.startswith('object_')) and c.dist < .002:
                        cons.add(tuple(sorted((b1, b2))))
                row['contacts'] = sorted(cons)
                diagrows.append(row)
            io.log = dlog
        res = run_episode(io, controller, sc.snapshot if fam is None else (lambda: sc.S.sweep_snapshot(m_, d_, sc.roles)), CAP)
        rec.update(vetoes=vetoes(res), controller_error=res['controller_error'], latched_error=res['latched_error'], time_limit=res['time_limit'],
                   numerical_invalid=res['numerical_invalid'], dwell_ok=res['dwell_ok'], dwell_reason=res['dwell_reason'], sim_time=io.now(), records=io.records)
        if fam is None:
            ev = sc.evaluate(res)
            sr = ev['scorer_result']
            rec.update(scorer_vetoed=ev['vetoed'], scorer_result=sr, scorer_family=ev['family'])
            rec['success'] = bool(sr is not None and sr.get('success') and not ev['vetoed'])
        else:
            S = sc.S
            nv, ce = bool(res['numerical_invalid']), res['controller_error']
            vetoed = bool(res['latched_error'] or res['time_limit'] or nv or ce or not res['dwell_ok'])
            scored = S.sweep_evaluate(res['history'], h.monitor.summary(), nv) if res['dwell_ok'] else None
            fin = S.sweep_final_success(scored, ce, nv) if scored is not None else (False, ['no_dwell'])
            rec.update(scorer_vetoed=vetoed, scorer_result=scored, scorer_family='sweep', sweep_final=list(fin), monitor_summary=h.monitor.summary())
            rec['success'] = bool(fin[0] and not vetoed)
        if diag:
            rec['diag_truth'] = diagrows
        if video and frames:
            vp = '%s/frames_seed_%02d.npy' % (outdir, seed)
            np.save(vp, np.stack(frames))
            rec['frames_file'] = vp
            rec['n_frames'] = len(frames)
    except Exception as e:
        rec['error'] = repr(e)
        rec['traceback'] = traceback.format_exc()
    rec['encoder_clamp'] = C.CLAMP
    rec['wall_sec'] = round(time.time() - t0, 1)
    json.dump(rec, open(outp, 'w'), indent=1, default=C.J)
    sr = rec.get('scorer_result') or {}
    print('SEED_DONE', task, seed, 'success', rec['success'], 'error', rec['error'], 'vetoes', rec.get('vetoes'),
          'failed', sr.get('failed_checks') if isinstance(sr, dict) else None, 'ctl_err', rec.get('controller_error'), 'wall', rec['wall_sec'], flush=True)
    return rec
