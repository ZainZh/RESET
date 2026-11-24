# towc_seed_r1 = copy of g1task_r3_seed.py (the seed script the r11 batch actually launched; read-only reuse) with: --world (default b3; the source hard-codes worlds/b2), world recorded in rec, and the in-process override g1_common_<ver>.PARK = PARK_CLEAN = (0,0.6,0,0.3,0,-0.4,0) (or --park), mirrored for the right arm by set_park_pose; g1_common_r11.py is not edited. The controller's own C.park(io) therefore goes to the clean pose too. Controller and common are imported read-only from ../common/.
# g1task_r3_seed (generic: --ver selects g1_common_<ver>.py and ctl_<task>_<ver>.py; same trusted side as r2): ONE b2 DEV episode for one task in its own process (called by g1task_r3_parent.py).
# Runtime side (controller function) uses only io: rendered RGB-D, calibration, encoders, demo priors. It never reads object truth, scorer or contacts. Park pose, candidate orientations and grasp-profile priors are code constants (g1_common_<ver>.py).
# Trusted side (main): builds the world, starts the robot AT the fixed park pose, applies the DEV layout once, clamps float-level finger encoder undershoot at the read and logs before/after, scores with the independent G1 scorer.
# Disclosure: G1 pelvis, legs and waist are fixed (no freejoint); robot-body gravcomp is in the world. This is not walking or balance. DEV seeds only. No VAL, TEST or hardware.
# Note: cv2 is absent in the G1 venv; a stub that raises on any use lets rsi.perception import.
RESET_ROOT = __import__('os').environ.get('RESET_ROOT') or next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'eval.py').is_file()).as_posix()
import sys, os, json, time, types, argparse, traceback, importlib
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--task', required=True)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--outdir', required=True)
ap.add_argument('--tag', default='tower_r11_cleanpark')
ap.add_argument('--ver', default='r11')
ap.add_argument('--video', action='store_true')
ap.add_argument('--world', default='b3')
ap.add_argument('--park', default='0,0.6,0,0.3,0,-0.4,0')
ap.add_argument('--diag', action='store_true')  # trusted-side diagnosis only: object truth + TCP at each controller log event, written to rec['diag_truth']; never visible to the controller; not used for scored DEV batches
ap.add_argument('--layout-file', required=True)
ap.add_argument('--layout-key', required=True, choices=['dev', 'test'])
args = ap.parse_args()
HERE = RESET_ROOT + '/programs/g1/build_tower'  # original dir on sys.path
ROOT = RESET_ROOT
R0 = ROOT + '/programs/g1'
sys.path.insert(0, ROOT)
sys.path.insert(0, R0 + '/common')
sys.path.insert(0, HERE)
os.environ['RSI_SAM3_REMOTE'] = '1'
os.environ.setdefault('RSI_SAM3_ADDR', __import__('tempfile').gettempdir() + '/sam3_g1_r4.sock')
INSTR = {'build_tower': 'Stack the purple cube on the purple bar, then put the yellow cube on top.',
         'stack_cups': 'Put the purple cup into the blue cup.',
         'put_mug_on_box': 'Put the mug on the wooden box.',
         'open_drawer_and_place_object': 'Open the drawer and put the orange cube inside.',
         'sweep_objects': 'Sweep the orange cube into the dustpan with the brush.',
         'lift_box': 'Put the carton upright on top of the wooden box.'}
if args.task not in INSTR:
    raise SystemExit('unsupported task for this seed script: ' + args.task)


class _NoCV2(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        raise RuntimeError('cv2 unavailable in G1 venv: cv2.' + k)


sys.modules['cv2'] = _NoCV2('cv2')
import mujoco as mj
C = importlib.import_module('g1_common_' + args.ver)
C.PARK = tuple(float(x) for x in args.park.split(','))
assert len(C.PARK) == 7
from rsi.g1_suite.frame import TaskFrame
from rsi.g1_suite.hand.hand_arm import HandArmModel
from rsi.g1_suite.runtime import robot_adapter as RA
from rsi.g1_suite.runtime.mujoco_backend import build_trusted
from rsi.g1_suite.runtime.robot_adapter import RobotAdapter, T_source_from_g1
from rsi.g1_suite.runtime.public_io import PublicIO
from rsi.g1_suite.runtime.trusted_runner import run_episode, vetoes, CAPS
from rsi.g1_suite.runtime.demo_priors import load_demo_priors
from rsi.g1_suite.runtime.scoring_wire import Scoring

C.install_profiles(RA)
CTL = importlib.import_module('ctl_%s_%s' % (args.task, args.ver))
TASK = args.task
CAP = CAPS[TASK]
DISC = 'fixed pelvis, legs, waist; robot-body gravcomp; not walking or balance; objects passive dynamic bodies; park pose, orientation candidates and grasp-profile priors %s are code constants; finger encoder reads clamped into jnt_range on the trusted side and logged; DEV only' % (C.PROFILE_PRIORS.get(TASK),)


def main():
    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    outp = '%s/seed_%02d.json' % (args.outdir, args.seed)
    rec = {'seed': args.seed, 'task': TASK, 'tag': args.tag, 'ver': args.ver, 'world': args.world, 'mujoco': mj.__version__, 'disclosure': DISC, 'park_pose': list(C.PARK), 'grasp_profile': C.PROFILE_PRIORS.get(TASK), 'success': False, 'error': None}
    try:
        lay = json.load(open(args.layout_file))[TASK][args.layout_key][str(args.seed)]['poses']
        fr = json.load(open('%s/worlds/g1/%s/%s/build.json' % (RESET_ROOT, args.world, TASK)))['frame']
        tf = TaskFrame(fr['top_center_source'], fr['gravity_source'], fr['mount_g1'])
        scene = '%s/worlds/g1/%s/%s/scene.xml' % (RESET_ROOT, args.world, TASK)
        cb, h = build_trusted(scene, lay, 640, 480, family=None)
        C.set_park_pose(h)
        C.install_clamp(cb, h.model)
        adapter = RobotAdapter(HandArmModel.from_robot_xml(), T_source_from_g1(tf), TASK)
        demo, prov = load_demo_priors(TASK)
        io = PublicIO(TASK, cb, adapter, demo, CAP, instruction=INSTR[TASK])
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
        diag = []
        if args.diag:
            fb = [i for i in range(h.model.nbody) if h.model.body_jntnum[i] and h.model.jnt_type[h.model.body_jntadr[i]] == mj.mjtJoint.mjJNT_FREE]
            olog = io.log

            def dlog(event, **k):
                olog(event, **k)
                row = {'t': io.now(), 'event': event, 'obj': {mj.mj_id2name(h.model, mj.mjtObj.mjOBJ_BODY, i): h.data.xpos[i].tolist() for i in fb}}
                for sd in ('left', 'right'):
                    try:
                        row['tcp_' + sd] = adapter.tcp(cb.encoders(), sd)[:3, 3].tolist()
                    except Exception as e:
                        row['tcp_' + sd] = repr(e)[:60]
                diag.append(row)
            io.log = dlog
        res = run_episode(io, CTL.controller, sc.snapshot, CAP)
        if args.diag:
            rec['diag_truth'] = diag
        ev = sc.evaluate(res)
        sr = ev['scorer_result']
        rec.update(vetoes=vetoes(res), controller_error=res['controller_error'], latched_error=res['latched_error'], time_limit=res['time_limit'], numerical_invalid=res['numerical_invalid'],
                   dwell_ok=res['dwell_ok'], dwell_reason=res['dwell_reason'], sim_time=io.now(), scorer_vetoed=ev['vetoed'], scorer_result=sr, records=io.records)
        rec['success'] = bool(sr is not None and sr.get('success') and not ev['vetoed'])
        if args.video and frames:
            vp = '%s/frames_seed_%02d.npy' % (args.outdir, args.seed)
            np.save(vp, np.stack(frames))
            rec['frames_file'] = vp
            rec['n_frames'] = len(frames)
    except Exception as e:
        rec['error'] = repr(e)
        rec['traceback'] = traceback.format_exc()
    rec['encoder_clamp'] = C.CLAMP
    rec['wall_sec'] = round(time.time() - t0, 1)
    json.dump(rec, open(outp, 'w'), indent=1, default=C.J)
    print('SEED_DONE', TASK, args.seed, 'success', rec['success'], 'error', rec['error'], 'vetoes', rec.get('vetoes'), 'clamped_reads', C.CLAMP['n_changed'], 'wall', rec['wall_sec'], flush=True)


main()
