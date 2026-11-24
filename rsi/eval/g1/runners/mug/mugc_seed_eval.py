# mugc_seed_r1 = copy of tasks_mug_cups/mug_r4_seed.py (read-only reuse) with ONE change: the start/park pose g1_common_r2.PARK is overridden in-process by PARK_CLEAN=(0,0.6,0,0.3,0,-0.4,0) (mirrored for the right arm by set_park_pose), or by --park; g1_common_r2.py is not edited. The controller's own C.park(io) therefore also goes to the clean pose (same in-process override as tdbs_seed_r2.py for drawer r9). Controller, grasp installer, calibration and hand points are imported read-only from ../mug_cups_common/.
# mug_r4_seed (mug_r2_seed + trusted-side post-hoc 'on' metrics written to the record after the episode only, never to the controller: mug lowest vertex, box highest vertex, mug xy vs box xy bounds): ONE G1 put_mug_on_box DEV episode in its own process (called by cups_r1_parent.py --script mug_r4_seed.py). Controller module is chosen by --ctl next to this file; the module names its GRASP installer.
# r8 trusted-side addition: a head-camera render with an invalid depth frame (seen: 307192 of 307200 pixels nan/-0, while the next render of the same state is clean, diag_depth2_cups_r7_seed3) is re-rendered up to 3 times; every retry is recorded in obs_retries. Physics state is not touched.
# Runtime side (controller function) uses only io: rendered RGB-D, calibration, encoders, demo priors. It never reads object truth, scorer or contacts.
# Trusted side (main): builds the world (--world b2 or b3), starts the robot AT the fixed park pose, applies the DEV layout once, clamps float-level finger encoder undershoot at the read
# (logged before/after, via baseline g1_common_r2, read only), installs the task grasp shape in-process (contact prior, disclosed; hand_arm.py and runtime files on disk are not edited), scores with the independent G1 scorer.
# Disclosure: G1 pelvis, legs and waist are fixed (no freejoint); robot-body gravcomp is in the world. This is not walking or balance. DEV seeds only. No VAL, TEST or hardware.
# Note: cv2 is absent in the G1 venv; a stub that raises on any use lets rsi.perception import.
RESET_ROOT = __import__('os').environ.get('RESET_ROOT') or next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'eval.py').is_file()).as_posix()
import sys, os, json, time, types, argparse, traceback, importlib
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--outdir', required=True)
ap.add_argument('--world', default='b2')
ap.add_argument('--ctl', default='mug_r4_ctl')
ap.add_argument('--tag', default='mug_r4_cleanpark')
ap.add_argument('--video', action='store_true')
ap.add_argument('--park', default='0,0.6,0,0.3,0,-0.4,0')
ap.add_argument('--layout-file', required=True)
ap.add_argument('--layout-key', required=True, choices=['dev', 'test'])
args = ap.parse_args()
HERE = RESET_ROOT + '/programs/g1/put_mug_on_box'  # original dir on sys.path
ROOT = RESET_ROOT
R0 = ROOT + '/programs/g1'
sys.path.insert(0, ROOT)
sys.path.insert(0, R0 + '/common')
sys.path.insert(0, R0 + '/mug_cups_common')
sys.path.insert(0, HERE)
os.environ['RSI_SAM3_REMOTE'] = '1'
os.environ.setdefault('RSI_SAM3_ADDR', __import__('tempfile').gettempdir() + '/sam3_g1_r4.sock')
TASK = 'put_mug_on_box'
INSTR = 'Put the mug on the wooden box.'
if not 0 <= args.seed <= 49:
    raise SystemExit('only seeds 0-49 are allowed')


class _NoCV2(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        raise RuntimeError('cv2 unavailable in G1 venv: cv2.' + k)


sys.modules['cv2'] = _NoCV2('cv2')
import mujoco as mj
import g1_common_r2 as C
PARK_CLEAN = tuple(float(x) for x in args.park.split(','))
assert len(PARK_CLEAN) == 7
C.PARK = PARK_CLEAN
from rsi.g1_suite.frame import TaskFrame
from rsi.g1_suite.hand import hand_arm as HAM
from rsi.g1_suite.hand.hand_arm import HandArmModel
from rsi.g1_suite.runtime import robot_adapter as RA
from rsi.g1_suite.runtime.mujoco_backend import build_trusted
from rsi.g1_suite.runtime.robot_adapter import RobotAdapter, T_source_from_g1
from rsi.g1_suite.runtime.public_io import PublicIO
from rsi.g1_suite.runtime.trusted_runner import run_episode, vetoes, CAPS
from rsi.g1_suite.runtime.demo_priors import load_demo_priors
from rsi.g1_suite.runtime.scoring_wire import Scoring

CTL = importlib.import_module(args.ctl)
GRASP_INFO = importlib.import_module(CTL.GRASP).install(HAM, RA, TASK)
OBS_RETRIES = []


def install_obs_retry(cb):
    orig = cb.observe

    def obs():
        for k in range(4):
            try:
                return orig()
            except Exception as e:
                if 'invalid_observation_frame' not in str(e) or k == 3:
                    raise
                OBS_RETRIES.append({'t': float(cb.time()), 'attempt': k + 1, 'err': str(e)})
    cb.observe = obs

CAP = CAPS[TASK]
DISC = ('fixed pelvis, legs, waist; robot-body gravcomp; not walking or balance; objects passive dynamic bodies; park pose and orientation candidates are code constants; '
        'grasp shape %s installed in-process on the trusted side (contact prior); finger encoder reads clamped into jnt_range on the trusted side and logged; invalid head-camera frames re-rendered and logged; DEV only' % (GRASP_INFO,))


def main():
    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    outp = '%s/seed_%02d.json' % (args.outdir, args.seed)
    rec = {'seed': args.seed, 'task': TASK, 'tag': args.tag, 'world': args.world, 'ctl': args.ctl, 'mujoco': mj.__version__, 'disclosure': DISC, 'park_pose': list(C.PARK),
           'grasp_shape': GRASP_INFO, 'success': False, 'error': None}
    try:
        lay = json.load(open(args.layout_file))[TASK][args.layout_key][str(args.seed)]['poses']
        wd = '%s/worlds/g1/%s/%s' % (RESET_ROOT, args.world, TASK)
        fr = json.load(open(wd + '/build.json'))['frame']
        tf = TaskFrame(fr['top_center_source'], fr['gravity_source'], fr['mount_g1'])
        cb, h = build_trusted(wd + '/scene.xml', lay, 640, 480, family=None)
        C.set_park_pose(h)
        C.install_clamp(cb, h.model)
        install_obs_retry(cb)
        adapter = RobotAdapter(HandArmModel.from_robot_xml(), T_source_from_g1(tf), TASK)
        demo, prov = load_demo_priors(TASK)
        io = PublicIO(TASK, cb, adapter, demo, CAP, instruction=INSTR)
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
        res = run_episode(io, CTL.controller, sc.snapshot, CAP)
        ev = sc.evaluate(res)
        sr = ev['scorer_result']
        rec.update(vetoes=vetoes(res), controller_error=res['controller_error'], latched_error=res['latched_error'], time_limit=res['time_limit'], numerical_invalid=res['numerical_invalid'],
                   dwell_ok=res['dwell_ok'], dwell_reason=res['dwell_reason'], sim_time=io.now(), scorer_vetoed=ev['vetoed'], scorer_result=sr, records=io.records)
        rec['success'] = bool(sr is not None and sr.get('success') and not ev['vetoed'])
        from twin.practice.layouts import world_vertices
        sv = world_vertices(h.model, h.data, 'object_box'); cv = world_vertices(h.model, h.data, 'object_mug'); mx = h.data.body('object_mug').xpos
        rec['posthoc_on'] = dict(mug_min_z=float(cv[:, 2].min()), box_max_z=float(sv[:, 2].max()), margin_z=float(cv[:, 2].min() - (sv[:, 2].max() - .01)),
                                 mug_xy=mx[:2].tolist(), box_lo=sv[:, :2].min(0).tolist(), box_hi=sv[:, :2].max(0).tolist(), box_top_corner=sv[int(np.argmax(sv[:, 2]))].tolist())
        if args.video and frames:
            vp = '%s/frames_seed_%02d.npy' % (args.outdir, args.seed)
            np.save(vp, np.stack(frames))
            rec['frames_file'] = vp
            rec['n_frames'] = len(frames)
    except Exception as e:
        rec['error'] = repr(e)
        rec['traceback'] = traceback.format_exc()
    rec['encoder_clamp'] = C.CLAMP
    rec['obs_retries'] = OBS_RETRIES
    rec['wall_sec'] = round(time.time() - t0, 1)
    json.dump(rec, open(outp, 'w'), indent=1, default=C.J)
    sr = rec.get('scorer_result') or {}
    print('SEED_DONE', TASK, args.seed, 'success', rec['success'], 'error', rec['error'], 'ctl_err', str(rec.get('controller_error'))[:160], 'vetoes', rec.get('vetoes'),
          'failed', sr.get('failed_checks'), 'on', rec.get('posthoc_on'), 'wall', rec['wall_sec'], flush=True)


main()
