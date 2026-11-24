# tdbs_seed_r2 (r2 = r1 + TDBS_PARK env overrides the fixed park pose g1_common_r2.PARK in-process, code constant per batch, recorded in rec park_pose; g1_common_r2.py itself is not edited): ONE DEV episode (task, controller module, seed) in its own process; called by tdbs_parent_r1.py. Trusted side; see tdbs_common_r1.py.
RESET_ROOT = __import__('os').environ.get('RESET_ROOT') or next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'eval.py').is_file()).as_posix()
import sys, os, json, argparse, importlib
ap = argparse.ArgumentParser()
ap.add_argument('--task', required=True)
ap.add_argument('--ctl', required=True)          # controller module name next to this file, e.g. drawer_r1_ctl
ap.add_argument('--profiles', required=True)     # JSON {side: [kind, size, squeeze]} (trusted-side grasp-profile contact prior)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--outdir', required=True)
ap.add_argument('--world', default='b2')
ap.add_argument('--tag', default='')
ap.add_argument('--video', action='store_true')
ap.add_argument('--diag', action='store_true')
ap.add_argument('--layout-file', required=True)
ap.add_argument('--layout-key', required=True, choices=['dev', 'test'])
a = ap.parse_args()
sys.path.insert(0, RESET_ROOT + '/programs/g1/drawer_box_sweep')  # original dir
import tdbs_common_r1 as T


def _eval_layout(world, task, seed):  # replaces tdbs_common_r1.layout (DEV-only file lookup) by --layout-file/--layout-key; nothing else
    if not 0 <= seed <= 49:
        raise SystemExit('only seeds 0-49 are allowed')
    return json.load(open(a.layout_file))[task][a.layout_key][str(seed)]['poses'], os.path.abspath(a.layout_file) + '#' + a.layout_key


T.layout = _eval_layout
if os.environ.get('TDBS_PARK'):
    T.C.PARK = tuple(float(x) for x in os.environ['TDBS_PARK'].split(','))
ctl = importlib.import_module(a.ctl)
prof = {k: tuple(v) for k, v in json.loads(a.profiles).items()}
T.run_seed(a.task, a.seed, a.outdir, ctl.run, prof, a.tag or a.ctl, world=a.world, video=a.video, diag=a.diag)
