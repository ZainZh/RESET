# One-shot demonstration priors for the G1 suite (r2). Trusted-side loader; never imported by controllers.
# Rebuilds exactly the source rsi.env.demo_priors structure {events, home, objects0} from the SOURCE reference (reference.npz) using the SOURCE xTrainer FK
# (twin.practice.motion.RobotKinematics built from the source scene). xTrainer joints never go through G1 FK. No current DEV layout or object state is read.
# Poses are in the unchanged SOURCE frame. Provenance (full SHA256 of every source file read) is returned separately and never enters the public priors.
# mujoco/twin imports are lazy so importing this package stays model-free.
import sys
from pathlib import Path
import numpy as np
from .. import specs

ROOT = Path(__file__).resolve().parents[3]
SIDES = ('left', 'right')


class PriorError(RuntimeError):
    pass


def _finite(a, tag):
    a = np.asarray(a, float)
    if not np.all(np.isfinite(a)):
        raise PriorError('nonfinite:' + tag)
    return a


def load_demo_priors(task):
    # returns (priors, provenance)
    world = Path(specs.get(task)['world'])
    ref_path, scene_path, rec_path = world / 'reference.npz', world / 'scene.xml', world / 'reconstruction.json'
    for p in (ref_path, scene_path, rec_path):
        if not p.is_file():
            raise PriorError('missing_source_file:' + str(p))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import mujoco
    from twin.practice.motion import RobotKinematics
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    kin = RobotKinematics.from_model(model)
    ref = np.load(ref_path)
    t, q, g = _finite(ref['timestamps'], 'timestamps'), _finite(ref['joints'], 'joints'), _finite(ref['gripper'], 'gripper')
    if q.ndim != 3 or q.shape[1:] != (2, 6) or q.shape[0] != t.shape[0] or g.shape != (t.shape[0], 2):
        raise PriorError('reference_shape')
    names = [str(n) for n in ref['object_names']]
    poses, cam_t = _finite(ref['object_poses'], 'object_poses'), _finite(ref['camera_timestamps'], 'camera_timestamps')
    events = []
    for a, arm in enumerate(SIDES):
        for k in range(1, len(t)):
            for kind, cond in (('close', g[k - 1, a] > .5 >= g[k, a]), ('open', g[k - 1, a] <= .5 < g[k, a])):
                if cond:
                    T = kin.fk(arm, q[k, a])
                    events.append({'arm': arm, 'kind': kind, 'time': float(t[k]), 'position': T[:3, 3].copy(), 'rotation': T[:3, :3].copy()})
    events.sort(key=lambda e: e['time'])
    home = {arm: kin.fk(arm, q[0, a]) for a, arm in enumerate(SIDES)}
    objects0 = {n: poses[0, i, :3, 3].copy() for i, n in enumerate(names)}
    for e in events:  # demo object positions at the event, tracked from the real recording (not current DEV objects)
        k = int(np.argmin(np.abs(cam_t - e['time'])))
        e['objects'] = {n: poses[k, i, :3, 3].copy() for i, n in enumerate(names)}
    prov = {'task': task, 'source': 'xTrainer FK on reference.npz', 'reference': str(ref_path), 'reference_sha256': specs.sha256(ref_path),
            'scene': str(scene_path), 'scene_sha256': specs.sha256(scene_path), 'record': str(rec_path), 'record_sha256': specs.sha256(rec_path),
            'kinematics_source': str(ROOT / 'twin/practice/motion.py'), 'kinematics_sha256': specs.sha256(ROOT / 'twin/practice/motion.py'),
            'n_events': len(events), 'uses_current_dev_objects': False}
    return {'events': events, 'home': home, 'objects0': objects0}, prov
