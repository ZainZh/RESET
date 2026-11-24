'''Task-parameterized LIFT2 world builder.

Starts from the SOURCE task scene.xml and removes ONLY source-robot-linked content:
  - the left_base/right_base subtrees and the source 'main' camera,
  - any actuator/equality/contact/sensor/tendon/custom element that references a name from those subtrees (or a gripper_*/left_*/right_* custom),
  - keyframes (their qpos layout belongs to the old robot; disclosed),
  - meshes used only by removed geoms.
Everything else (table, bench, lights, options, object bodies incl. static fixtures and slide-jointed children, their joints,
non-robot equality/contact/custom entries, assets) is kept byte-for-byte as XML. The LIFT2 subtree is inserted at the declared mount.
After compile, verify_mechanics() compares every non-robot body/joint/geom/option between source and target models.
Usage: python build_world.py TASK_ID OUT_DIR   (OUT_DIR must not exist)
'''
import copy, hashlib, json, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from rsi.lift2_suite.common.embodiment import lift2 as R  # noqa: E402
from rsi.lift2_suite.common import specs  # noqa: E402

SRC_BASES = ('left_base', 'right_base')
SRC_CAMERAS = ('main',)
ROBOT_CUSTOM_PREFIX = ('gripper', 'left_', 'right_')
CONTAINERS = ('actuator', 'equality', 'contact', 'sensor', 'tendon', 'custom', 'keyframe')
OV_POS, OV_TARGET = np.array([.95, -1.05, .8]), np.array([0., -.42, .08])  # presentation-only free camera
RZ180 = np.diag([-1., -1., 1.])


class MechanicsMismatch(RuntimeError):
    pass


def _fmt(v):
    return ' '.join('%.10g' % x for x in v)


def _lookat(pos, target):
    f = target - pos; f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.]); x /= np.linalg.norm(x); y = np.cross(x, f)
    return np.r_[x, y]


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def table(record):
    t = record['table']
    return np.array(t['rotation']), np.array(t['center']), np.array(t['half_size'])


def mount_transform(record, base_height, mount):
    Rt, ctop, half = table(record)
    sgn = 1. if mount['edge'].startswith('+X') else -1.
    T = np.eye(4); T[:3, :3] = Rt @ (RZ180 if sgn > 0 else np.eye(3))
    T[:3, 3] = ctop + Rt @ [sgn * (half[0] + mount['overhang_m'] - mount['platform_back_x_m']), mount['lateral_v_m'], 0.] + Rt[:, 2] * base_height
    return T


def _tokens(el):
    out = set()
    for e in el.iter():
        for v in e.attrib.values():
            out.add(v); out.update(v.split())
    return out


def build(task_id, out_dir, mount=None):
    spec = specs.runnable(task_id); mount = dict(mount or spec['mount'])
    src_dir = specs.world_dir(task_id); src_scene = src_dir / 'scene.xml'
    tree = ET.parse(src_scene); root = tree.getroot(); wb = root.find('worldbody')
    root.set('model', f'{task_id}_lift2')
    gone = [b for b in wb if (b.tag == 'body' and b.get('name') in SRC_BASES) or (b.tag == 'camera' and b.get('name') in SRC_CAMERAS)]
    if not any(b.get('name') in SRC_BASES for b in gone):
        raise ValueError('source robot bases not found in ' + str(src_scene))
    names, gone_meshes = set(), set()
    for b in gone:
        for e in b.iter():
            if e.get('name'):
                names.add(e.get('name'))
            if e.tag == 'geom' and e.get('mesh'):
                gone_meshes.add(e.get('mesh'))
        wb.remove(b)
    rec = {'removed': [], 'preserved': []}; n_removed_eq = 0
    for tag in CONTAINERS:
        for c in root.findall(tag):
            for e in list(c):
                nm = e.get('name', '')
                drop = tag == 'keyframe' or bool(_tokens(e) & names) or (tag == 'custom' and nm.startswith(ROBOT_CUSTOM_PREFIX))
                (rec['removed'] if drop else rec['preserved']).append([tag, e.tag, nm])
                if drop:
                    c.remove(e); n_removed_eq += tag == 'equality'
            if len(c) == 0:
                root.remove(c)
    asset = root.find('asset'); comp = root.find('compiler')
    mdir = comp.get('meshdir', '') if comp is not None else ''; tdir = comp.get('texturedir', '') if comp is not None else ''
    kept_mesh = {g.get('mesh') for g in root.iter('geom') if g.get('mesh')}
    if asset is not None:
        for e in list(asset):
            if e.tag == 'mesh' and e.get('name') in gone_meshes and e.get('name') not in kept_mesh:
                asset.remove(e); continue
            if e.get('file'):
                sub = mdir if e.tag == 'mesh' else tdir if e.tag == 'texture' else ''
                e.set('file', str((src_dir / sub / e.get('file')).resolve()))
    if comp is not None:
        for k in ('meshdir', 'texturedir'):
            comp.attrib.pop(k, None)
    rob = ET.parse(R.MJCF).getroot()
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    for mesh in rob.find('asset').findall('mesh'):
        mesh = copy.deepcopy(mesh); mesh.set('file', str((R.MODEL / mesh.get('file')).resolve())); asset.append(mesh)
    base = copy.deepcopy(rob.find('worldbody').find("body[@name='%s']" % R.BASE_BODY))
    T = mount_transform(specs.record(task_id), R.BASE_HEIGHT, mount)
    base.set('pos', _fmt(T[:3, 3])); base.set('quat', _fmt(np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)))
    if R.GRAVITY_COMPENSATION:
        actuated = {a.get('joint') for a in rob.find('actuator')}
        for b in base.iter('body'):
            b.set('gravcomp', '1')
        for j in base.iter('joint'):
            if j.get('name') in actuated:
                j.set('actuatorgravcomp', 'true')
    for arm, s in R.ARMS.items():
        tip = next(b for b in base.iter('body') if b.get('name') == s['tip'])
        ET.SubElement(tip, 'site', name=f'{arm}_tcp', pos=_fmt(s['tool_xyz']), size='.004', rgba='1 0 0 0')
    wb.append(base)
    ET.SubElement(wb, 'camera', name='overview_lift2', pos=_fmt(OV_POS), xyaxes=_fmt(_lookat(OV_POS, OV_TARGET)))
    for tag in ('actuator', 'equality', 'contact'):
        src_c = rob.find(tag)
        if src_c is None:
            continue
        dst = root.find(tag)
        if dst is None:
            dst = ET.SubElement(root, tag)
        for e in src_c:
            dst.append(copy.deepcopy(e))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=False)
    ET.indent(tree, space='  ')
    scene = out_dir / 'scene.xml'; tree.write(scene, encoding='utf-8')
    mech = verify_mechanics(src_scene, scene, names, n_removed_eq)
    info = {'task': task_id, 'scene': str(scene), 'scene_sha256': sha256_file(scene), 'source_scene': str(src_scene), 'source_scene_sha256': sha256_file(src_scene),
        'mount': mount, 'mount_T_task_base': T.tolist(), 'removed_source_names': sorted(names), 'elements': rec, 'mechanics': mech, 'checks': static_checks(task_id, scene, T)}
    (out_dir / 'build.json').write_text(json.dumps(info, indent=1))  # written before the check so a failing build keeps its evidence
    if not mech['ok']:
        raise MechanicsMismatch(f"{task_id}: compiled non-robot mechanics differ from source ({mech['n_diffs']} diffs): {mech['diffs'][:10]}")
    return info


def verify_mechanics(src_scene, tgt_scene, removed_names, n_removed_eq):
    '''Structural invariants: every non-robot body/joint/dof/geom and the solver options compile identically in source and target.'''
    a = mujoco.MjModel.from_xml_path(str(src_scene)); b = mujoco.MjModel.from_xml_path(str(tgt_scene))
    diffs = []

    def cmp(what, x, y):
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            diffs.append(what)
    for f in ('timestep', 'impratio', 'tolerance', 'iterations', 'ls_iterations', 'noslip_iterations', 'cone', 'solver', 'integrator', 'jacobian', 'gravity'):
        cmp('opt.' + f, getattr(a.opt, f), getattr(b.opt, f))
    nv = {int(mujoco.mjtJoint.mjJNT_FREE): 6, int(mujoco.mjtJoint.mjJNT_BALL): 3}; n = 0
    for i in range(a.nbody):
        nm = a.body(i).name or 'world'
        if nm in removed_names:
            continue
        try:
            j = b.body(nm).id
        except KeyError:
            diffs.append('missing body ' + nm); continue
        n += 1
        for f in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos', 'body_quat'):
            cmp(f'{nm}.{f}', getattr(a, f)[i], getattr(b, f)[j])
        cmp(f'{nm}.parent', a.body(int(a.body_parentid[i])).name, b.body(int(b.body_parentid[j])).name)
        cmp(f'{nm}.njnt', a.body_jntnum[i], b.body_jntnum[j])
        for k in range(int(a.body_jntnum[i])):
            ja, jb = int(a.body_jntadr[i]) + k, int(b.body_jntadr[j]) + k
            for f in ('jnt_type', 'jnt_range', 'jnt_axis', 'jnt_pos', 'jnt_solref', 'jnt_solimp', 'jnt_stiffness', 'jnt_margin', 'jnt_limited'):
                cmp(f'{nm}.{f}', getattr(a, f)[ja], getattr(b, f)[jb])
            da, db = int(a.jnt_dofadr[ja]), int(b.jnt_dofadr[jb])
            for d in range(nv.get(int(a.jnt_type[ja]), 1)):
                for f in ('dof_damping', 'dof_frictionloss', 'dof_armature'):
                    cmp(f'{nm}.{f}', getattr(a, f)[da + d], getattr(b, f)[db + d])
        ga, gb = np.where(a.geom_bodyid == i)[0], np.where(b.geom_bodyid == j)[0]
        if len(ga) != len(gb):
            diffs.append(f'{nm}.ngeom'); continue
        for x, y in zip(ga, gb):
            for f in ('geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_solref', 'geom_solimp', 'geom_contype', 'geom_conaffinity', 'geom_condim', 'geom_margin', 'geom_gap', 'geom_priority', 'geom_solmix'):
                cmp(f'{nm}.{f}', getattr(a, f)[x], getattr(b, f)[y])
            if a.geom_type[x] == mujoco.mjtGeom.mjGEOM_MESH:
                cmp(f'{nm}.mesh', a.mesh(int(a.geom_dataid[x])).name, b.mesh(int(b.geom_dataid[y])).name)
    eq = 'compared' if n_removed_eq == 0 else 'not_compared_removed_equalities_present'
    if n_removed_eq == 0:
        cmp('neq', a.neq, b.neq - sum(1 for e in range(b.neq) if b.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and b.eq_obj1id[e] >= 0 and b.jnt_bodyid[b.eq_obj1id[e]] in robot_ids(b)))
        for f in ('eq_type', 'eq_data', 'eq_solref', 'eq_solimp', 'eq_active0'):
            cmp(f, getattr(a, f), getattr(b, f)[:a.neq])
    return {'ok': not diffs, 'diffs': diffs[:50], 'n_diffs': len(diffs), 'bodies_compared': n, 'equality_check': eq, 'source_neq': int(a.neq), 'target_neq': int(b.neq)}


def robot_ids(m, base=R.BASE_BODY):
    b0 = m.body(base).id; out = set()
    for b in range(1, m.nbody):
        a = b
        while a and a != b0:
            a = int(m.body_parentid[a])
        if a == b0:
            out.add(b)
    return out


def static_checks(task_id, scene, T):
    '''Static checks after compile (no dynamics): mount pose, platform vs tabletop plane, platform-table signed distance.'''
    m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    Rt, ctop, half = table(specs.record(task_id)); b = m.body(R.BASE_BODY).id
    out = {'nbody': m.nbody, 'nu': m.nu, 'neq': m.neq, 'nmocap': m.nmocap, 'timestep': m.opt.timestep, 'gravity': m.opt.gravity.tolist()}
    out['base_pos_err_m'] = float(np.abs(d.xpos[b] - T[:3, 3]).max()); out['base_rot_err'] = float(np.abs(d.xmat[b].reshape(3, 3) - T[:3, :3]).max())
    rb = sorted(robot_ids(m)); pts = []; dist = []; gt = m.geom('table').id
    for g in np.where(np.isin(m.geom_bodyid, rb))[0]:
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        if m.body(int(m.geom_bodyid[g])).name == R.BASE_BODY and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        if m.geom_contype[g]:
            dist.append(mujoco.mj_geomDistance(m, d, int(g), gt, .05, None))
    if pts:
        P = (np.concatenate(pts) - ctop) @ Rt
        out['platform_min_height_above_tabletop_m'] = float(P[:, 2].min())
    out['robot_table_signed_distance_m'] = float(min(dist)) if dist else None
    return out


if __name__ == '__main__':
    try:
        info = build(sys.argv[1], sys.argv[2])
    except MechanicsMismatch as e:
        print('MECHANICS MISMATCH:', e); sys.exit(1)
    print(json.dumps({'mechanics': info['mechanics'], 'checks': info['checks']}, indent=1))
