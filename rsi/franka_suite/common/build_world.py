'''Franka suite world builder (simulation only, DEV only).

Starts from the SOURCE xTrainer task scene (same source worlds as the LIFT2 suite, rsi/lift2_suite/common/specs.py) and
removes ONLY: the source robot subtrees left_base/right_base, the source 'main' camera, robot-only actuators/customs, and the
optical bench under the board (optical_bench_top, bench_slot_*, bench_leg_*, bench_isolator_*: the user asked to keep only the desk,
the aluminium/bench parts may go). Kept unchanged: the source 'table' board geom (collision support named 'table', size, pose,
friction, solref/solimp), tilted gravity, solver options, every object body/joint/geom/asset, lights.
Added (all disclosed in README):
- desk borders (fix B): four boxes coplanar with the board
  top, same thickness/contact parameters as the board, surrounding the unchanged board so the desk is larger without
  enlarging the 'table' collision box;
- the Franka robot (model/franka_robot.xml, body base_link with its main_cam) at ONE fixed station per task (same for every seed);
  the station puts the robot z axis along the board normal (the source gravity is tilted with the board, so the desk is level
  for the robot) and the board top 0.81 m above the robot floor origin (desk height, user-given);
- presentation-only camera 'global_franka' (not a policy camera), offscreen buffer 1920x1080 (visual only).
Usage: build_world.py TASK OUT_DIR
'''
import copy, hashlib, json, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from rsi.lift2_suite.common import specs  # noqa: E402  read-only use of the source world registry

ROBOT_XML = ROOT / 'rsi/franka_suite/model/franka_robot.xml'
DESK_TOP = .81
DESK = {'back_x': .32, 'depth': .80, 'half_width': .70}  # desk in the robot frame: 0.80 x 1.40 m from the ego-video fit (ESTIMATE), near edge moved from 0.44 to 0.32 m so the board (task table) fits on it
SRC_BASES = ('left_base', 'right_base')
BENCH_PREFIX = ('optical_bench_top', 'bench_slot_', 'bench_leg_', 'bench_isolator_')
BORDER_RGBA = '.10 .09 .08 1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _fmt(v):
    return ' '.join('%.10g' % x for x in np.ravel(v))


def table_frame(record):
    t = record['table']
    return np.array(t['rotation']), np.array(t['center']), np.array(t['half_size'])


def station_transform(record, board_xy, align='world_x'):
    '''Robot base_link pose in the task frame. board_xy = board top centre in the robot frame (x forward, y left).
    align='world_x' (v1 worlds): robot x = task-frame +x projected on the board plane (the board itself is yawed ~93 deg, so its edges are
    3.3-3.7 deg off the robot axes). align='board_edge' (v2_parallel): robot x = the board edge axis (board-frame
    +-x/+-y column, projected on the board plane) closest to the v1 direction, i.e. the robot is turned about the board-centre normal until its
    base is parallel to the board edges; the board centre stays 0.70 m ahead along the new robot x.'''
    Rt, ctop, _ = table_frame(record)
    z = Rt[:, 2] / np.linalg.norm(Rt[:, 2])
    x = np.array([1., 0, 0]) - z[0] * z; x /= np.linalg.norm(x)
    if align == 'board_edge':
        cands = [sgn * Rt[:, k] for k in (0, 1) for sgn in (1., -1.)]
        c = max(cands, key=lambda v: float(v @ x)); c = c - (c @ z) * z; x = c / np.linalg.norm(c)
    elif align != 'world_x':
        raise ValueError(align)
    y = np.cross(z, x)
    Rr = np.column_stack([x, y, z])
    T = np.eye(4); T[:3, :3] = Rr; T[:3, 3] = ctop - Rr @ np.array([board_xy[0], board_xy[1], DESK_TOP])
    return T


def desk_borders(record, T, desk):
    '''Four border boxes in the TABLE frame around the unchanged board, covering the desk rectangle given in the robot frame.'''
    Rt, ctop, half = table_frame(record)
    corners = [T[:3, :3] @ [x, y, DESK_TOP] + T[:3, 3] for x in (desk['back_x'], desk['back_x'] + desk['depth']) for y in (-desk['half_width'], desk['half_width'])]
    loc = np.array([Rt.T @ (c - ctop) for c in corners])[:, :2]
    lo, hi = loc.min(0), loc.max(0)
    if np.any(lo > -half[:2]) or np.any(hi < half[:2]):
        raise ValueError(f'desk {lo}..{hi} does not contain the board {half[:2]}')
    th = half[2]
    boxes = {  # (centre u, v), half sizes (u, v) in the table frame
        'desk_border_umin': ((lo[0] + -half[0]) / 2, (lo[1] + hi[1]) / 2, (-half[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2),
        'desk_border_umax': ((half[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (hi[0] - half[0]) / 2, (hi[1] - lo[1]) / 2),
        'desk_border_vmin': (0., (lo[1] - half[1]) / 2, half[0], (-half[1] - lo[1]) / 2),
        'desk_border_vmax': (0., (hi[1] + half[1]) / 2, half[0], (hi[1] - half[1]) / 2),
    }
    out = []
    for n, (u, v, hu, hv) in boxes.items():
        if hu <= 1e-4 or hv <= 1e-4:
            continue
        out.append((n, ctop + Rt @ [u, v, -th], np.r_[hu, hv, th], Rt))
    return out, (lo.tolist(), hi.tolist())


def build(task_id, out_dir, board_xy, desk=None, align='world_x'):
    desk = dict(desk or DESK)
    spec = specs.runnable(task_id); src_dir = Path(spec['world']); src_scene = src_dir / 'scene.xml'
    record = json.loads((src_dir / 'reconstruction.json').read_text())
    tree = ET.parse(src_scene); root = tree.getroot(); wb = root.find('worldbody')
    root.set('model', f'{task_id}_franka')
    removed, names, gone_meshes = [], set(), set()
    for el in list(wb):
        nm = el.get('name') or ''
        if (el.tag == 'body' and nm in SRC_BASES) or (el.tag == 'camera' and nm == 'main') or (el.tag == 'geom' and nm.startswith(BENCH_PREFIX)):
            for e in el.iter():
                if e.get('name'): names.add(e.get('name'))
                if e.tag == 'geom' and e.get('mesh'): gone_meshes.add(e.get('mesh'))
            wb.remove(el); removed.append([el.tag, nm])
    for tag in ('actuator', 'equality', 'contact', 'sensor', 'tendon', 'custom', 'keyframe'):
        for c in root.findall(tag):
            for e in list(c):
                toks = {t for v in e.attrib.values() for t in [v, *v.split()]}
                if tag in ('keyframe', 'actuator') or toks & names or (tag == 'custom' and (e.get('name') or '').startswith(('gripper', 'left_', 'right_'))):
                    c.remove(e); removed.append([tag, e.get('name', '')])
            if len(c) == 0: root.remove(c)
    comp = root.find('compiler'); asset = root.find('asset')
    mdir = comp.get('meshdir', '') if comp is not None else ''; tdir = comp.get('texturedir', '') if comp is not None else ''
    kept_mesh = {g.get('mesh') for g in root.iter('geom') if g.get('mesh')}
    for e in list(asset):
        if e.tag == 'mesh' and e.get('name') in gone_meshes and e.get('name') not in kept_mesh:
            asset.remove(e); continue
        if e.get('file'):
            sub = mdir if e.tag == 'mesh' else tdir if e.tag == 'texture' else ''
            e.set('file', str((src_dir / sub / e.get('file')).resolve()))
    if comp is not None:
        for k in ('meshdir', 'texturedir'): comp.attrib.pop(k, None)
    vis = root.find('visual')
    if vis is None: vis = ET.SubElement(root, 'visual')
    g = vis.find('global')
    if g is None: g = ET.SubElement(vis, 'global')
    g.set('offwidth', '1920'); g.set('offheight', '1080')
    # desk borders (fix B): copy the board's contact parameters
    table = next(e for e in wb if e.tag == 'geom' and e.get('name') == 'table')
    T = station_transform(record, board_xy, align)
    T1 = station_transform(record, board_xy, 'world_x')
    yaw_vs_v1 = float(np.degrees(np.arctan2(np.cross(T1[:3, 0], T[:3, 0]) @ T[:3, 2], T1[:3, 0] @ T[:3, 0])))
    borders, desk_tbl = desk_borders(record, T, desk)
    idx = list(wb).index(table)
    for k, (n, c, hs, Rt) in enumerate(borders):
        b = ET.Element('geom', {a: v for a, v in table.attrib.items() if a in ('type', 'friction', 'solref', 'solimp', 'condim', 'contype', 'conaffinity', 'margin', 'gap', 'priority')})
        b.set('name', n); b.set('pos', _fmt(c)); b.set('quat', _fmt(np.roll(Rotation.from_matrix(Rt).as_quat(), 1))); b.set('size', _fmt(hs)); b.set('rgba', BORDER_RGBA)
        wb.insert(idx + 1 + k, b)
    # robot
    rob = ET.parse(ROBOT_XML).getroot()
    for mesh in rob.find('asset').findall('mesh'):
        mesh = copy.deepcopy(mesh); mesh.set('file', str((ROBOT_XML.parent / 'meshes' / mesh.get('file')).resolve())); asset.append(mesh)
    base = copy.deepcopy(rob.find('worldbody').find("body[@name='base_link']"))
    base.set('pos', _fmt(T[:3, 3])); base.set('quat', _fmt(np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)))
    for j in base.iter('joint'):
        if '_arm_joint' in j.get('name') or 'finger_joint1' in j.get('name'): j.set('actuatorgravcomp', 'true')
    wb.append(base)
    gpos = T[:3, :3] @ [-.35, 0., 1.95] + T[:3, 3]; gtar = T[:3, :3] @ [board_xy[0] - .05, board_xy[1], DESK_TOP + .05] + T[:3, 3]
    f = gtar - gpos; f /= np.linalg.norm(f); xa = np.cross(f, T[:3, 2]); xa /= np.linalg.norm(xa); ya = np.cross(xa, f)
    ET.SubElement(wb, 'camera', name='global_franka', pos=_fmt(gpos), xyaxes=_fmt(np.r_[xa, ya]), fovy='58')
    for tag in ('actuator', 'equality', 'contact'):
        dst = root.find(tag)
        if dst is None: dst = ET.SubElement(root, tag)
        for e in rob.find(tag): dst.append(copy.deepcopy(e))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=False)
    ET.indent(tree, space='  ')
    scene = out_dir / 'scene.xml'; tree.write(scene, encoding='utf-8')
    mech = verify_mechanics(src_scene, scene, names)
    info = {'task': task_id, 'robot': 'franka1_tilted_umi', 'scene': str(scene), 'scene_sha256': sha(scene), 'source_scene': str(src_scene), 'source_scene_sha256': sha(src_scene),
        'robot_xml_sha256': sha(ROBOT_XML), 'board_xy_robot_frame': list(board_xy), 'desk_robot_frame': desk, 'desk_table_frame_extent': desk_tbl, 'desk_top_m': DESK_TOP,
        'station_T_task_base': T.tolist(), 'station_align': align, 'station_yaw_vs_world_x_station_deg': yaw_vs_v1, 'removed': removed, 'borders': [b[0] for b in borders], 'mechanics': mech}
    (out_dir / 'build.json').write_text(json.dumps(info, indent=1))
    if not mech['ok']:
        raise RuntimeError(f'{task_id}: non-robot mechanics differ from source: {mech["diffs"][:10]}')
    return info


def verify_mechanics(src_scene, scene, removed_names):
    '''Every kept non-robot body/joint/geom and the options must equal the source compiled model.'''
    import mujoco
    a = mujoco.MjModel.from_xml_path(str(src_scene)); b = mujoco.MjModel.from_xml_path(str(scene)); diffs = []
    def body_ok(m, i): return m.body(i).name.startswith('object_')
    for i in range(1, a.nbody):
        n = a.body(i).name
        if not n.startswith('object_'): continue
        j = b.body(n).id
        for f in ('mass', 'inertia', 'pos', 'quat', 'ipos', 'iquat'):
            if not np.allclose(getattr(a.body(i), f), getattr(b.body(j), f), atol=1e-9): diffs.append(f'body {n}.{f}')
        if a.body(a.body_parentid[i]).name != b.body(b.body_parentid[j]).name: diffs.append(f'body {n}.parent')
    for i in range(a.njnt):
        n = a.joint(i).name
        if not a.body(a.jnt_bodyid[i]).name.startswith('object_'): continue
        j = b.joint(n).id
        for f in ('jnt_type', 'jnt_range', 'jnt_axis', 'jnt_solref', 'jnt_solimp', 'jnt_stiffness', 'jnt_pos'):
            if not np.allclose(getattr(a, f)[i], getattr(b, f)[j]): diffs.append(f'joint {n}.{f}')
        da, db = a.jnt_dofadr[i], b.jnt_dofadr[j]
        for f in ('dof_damping', 'dof_frictionloss', 'dof_armature'):
            if not np.allclose(getattr(a, f)[da], getattr(b, f)[db]): diffs.append(f'dof {n}.{f}')
    for i in range(a.ngeom):
        n = a.geom(i).name; bn = a.body(a.geom_bodyid[i]).name
        if bn.startswith('object_'):
            cand = [g for g in range(b.ngeom) if b.body(b.geom_bodyid[g]).name == bn and b.geom(g).name == n]
            k = [g for g in range(a.ngeom) if a.geom_bodyid[g] == a.geom_bodyid[i] and a.geom(g).name == n].index(i)
            if len(cand) <= k: diffs.append(f'geom {bn}/{n} missing'); continue
            j = cand[k]
        elif n == 'table':
            j = b.geom('table').id
        else:
            continue
        for f in ('type', 'size', 'pos', 'quat', 'friction', 'solref', 'solimp', 'contype', 'conaffinity', 'condim', 'margin', 'gap', 'dataid'):
            va, vb = getattr(a.geom(i), f), getattr(b.geom(j), f)
            if f == 'dataid':
                va, vb = int(np.ravel(va)[0]), int(np.ravel(vb)[0])
                if (a.mesh(va).name if va >= 0 else None) != (b.mesh(vb).name if vb >= 0 else None): diffs.append(f'geom {bn}/{n}.mesh')
            elif not np.allclose(va, vb, atol=1e-9): diffs.append(f'geom {bn}/{n}.{f}')
    for f in ('timestep', 'gravity', 'impratio', 'tolerance', 'iterations', 'cone', 'integrator', 'noslip_iterations'):
        if not np.allclose(getattr(a.opt, f), getattr(b.opt, f)): diffs.append(f'opt.{f}')
    return {'ok': not diffs, 'n_diffs': len(diffs), 'diffs': diffs, 'n_object_bodies': int(sum(a.body(i).name.startswith('object_') for i in range(a.nbody)))}


if __name__ == '__main__':
    t, out = sys.argv[1], sys.argv[2]
    bx, by = (float(v) for v in sys.argv[3].split(',')) if len(sys.argv) > 3 else (.68, 0.)
    info = build(t, out, (bx, by))
    print(json.dumps({k: info[k] for k in ('task', 'scene_sha256', 'board_xy_robot_frame', 'borders')}), info['mechanics']['ok'])
