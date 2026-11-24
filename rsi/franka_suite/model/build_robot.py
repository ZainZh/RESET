'''Build the Franka robot-only MJCF (simulation only, NOT hardware-identified).

Source: franka1_tilted (single-camera variant) mujoco_flat/franka1_tilted_mounting_flat.urdf (decimated base_link.STL, flat mesh dir),
copied with the STL meshes into model/meshes. MuJoCo compiles the URDF, mj_saveLastXML gives a plain MJCF, and this script edits
that XML only:
- every fused world geom/body is wrapped into ONE root body 'base_link' (robot origin = URDF world origin = floor under the robot);
  the world builder places it with pos/quat (fixed station, same for every seed);
- named geoms for the fused base and torso meshes;
- collision classes (DISCLOSED): proximal geoms (base, torso, link0..link3) contype=4 conaffinity=1 collide with the table/objects
  and with distal geoms, not with each other (inherent root mesh overlap of the tilted mount); distal geoms (link4..7, hand, UMI
  fingers) contype=2 conaffinity=7 collide with everything; contacts between two bodies of the SAME arm (link1..link7, both UMI
  fingers) are excluded (convex hulls of neighbouring FR3 links overlap in normal poses; the real arm has its own self-collision
  limits, not modelled). Cross-arm, robot-table, robot-desk and robot-object contacts stay;
- URDF finger mimic -> joint equality (finger_joint2 = finger_joint1);
- position servos: arm kp 1000 kv 100 (force limits from the URDF), armature 0.1, robot-body gravity compensation (gravcomp=1);
  hand kp 2000 kv 100 on finger_joint1, ctrlrange 0..0.04 m per finger (ASSUMED, not identified);
- TCP sites <arm>_tcp in link7: fingertip end of the UMI fingers (finger body origin + 0.149 m; the mesh spans 0..0.150 m), as the source
  canonical TCP (xTrainer TCP site sits at the fingertip end), axes = hand axes: +z approach, +y closing

  (the source canonical TCP convention, so no axis remapping is needed);
- ONE policy camera main_cam on base_link: ZED 2i 1920x1080, fx=fy=1065.199, at (0.259, 0, 1.447) m, 55 deg below horizontal,
  looking along +x (the pose fitted to an ego-view video of the real setup).
'''
import math, shutil, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MESHES = HERE / 'meshes'
URDF = MESHES / 'franka1_tilted_mounting_flat.urdf'
OUT = HERE / 'franka_robot.xml'
PLAN = HERE / 'franka_planning.xml'  # robot-only planning model for controllers: same robot + a 'table' box (placed by the controller from the fitted plane)
FX = 1065.199
CAM_POS = (0.259, 0.0, 1.447)
CAM_PITCH_DEG = 55.0
TCP_Z = 0.1654 + 0.149  # fingertip end (UMI finger mesh ends at 0.150 m), matching the source canonical TCP (xTrainer site at the fingertip end); v1 used +0.135
KP_ARM, KV_ARM, KP_HAND, KV_HAND = 1000., 100., 2000., 100.


def cam_quat(pitch_deg):
    a = math.radians(pitch_deg)
    v = np.array([math.cos(a), 0., -math.sin(a)]); u = np.array([math.sin(a), 0., math.cos(a)]); r = np.cross(v, u)
    R = np.stack([r, u, -v], axis=1)
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return (w, x, y, z)


def build():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(URDF))
    raw = HERE / '_raw_from_urdf.xml'
    mujoco.mj_saveLastXML(str(raw), m)
    root = ET.parse(raw).getroot(); raw.unlink()
    root.find('compiler').set('meshdir', str(MESHES))
    wb = root.find('worldbody')
    base = ET.Element('body', name='base_link', pos='0 0 0')
    kids = list(wb)
    for c in kids:
        wb.remove(c); base.append(c)
    wb.append(base)
    unnamed = [g for g in base.findall('geom') if not g.get('name')]
    for g, n in zip(unnamed, ('base_link_collision', 'torso_collision')):
        g.set('name', n)
    for g in base.iter('geom'):
        n = g.get('name') or ''
        prox = n in ('base_link_collision', 'torso_collision') or any(f'_arm_link{i}_collision' in n for i in range(4))
        g.set('contype', '4' if prox else '2'); g.set('conaffinity', '1' if prox else '7')
        g.set('rgba', '0.75 0.75 0.77 1' if 'finger' not in n else '0.85 0.80 0.62 1')
    for b in base.iter('body'):
        if b is base: continue
        b.set('gravcomp', '1')
        n = b.get('name')
        for fg in b.findall('geom'):
            if not fg.get('name') and 'finger' in n:
                fg.set('name', n + '_collision'); fg.set('rgba', '0.85 0.80 0.62 1')
        if n.endswith('_arm_link7'):
            arm = n.split('_')[0]
            ET.SubElement(b, 'site', name=f'{arm}_tcp', pos=f'0 0 {TCP_Z}', quat='0.92388 0 0 -0.382683', size='0.004', rgba='1 0 0 0', group='4')
    for j in base.iter('joint'):
        if '_arm_joint' in j.get('name'): j.set('armature', '0.1')
    w, x, y, z = cam_quat(CAM_PITCH_DEG)
    fovy = math.degrees(2 * math.atan(540 / FX))
    ET.SubElement(base, 'camera', name='main_cam', pos=' '.join(map(str, CAM_POS)), quat=f'{w:.9f} {x:.9f} {y:.9f} {z:.9f}', fovy=f'{fovy:.6f}', resolution='1920 1080')
    ex = ET.SubElement(root, 'contact')
    for arm in ('left', 'right'):
        bs = [f'{arm}_arm_link{i}' for i in range(1, 8)] + [f'{arm}_arm_leftfinger', f'{arm}_arm_rightfinger']
        for i, b1 in enumerate(bs):
            for b2 in bs[i + 1:]:
                ET.SubElement(ex, 'exclude', name=f'{b1}__{b2}', body1=b1, body2=b2)
    eq = ET.SubElement(root, 'equality'); act = ET.SubElement(root, 'actuator')
    for arm in ('left', 'right'):
        ET.SubElement(eq, 'joint', name=f'{arm}_finger_mimic', joint1=f'{arm}_arm_finger_joint2', joint2=f'{arm}_arm_finger_joint1', polycoef='0 1 0 0 0', solref='0.005 1')
        for i in range(1, 8):
            jn = f'{arm}_arm_joint{i}'
            je = next(j for j in base.iter('joint') if j.get('name') == jn)
            ET.SubElement(act, 'position', name=f'{arm}_arm_act{i}', joint=jn, kp=str(KP_ARM), kv=str(KV_ARM), ctrlrange=je.get('range'), forcerange=je.get('actuatorfrcrange'))
        ET.SubElement(act, 'position', name=f'{arm}_hand_act', joint=f'{arm}_arm_finger_joint1', kp=str(KP_HAND), kv=str(KV_HAND), ctrlrange='0 0.04', forcerange='-100 100')
    ET.indent(root)
    OUT.write_text(ET.tostring(root, encoding='unicode'))
    wb.insert(0, ET.Element('geom', name='table', type='box', pos='0.7 0 0.79', size='2 2 0.02', contype='1', conaffinity='7', rgba='.3 .3 .3 1'))
    for c in list(base):
        if c.tag == 'camera': base.remove(c)
    PLAN.write_text(ET.tostring(root, encoding='unicode'))
    mm = mujoco.MjModel.from_xml_path(str(OUT))
    print('robot xml', OUT, 'nbody', mm.nbody, 'nu', mm.nu, 'neq', mm.neq, 'ncam', mm.ncam)


if __name__ == '__main__':
    build()
