from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
def fk(joints):
    T=np.eye(4)
    tree=ET.parse(ROOT/'robots/xtrainer/nova2_reference.urdf')
    for j,angle in zip((j for j in tree.findall('joint') if j.get('type')=='revolute'),joints):
        origin=j.find('origin');U=np.eye(4);U[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy'),sep=' ')).as_matrix();U[:3,3]=np.fromstring(origin.get('xyz'),sep=' ')
        V=np.eye(4);V[:3,:3]=Rotation.from_rotvec(np.array([0,0,angle])).as_matrix();T=T@U@V
    return T

def set_robot(model,data,joints,gripper):
    for i,value in enumerate(joints,1):
        data.qpos[model.joint(f'left_j{i}').qposadr[0]]=value
        data.ctrl[model.actuator(f'left_motor{i}').id]=value
    # Native command: 1=open, 0=closed; use the exported empty-close stop.
    try:
        travel=float(model.numeric('gripper_closing_travel').data[0])
    except KeyError:
        travel=.04
    closing=travel*(1-np.clip(gripper,0,1))
    for i,value in [(1,-closing),(2,closing)]:
        data.qpos[model.joint(f'left_gripper{i}').qposadr[0]]=value
        data.ctrl[model.actuator(f'gripper_motor{i}').id]=value


def rotate_gripper_mount(robot, yaw_deg):
    """Apply the measured/estimated mounting yaw without changing arm joint frames.

    The imported final-link mesh includes the housing; its geometry and inertia
    rotate together with the child fingers. The flange and TCP frames stay fixed.
    """
    if not yaw_deg:
        return
    rotation=Rotation.from_euler('z',yaw_deg,degrees=True).as_matrix()
    flange=robot.find('.//body[@name="left_link6"]')
    def values(element,key,default):
        return np.fromstring(element.get(key,default),sep=' ')
    def formatted(array):
        return ' '.join(f'{value:.10g}' for value in np.asarray(array).ravel())
    for element in list(flange):
        if element.tag not in ('geom','body','inertial'):
            continue
        element.set('pos',formatted(rotation@values(element,'pos','0 0 0')))
        if element.tag=='inertial' and element.get('fullinertia') is not None:
            xx,yy,zz,xy,xz,yz=values(element,'fullinertia','')
            inertia=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
            inertia=rotation@inertia@rotation.T
            element.set('fullinertia',formatted([inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]))
        else:
            w,x,y,z=values(element,'quat','1 0 0 0')
            q=Rotation.from_matrix(rotation@Rotation.from_quat([x,y,z,w]).as_matrix()).as_quat()
            element.set('quat',formatted([q[3],q[0],q[1],q[2]]))
    custom=robot.find('custom')
    if custom is None:custom=ET.SubElement(robot,'custom')
    ET.SubElement(custom,'numeric',name='gripper_mount_yaw_deg',data=str(yaw_deg))
