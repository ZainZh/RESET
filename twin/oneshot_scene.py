"""Compose both official arms using their measured relative base transform."""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .robot import ROOT,rotate_gripper_mount

def fmt(a):return ' '.join(f'{x:.10g}' for x in np.asarray(a).ravel())
def quat(matrix):
    q=Rotation.from_matrix(matrix).as_quat();return fmt(np.r_[q[3],q[:3]])

def robot_scene(calibration,intrinsic,mounts=None):
    mounts=mounts or {'left':90,'right':90}
    root=ET.Element('mujoco',model='xtrainer_calibrated_bimanual')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true',meshdir=str(ROOT/'robots/xtrainer'))
    option=ET.SubElement(root,'option',timestep='.002',integrator='implicitfast',ccd_tolerance='1e-9',ccd_iterations='100')
    ET.SubElement(option,'flag',nativeccd='enable')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='1280',offheight='720')
    ET.SubElement(visual,'headlight',ambient='.3 .3 .3',diffuse='.5 .5 .5',specular='.1 .1 .1');ET.SubElement(visual,'map',znear='.01')
    world=ET.SubElement(root,'worldbody');assets=None;actuators=ET.SubElement(root,'actuator')
    for arm in ('left','right'):
        robot=ET.parse(ROOT/'robots/xtrainer/left_arm_cad.xml').getroot();rotate_gripper_mount(robot,float(mounts[arm]))
        if assets is None:assets=copy.deepcopy(robot.find('asset'));root.insert(2,assets)
        body=robot.find('worldbody/body')
        for element in body.iter():
            for key in ('name','joint'):
                name=element.get(key)
                if name is not None:
                    if name.startswith('left_'):name=arm+name[4:]
                    elif name=='tcp':name=arm+'_tcp'
                    else:name=arm+'_'+name
                    element.set(key,name)
        if arm=='right':
            T=calibration['world_from_right'];body.set('pos',fmt(T[:3,3]));body.set('quat',quat(T[:3,:3]))
        world.append(body)
        for actuator in robot.find('actuator'):
            for key in ('name','joint'):
                name=actuator.get(key)
                if name.startswith('left_'):name=arm+name[4:]
                else:name=arm+'_'+name
                actuator.set(key,name)
            actuators.append(actuator)
    custom=ET.SubElement(root,'custom')
    travel=float(ET.parse(ROOT/'robots/xtrainer/left_arm_cad.xml').find('./custom/numeric[@name="gripper_closing_travel"]').get('data'))
    ET.SubElement(custom,'numeric',name='gripper_closing_travel',data=str(travel))
    ET.SubElement(custom,'text',name='gripper_variant',data='official_CAD')
    ET.SubElement(world,'light',pos='0 -.5 1.8',dir='0 0 -1',diffuse='.3 .3 .3',castshadow='true')
    T=calibration['world_from_camera'];K=intrinsic;w,h=640,360
    ET.SubElement(world,'camera',name='main',pos=fmt(T[:3,3]),quat=quat(T[:3,:3]@np.diag([1,-1,-1])),resolution=f'{w} {h}',sensorsize=fmt([w/K[0,0],h/K[1,1]]),focal='1 1',principal=fmt([(w/2-K[0,2])/K[0,0],(K[1,2]-h/2)/K[1,1]]))
    ET.SubElement(world,'camera',name='overview',pos='-.6 -1.3 1.2',xyaxes='0.85 -0.52 0 0.33 0.54 0.77')
    return root


def set_bimanual(model,data,joints,gripper):
    travel=float(model.numeric('gripper_closing_travel').data[0])
    for index,arm in enumerate(('left','right')):
        for i,value in enumerate(joints[index],1):
            data.qpos[model.joint(f'{arm}_j{i}').qposadr[0]]=value;data.ctrl[model.actuator(f'{arm}_motor{i}').id]=value
        close=travel*(1-np.clip(gripper[index],0,1))
        for i,value in ((1,-close),(2,close)):
            data.qpos[model.joint(f'{arm}_gripper{i}').qposadr[0]]=value;data.ctrl[model.actuator(f'{arm}_gripper_motor{i}').id]=value
