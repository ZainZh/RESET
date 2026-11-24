"""Source-demonstration priors decoded with SOURCE xTrainer FK.

Step 1 (unchanged source semantics): rsi.env.demo_priors(task, source_world) where source_world carries only the source
robot's encoder-only RobotKinematics built from the SOURCE scene.xml. Target FK is never applied to source joints.
Source joint angles / home angles are not exported.

Two outputs, selected by FRAME (the single mapping lives in frames.py, imported here, never redefined):
  canonical_priors(task)  RUNTIME/POLICY/DEMO EXPORT. SOURCE CANONICAL TCP axes (approach +z, closing +y), exactly the source
                          controller's semantics; rotations are the source decode, positions/objects0/event.objects preserved.
                          Pair with the public canonical io.tcp/io.move/io._w.kinematics (CanonicalKinematics).
  target_priors(task)     DEPRECATED, TRUSTED DIAGNOSTIC ONLY (scene-audit helpers that use raw physical TaskKinematics). PHYSICAL LIFT
                          TCP axes (approach +x, closing +y) via frames.canonical_to_physical_R. Never feed it to a policy/io.
TCP position is unchanged in both (both TCPs are the fingertip-centre point between the pads).
"""
import copy
import numpy as np
import mujoco
from twin.practice.motion import RobotKinematics
from .frames import (M_CANON_TO_PHYS, M_SOURCE_TO_TARGET, canonical_to_physical_R, physical_to_canonical_R,  # noqa: F401  re-exported conversion helpers
    pose_canonical_to_physical, pose_physical_to_canonical)  # noqa: F401

FRAME_CANONICAL = 'source_canonical: TCP approach +z, closing +y (source xTrainer controller semantics)'
FRAME_PHYSICAL = 'physical_lift: TCP approach +x, closing +y; R_physical = R_canonical @ M (frames.M_CANON_TO_PHYS) [deprecated diagnostic]'
MAPPING = {
    'rotation': 'R_physical = R_canonical @ frames.M_CANON_TO_PHYS; columns (x_p, y_p, z_p) in canonical axes = (+z_c, +y_c, -x_c)',
    'position': 'identity: source TCP site and target TCP both at fingertip centre between pads',
    'objects': 'identity: task frame is the source world frame',
    'arm': 'API labels kept (left->left, right->right); a5 uses the demo grasp arm label',
    'not_exported': 'source joint angles, source home joint angles, source gripper scalar',
}


class _SourceRobot:
    """Only what rsi.env.demo_priors reads: .kinematics (source xTrainer FK)."""

    def __init__(self, source_scene):
        self.kinematics = RobotKinematics.from_model(mujoco.MjModel.from_xml_path(str(source_scene)))


def source_priors(task):
    from rsi.env import demo_priors
    return demo_priors(task, _SourceRobot(task.world / 'scene.xml'))


def canonical_priors(task):
    '''Runtime/policy priors in SOURCE CANONICAL TCP axes: a deep copy of the source decode plus frame metadata.'''
    out = copy.deepcopy(source_priors(task))
    out['frame'] = FRAME_CANONICAL
    return out


def map_priors(src):
    '''DEPRECATED diagnostic: source (canonical) priors -> PHYSICAL TCP axes through the single frames primitive.'''
    out = copy.deepcopy(src)
    for e, s in zip(out['events'], src['events']):
        e['rotation'] = canonical_to_physical_R(s['rotation'])
        e['source_rotation'] = np.asarray(s['rotation'])
    out['home'] = {a: pose_canonical_to_physical(T) for a, T in src['home'].items()}  # source TCP poses (not joint angles)
    out['mapping'] = dict(MAPPING)
    out['frame'] = FRAME_PHYSICAL
    return out


def target_priors(task):
    '''DEPRECATED, TRUSTED DIAGNOSTIC ONLY: physical-axes priors for scene-audit helpers on raw physical TaskKinematics.'''
    return map_priors(source_priors(task))
