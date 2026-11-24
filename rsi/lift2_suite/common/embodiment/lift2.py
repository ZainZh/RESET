"""LIFT2 robot configuration (robot-only; no task semantics). Declared values, NOT hardware-identified.

Model: checked copy models/lift2 (patched.urdf + scene.xml robot subtree, CoACD finger hulls ~0.6 mm proud).
"""
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1] / 'models' / 'lift2'
URDF = MODEL / 'patched.urdf'
MJCF = MODEL / 'scene.xml'
BASE_BODY = 'base_link'
BASE_HEIGHT = .082  # base frame above the platform bottom plane (source mounting)

ARMS = {  # API label -> chain spec
    'left': {'joints': ['left_joint1%d' % i for i in range(1, 7)], 'tip': 'left_link16',
        'tool_xyz': (.15556, .00179, .0001),  # fingertip centre between pads (probe p03), approach +x, closing +y
        'gripper': 'left_joint17', 'followers': ('left_joint18', 'left_gripper_pinion_joint')},
    'right': {'joints': ['right_joint2%d' % i for i in range(1, 7)], 'tip': 'right_link26',
        'tool_xyz': (.15556, .00179, .0001),
        'gripper': 'right_joint27', 'followers': ('right_joint28', 'right_gripper_pinion_joint')},
}
GRIPPER_OPEN_M = .042  # normalized opening o in [0,1] -> master target o * 0.042 m (0 = closed). Stroke is 0.044; 2 mm kept inside the
# upper limit because the soft mimic follower overshoots its limit (>0.2 mm) when the master is servoed AT the limit (pre-baseline probe).
OBS_CAMERA = 'head_d435'  # policy observation camera (rendered RGB + ideal depth), nominal pinhole
VIDEO_CAMERAS = ('head_d435', 'left_d405', 'right_d405')
POLICY_CAMERAS = ('head_d435', 'left_d405', 'right_d405')  # A1 (post-baseline): wrist D405 added, nominal pinhole fovy 58
CAMERA_LINKS = {'left': 'left_d405_link', 'right': 'right_d405_link'}
# Servo model: checked MuJoCo position actuators (kp/kv/force limits from the checked scene) plus gravity
# compensation routed through the joint actuators (MuJoCo gravcomp/actuatorgravcomp, robot bodies only),
# the counterpart of the source runtime's qfrc_bias feed-forward. Declared, not measured.
GRAVITY_COMPENSATION = True
