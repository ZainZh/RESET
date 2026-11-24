"""Franka robot configuration (robot-only; no task semantics). Declared values, NOT hardware-identified.

Model: model/franka_robot.xml (built by model/build_robot.py from the user's franka1_tilted_mounting URDF: two FR3 arms on the
tilted mount, Franka hands with UMI fingers, ONE ZED 2i camera main_cam). Root body base_link (= URDF world origin, floor under
the robot). TCP <arm>_tcp = UMI fingertip-pad centre, +z approach, +y closing (source canonical axes).
"""
from pathlib import Path
import numpy as np

MODEL = Path(__file__).resolve().parents[1] / 'model'
MJCF = MODEL / 'franka_robot.xml'
BASE_BODY = 'base_link'
NJ = 7
ARMS = {
    'left': {'joints': ['left_arm_joint%d' % i for i in range(1, 8)], 'gripper': 'left_arm_finger_joint1', 'followers': ('left_arm_finger_joint2',)},
    'right': {'joints': ['right_arm_joint%d' % i for i in range(1, 8)], 'gripper': 'right_arm_finger_joint1', 'followers': ('right_arm_finger_joint2',)},
}
GRIPPER_OPEN_M = .038  # normalized opening o in [0,1] -> each finger target GRIPPER_CLOSED_M + o * (GRIPPER_OPEN_M - GRIPPER_CLOSED_M)
GRIPPER_CLOSED_M = .001  # pre-baseline interface fix (same reason as LIFT2 GRIPPER_OPEN_M): servoing the finger exactly AT its 0/0.04 m joint limit
# let a table/object contact push the soft-mimic follower 0.2 mm past the limit (finger_limit guard) in the first source runs; 1-2 mm stay inside.
OBS_CAMERA = 'main_cam'
POLICY_CAMERAS = ('main_cam',)
VIDEO_CAMERAS = ('main_cam', 'global_franka')
# Start/ready pose: ONE code constant for every task and seed. Solved once by tools/solve_home.py:
# IK for parking points (left TCP (0.5, 0.5, 1.0), right TCP (0.5, -0.3, 1.2) in the robot frame, approach down), no robot contact with a desk box,
# best straight-line (io.move preflight) reach to 48 top-down work targets: left 33/48, right 40/48. Arms are outside the main_cam view of the board.
HOME = ((0.17612, -1.58404, 0.21825, -2.91385, 1.11062, 2.5924, -0.8538),
        (1.76858, -0.98136, 0.0829, -2.80533, -1.0467, 0.78764, 0.49363))
# Per-task start pose (code constant, the same for every seed). sweep_objects: the source task starts with the left gripper around the dustpan
# handle (source canonical home TCP, io.demo home['left']); left arm = Franka IK to that same task-frame TCP pose (tools/solve_contact_home.py,
# 345 seeds with zero robot contact over all 20 DEV resets, largest limit margin);
# right arm = HOME right.
HOME_TASK = {'sweep_objects': ((1.39364, -0.66408, -1.72293, -2.10448, -1.91168, 3.74897, 0.92735), HOME[1])}


# worlds_v2_parallel: the station is turned ~3.3 deg about the board normal, so the same task-frame contact-home TCP needs new joint angles
# (tools/solve_contact_home.py with FRANKA_WORLD_SET=worlds_v2_parallel,
# 354 seeds with zero robot contact over all 20 DEV resets, largest limit margin). Other tasks use HOME (robot-frame constant).
HOME_TASK_V2 = {'sweep_objects': ((1.41494, -0.64297, -1.73859, -2.06057, -1.93735, 3.77357, 1.04146), HOME[1])}


def home_for(task, world_set='worlds'):
    if world_set == 'worlds_v2_parallel':
        return HOME_TASK_V2.get(task, HOME)
    return HOME_TASK.get(task, HOME)


PLANNING_MJCF = MODEL / 'franka_planning.xml'  # robot-only planning model (robot + 'table' box), robot frame = base_link at the origin
