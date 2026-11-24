# mug_r1_grasp: trusted-side, in-process installation of the G1 'mug_wide' grasp shape (contact prior, disclosed). Does NOT edit rsi/g1_suite/hand/hand_arm.py or runtime files on disk.
# Shape (calib_wide_r1.json, model-only signed-distance calibration): thumb metacarpal = 1.0 x upper limit, thumb proximal = distal = 0.0 (thumb fully abducted, straight),
# four fingers closure f as in hand_arm; open gap 89.7/90.5 mm (left/right), thumb/finger pad gap 78 mm (mug body diameter at the base) at f 0.206/0.210.
# The pinch frame is the pad midpoint at that contact closure, frame definition identical to TCP_CALIBRATION.json. Grip opening 1 -> f 0 (open), opening 0 -> f F_MAX 0.5 (gap ~42 mm, i.e. squeezing).
# Installing changes the module-level thumb posture for this process only (each DEV episode is its own process and uses only this task's profile).
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = json.load(open(os.path.join(HERE, 'calib_wide_r1.json')))
KIND, KEY, SIZE = 'mug_wide', 'mug_78mm', 0.078
PROFILE = (KIND, SIZE)


def install(hand_arm_module, ra_module, task):
    H = hand_arm_module
    H.KIND_KEY[KIND] = KEY
    H.SIZE_RANGE[KIND] = (0.07, 0.095)
    H.THUMB_FRAC['thumb_metacarpal'] = CALIB['TM']
    H.THUMB_FRAC['thumb_proximal'] = CALIB['TP']
    for side, e in CALIB['sides'].items():
        H.CAL['sides'][side][KEY] = {'finger_closure_f': e['f_contact'], 'pad_gap_mm': e['gap_at_contact_mm'], 'target_mm': 1000 * CALIB['target_m'], 'usable_geometry': True,
                                     'tcp_in_base': e['tcp_in_base'], 'lateral_in_base': e['lateral_in_base'], 'approach_in_base': e['approach_in_base'], 'x_in_base': e['x_in_base'],
                                     'opp_axis_angle_from_base_z_deg': e['opp_axis_angle_from_base_z_deg']}
    ra_module.GRASP_PROFILE[task] = PROFILE
    return {'kind': KIND, 'key': KEY, 'size': SIZE, 'thumb_frac': dict(H.THUMB_FRAC), 'calib': 'calib_wide_r1.json'}
