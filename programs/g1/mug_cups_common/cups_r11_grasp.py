# cups_r11_grasp: trusted-side, in-process installation of the 'mug_wide' grasp shape (same calibration as mug_r1_grasp: calib_wide_r1.json) for stack_cups (contact prior, disclosed). No file on disk is edited.
import mug_r1_grasp as _M


def install(hand_arm_module, ra_module, task):
    info = _M.install(hand_arm_module, ra_module, task)
    info['task'] = task
    return info
