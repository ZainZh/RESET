"""Strict independent lift_box scorer. Controllers never import this.

Goal from the real demonstration (STATUS.md): the juice carton stands upright on top of the wooden box.
Per snapshot (simulator truth only), all must hold:
- on:carton:box     original rsi.scorer relation (supported by the box, above its top, centre over footprint);
- long_axis_vertical:carton carton body y (0.10 m long axis) within 15 deg of world vertical;
- upright:boxbox body z within 15 deg of vertical;
- released   no robot contact with task objects (original rsi.scorer check).
Success = every check in all 11 snapshots AND snapshot times finite, strictly increasing, spanning >= 0.5 s
(dwell_time check) AND drift < 5 mm AND world numerically valid AND no controller
or scoring exception (fail-closed). The original relation result is kept separately as raw_original.
"""
import numpy as np
from rsi import scorer as base

LONG_AXIS_MAX_DEG = 15.
UPRIGHT_MAX_DEG = 15.
DRIFT_M = .005
SNAPSHOTS = 11
DWELL_S = .5
VERSION = 'lift_box_strict_v2'


def tilt_deg(axis_z):
    return float(np.degrees(np.arccos(np.clip(abs(float(axis_z)), 0., 1.))))


def snapshot(task, model, data):
    s = base.snapshot(task, model, data)
    cm = data.body('object_carton').xmat.reshape(3, 3)
    bm = data.body('object_box').xmat.reshape(3, 3)
    long_tilt = tilt_deg(cm[2, 1])
    box_tilt = float(np.degrees(np.arccos(np.clip(bm[2, 2], -1., 1.))))
    s['original_checks'] = dict(s['checks'])
    s['checks']['long_axis_vertical:carton'] = bool(long_tilt <= LONG_AXIS_MAX_DEG)
    s['checks']['upright:box'] = bool(box_tilt <= UPRIGHT_MAX_DEG)
    s['carton_long_axis_tilt_deg'] = long_tilt
    s['box_tilt_deg'] = box_tilt
    s['valid'] = all(s['checks'].values())
    return s


def evaluate(history, numerical_invalid=False, controller_error=None):
    final = history[-1]
    drift = max(float(np.linalg.norm(np.array(final['poses'][n]) - np.array(h['poses'][n])))
        for h in history for n in final['poses'])
    failed = sorted({k for h in history for k, v in h['checks'].items() if not v})
    times = np.array([h.get('time', np.nan) for h in history], float)
    dwell_ok = bool(len(history) == SNAPSHOTS and np.all(np.isfinite(times)) and np.all(np.diff(times) > 0)
        and times[-1] - times[0] >= DWELL_S - 1e-6)
    if not dwell_ok: failed.append('dwell_time')
    if drift >= DRIFT_M: failed.append('dwell_drift')
    if numerical_invalid: failed.append('numerical_invalid')
    if controller_error: failed.append('controller_exception')
    ok = dwell_ok and all(h['valid'] for h in history) and drift < DRIFT_M and not numerical_invalid and not controller_error
    raw = all(all(h['original_checks'].values()) for h in history) and drift < DRIFT_M
    return {'scorer': VERSION, 'success': bool(ok), 'raw_original_success': bool(raw), 'drift_m': drift,
        'dwell_times': times.tolist(),
        'failed_checks': failed, 'carton_long_axis_tilt_deg': final['carton_long_axis_tilt_deg'],
        'box_tilt_deg': final['box_tilt_deg'], 'final': final}


def score_world(task, world, wait):
    """Take the 11-snapshot dwell (wait(.05) between) and evaluate. `wait` steps physics with the robot holding."""
    history = []
    for k in range(SNAPSHOTS):
        if k: wait(.05)
        world.synchronize(); history.append(snapshot(task, world.model, world.data))
    return history
