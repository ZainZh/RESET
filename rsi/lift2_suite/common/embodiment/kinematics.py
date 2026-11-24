"""Encoder-only target-robot kinematics in the TASK frame (= source xTrainer world frame).

Chains come from the robot's URDF (pure numpy, copied checked Chain); the fixed mount
transform T_task_base places the robot base in the task frame. IK reuses the source
twin.practice.motion.RobotKinematics.ik algorithm unchanged (same tolerances, weights,
step clamp), so io.move keeps the source success/failure semantics. No simulator access.
"""
import numpy as np
from twin.practice.motion import RobotKinematics, KinematicsError, numeric
from .urdf_chain import Chain, _T

ARMS = ('left', 'right')


class TaskKinematics:
    """fk/jacobian/ik/limits with the RobotKinematics interface for API arm labels left/right."""

    def __init__(self, urdf, T_task_base, arms):
        # arms: {api_label: {'tip': link, 'tool_xyz': (...), 'tool_rpy': (...), 'joints': [...]}}
        self.T_task_base = numeric(T_task_base, (4, 4), 'mount transform')
        self.chains, limits = {}, []
        for arm in ARMS:
            spec = arms[arm]
            ch = Chain(str(urdf), 'base_link', spec['tip'])
            ch.set_tool(spec['tool_xyz'], spec.get('tool_rpy', (0, 0, 0)))
            if list(ch.names) != list(spec['joints']):
                raise KinematicsError(f'URDF chain joints {ch.names} != declared {spec["joints"]}')
            self.chains[arm] = ch
            limits.append(np.stack([ch.lo, ch.hi], 1))
        self.limits = numeric(limits, (2, 6, 2), 'joint limits')

    def set_cameras(self, urdf, links):
        # ROS camera body (x forward, y left, z up) -> OpenCV optical (x right, y down, z forward)
        self.R_body_opt = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        self.cam_chains = {a: Chain(str(urdf), 'base_link', l) for a, l in links.items()}
        return self

    def camera_fk(self, arm, encoders):
        T = self.T_task_base @ self.cam_chains[arm].fk(numeric(encoders, (6,), 'encoders'))
        T[:3, :3] = T[:3, :3] @ self.R_body_opt
        return T

    def fk(self, arm, encoders):
        return self.T_task_base @ self.chains[arm].fk(numeric(encoders, (6,), 'encoders'))

    def jacobian(self, arm, encoders):
        J = self.chains[arm].jacobian(numeric(encoders, (6,), 'encoders'))
        R = self.T_task_base[:3, :3]
        return np.vstack((R @ J[:3], R @ J[3:]))

    def ik(self, arm, position, orientation, seed, max_iterations=120):
        return RobotKinematics.ik(self, arm, position, orientation, seed, max_iterations)


def mount_transform(xyz, yaw_rad):
    return _T(xyz, (0., 0., yaw_rad))


from .frames import canonical_to_physical_R, physical_to_canonical_R  # noqa: E402  the single frame primitive


class CanonicalKinematics:
    '''Policy-facing robot-only kinematics in SOURCE CANONICAL TCP axes (approach +z, closing +y), wrapping a physical
    TaskKinematics (approach +x, closing +y). fk: physical R -> canonical; ik: canonical R -> physical before solving.
    jacobian is the WORLD-frame spatial Jacobian of the unchanged TCP origin, so a fixed tool-axis relabelling does not
    change it (verified by finite differences in tests). limits, camera_fk and T_task_base are physical/unaffected.
    Holds only the encoder-only kinematics: no model, data or object state.'''
    __slots__ = ('_phys',)

    def __init__(self, physical):
        if isinstance(physical, CanonicalKinematics):
            raise TypeError('already canonical: refusing a double conversion')
        self._phys = physical

    @property
    def limits(self):
        return self._phys.limits

    @property
    def T_task_base(self):
        return self._phys.T_task_base

    def camera_fk(self, arm, encoders):
        return self._phys.camera_fk(arm, encoders)

    def fk(self, arm, encoders):
        T = np.array(self._phys.fk(arm, encoders), float)
        T[:3, :3] = physical_to_canonical_R(T[:3, :3])
        return T

    def jacobian(self, arm, encoders):
        return self._phys.jacobian(arm, encoders)

    def ik(self, arm, position, orientation, seed, max_iterations=120):
        return self._phys.ik(arm, position, canonical_to_physical_R(orientation), seed, max_iterations)
