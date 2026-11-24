"""Encoder-only Franka (dual FR3, tilted mount, UMI fingers) kinematics in the TASK frame.

Chains are extracted ONCE from the robot-only MJCF (model/franka_robot.xml) as plain numbers (fixed transforms, joint anchors/axes,
limits, TCP site); the station transform T_task_base places the robot root body 'base_link' in the task frame. No simulator objects
are kept. TCP = UMI fingertip-pad centre, axes: +z approach, +y closing (== source canonical TCP axes, so no remapping).
IK = the source twin.practice.motion.RobotKinematics.ik law unchanged except for the joint count (7 instead of 6):
same tolerances (1.5 mm / 0.015 rad), rotation weight 0.15, damping 0.002, step clamp 0.15 rad, clipping inside the limits.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from twin.practice.motion import KinematicsError, numeric, rotation

ARMS = ('left', 'right')
NJ = 7
ROBOT_XML = Path(__file__).resolve().parents[1] / 'model' / 'franka_robot.xml'


def _T(pos, quat):
    T = np.eye(4); T[:3, :3] = Rotation.from_quat(np.asarray(quat, float)[[1, 2, 3, 0]]).as_matrix(); T[:3, 3] = pos; return T


class FrankaKinematics:
    def __init__(self, chains, tips, limits, T_task_base):
        self.chains, self.tips = chains, tips
        self.limits = numeric(limits, (2, NJ, 2), 'joint limits')
        self.T_task_base = numeric(T_task_base, (4, 4), 'station transform')
        if np.any(self.limits[:, :, 0] >= self.limits[:, :, 1]):
            raise KinematicsError('empty joint limits')

    @classmethod
    def from_robot_xml(cls, T_task_base, xml=ROBOT_XML):
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(xml))
        base = m.body('base_link').id
        chains, tips, limits = {}, {}, []
        for arm in ARMS:
            s = m.site(f'{arm}_tcp').id; b = int(m.site_bodyid[s]); anc = []
            while b != base:
                if b == 0: raise KinematicsError('TCP not under base_link')
                anc.append(b); b = int(m.body_parentid[b])
            chain, lim, k = [], [], 0
            for b in reversed(anc):
                fixed = _T(m.body_pos[b], m.body_quat[b])
                if int(m.body_jntnum[b]) != 1: raise KinematicsError('expected one joint per arm link')
                j = int(m.body_jntadr[b])
                if m.joint(j).name != f'{arm}_arm_joint{k + 1}' or m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE: raise KinematicsError('unexpected joint')
                chain.append((fixed, m.jnt_pos[j].copy(), m.jnt_axis[j].copy())); lim.append(m.jnt_range[j].copy()); k += 1
            if k != NJ: raise KinematicsError('expected 7 joints')
            chains[arm] = tuple(chain); tips[arm] = _T(m.site_pos[s], m.site_quat[s]); limits.append(lim)
        return cls(chains, tips, limits, T_task_base)

    def _forward(self, arm, encoders):
        if arm not in self.chains: raise ValueError('arm must be left or right')
        q = numeric(encoders, (NJ,), 'encoders')
        T = self.T_task_base.copy(); axes, origins = [], []
        for k, (fixed, anchor, axis) in enumerate(self.chains[arm]):
            T = T @ fixed
            origins.append(T[:3, :3] @ anchor + T[:3, 3]); axes.append(T[:3, :3] @ axis)
            R = Rotation.from_rotvec(axis * q[k]).as_matrix(); U = np.eye(4); U[:3, :3] = R; U[:3, 3] = anchor - R @ anchor
            T = T @ U
        return T @ self.tips[arm], np.asarray(axes), np.asarray(origins)

    def fk(self, arm, encoders):
        return self._forward(arm, encoders)[0]

    def jacobian(self, arm, encoders):
        T, axes, origins = self._forward(arm, encoders)
        return np.vstack((np.cross(axes, T[:3, 3] - origins).T, axes.T))

    def ik(self, arm, position, orientation, seed, max_iterations=120):
        p, R = numeric(position, (3,), 'target position'), rotation(orientation)
        a = ARMS.index(arm); low, high = self.limits[a, :, 0], self.limits[a, :, 1]
        q = numeric(seed, (NJ,), 'IK seed')
        if np.any(q < low - .05) or np.any(q > high + .05):
            raise KinematicsError('encoders outside joint limits')
        q = np.clip(q, low + 1e-6, high - 1e-6)
        for _ in range(max_iterations):
            T = self.fk(arm, q); dp = p - T[:3, 3]; dr = Rotation.from_matrix(R @ T[:3, :3].T).as_rotvec()
            if np.linalg.norm(dp) <= .0015 and np.linalg.norm(dr) <= .015:
                return q
            J = self.jacobian(arm, q); J[3:] *= .15
            dq = J.T @ np.linalg.solve(J @ J.T + .002 ** 2 * np.eye(6), np.r_[dp, dr * .15])
            dq *= min(1., .15 / max(np.max(np.abs(dq)), 1e-12))
            q = np.clip(q + dq, low + 1e-6, high - 1e-6)
        raise KinematicsError(f'IK did not converge: position={np.linalg.norm(dp):.5f}m rotation={np.linalg.norm(dr):.5f}rad')
