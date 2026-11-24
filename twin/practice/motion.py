"""Encoder-only robot FK/IK. The extracted chains contain no simulator objects."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation


class KinematicsError(RuntimeError):
    pass


def numeric(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'invalid {name}: expected finite {shape}')
    return result.copy()


def rotation(value):
    result = numeric(value, (3, 3), 'rotation')
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-6) or abs(np.linalg.det(result) - 1) > 1e-6:
        raise ValueError('rotation must be a proper orthonormal matrix')
    return result


def transform(position, quat):
    result = np.eye(4)
    q = numeric(quat, (4,), 'quaternion')
    if abs(np.linalg.norm(q) - 1) > 1e-5:
        raise KinematicsError('invalid robot quaternion')
    result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = numeric(position, (3,), 'position')
    return result


@dataclass(frozen=True)
class Link:
    fixed: np.ndarray
    index: int | None = None
    anchor: np.ndarray | None = None
    axis: np.ndarray | None = None
    reference: float = 0.0


class RobotKinematics:
    """Pure numeric chains in world coordinates, extracted once from robot metadata.

    No MjData, geometry, object transforms or world model is retained. Even the
    Jacobian is computed here from joint axes and current measured encoders.
    """
    def __init__(self, chains, tips, limits):
        self.chains = chains
        self.tips = tips
        self.limits = numeric(limits, (2, 6, 2), 'joint limits')
        if np.any(self.limits[:, :, 0] >= self.limits[:, :, 1]):
            raise KinematicsError('empty joint limits')

    @classmethod
    def from_model(cls, model):
        import mujoco
        chains, tips, limits = {}, {}, []
        for arm in ('left', 'right'):
            site = model.site(f'{arm}_tcp').id
            bid = int(model.site_bodyid[site])
            ancestry = []
            while bid:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
                if not name or not name.startswith(arm + '_'):
                    raise KinematicsError('TCP ancestry must contain robot bodies only')
                ancestry.append(bid)
                bid = int(model.body_parentid[bid])
            chain, arm_limits, found = [], [], []
            for bid in reversed(ancestry):
                fixed = transform(model.body_pos[bid], model.body_quat[bid])
                nj = int(model.body_jntnum[bid])
                if nj > 1:
                    raise KinematicsError('multiple robot joints on a body are unsupported')
                if nj:
                    jid = int(model.body_jntadr[bid])
                    index = len(found)
                    if index >= 6 or model.joint(f'{arm}_j{index + 1}').id != jid:
                        raise KinematicsError('unexpected robot joint ancestry')
                    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[jid]:
                        raise KinematicsError('expected limited hinge')
                    axis = numeric(model.jnt_axis[jid], (3,), 'joint axis')
                    if abs(np.linalg.norm(axis) - 1) > 1e-6:
                        raise KinematicsError('invalid joint axis')
                    chain.append(Link(fixed, index, model.jnt_pos[jid].copy(), axis,
                                      float(model.qpos0[model.jnt_qposadr[jid]])))
                    arm_limits.append(model.jnt_range[jid].copy())
                    found.append(jid)
                else:
                    chain.append(Link(fixed))
            if len(found) != 6:
                raise KinematicsError('expected six joints per arm')
            chains[arm] = tuple(chain)
            tips[arm] = transform(model.site_pos[site], model.site_quat[site])
            limits.append(arm_limits)
        return cls(chains, tips, limits)

    def _forward(self, arm, encoders):
        if arm not in self.chains:
            raise ValueError('arm must be left or right')
        q = numeric(encoders, (6,), 'encoders')
        T, axes, origins = np.eye(4), [], []
        for link in self.chains[arm]:
            T = T @ link.fixed
            if link.index is not None:
                origins.append(T[:3, :3] @ link.anchor + T[:3, 3])
                axes.append(T[:3, :3] @ link.axis)
                R = Rotation.from_rotvec(link.axis * (q[link.index] - link.reference)).as_matrix()
                U = np.eye(4)
                U[:3, :3] = R
                U[:3, 3] = link.anchor - R @ link.anchor
                T = T @ U
        return T @ self.tips[arm], np.asarray(axes), np.asarray(origins)

    def fk(self, arm, encoders):
        return self._forward(arm, encoders)[0]

    def jacobian(self, arm, encoders):
        T, axes, origins = self._forward(arm, encoders)
        return np.vstack((np.cross(axes, T[:3, 3] - origins).T, axes.T))

    def ik(self, arm, position, orientation, seed, max_iterations=120):
        p, R = numeric(position, (3,), 'target position'), rotation(orientation)
        a = ('left', 'right').index(arm)
        low, high = self.limits[a, :, 0], self.limits[a, :, 1]
        q = numeric(seed, (6,), 'IK seed')
        if np.any(q < low - .05) or np.any(q > high + .05):
            raise KinematicsError('encoders outside joint limits')
        q = np.clip(q, low + 1e-6, high - 1e-6)
        for _ in range(max_iterations):
            T = self.fk(arm, q)
            dp = p - T[:3, 3]
            dr = Rotation.from_matrix(R @ T[:3, :3].T).as_rotvec()
            if np.linalg.norm(dp) <= .0015 and np.linalg.norm(dr) <= .015:
                return q
            J = self.jacobian(arm, q)
            # Rotation is weighted in metres to avoid sacrificing translation.
            J[3:] *= .15
            error = np.r_[dp, dr * .15]
            dq = J.T @ np.linalg.solve(J @ J.T + .002**2 * np.eye(6), error)
            dq *= min(1., .15 / max(np.max(np.abs(dq)), 1e-12))
            q = np.clip(q + dq, low + 1e-6, high - 1e-6)
        raise KinematicsError(f'IK did not converge: position={np.linalg.norm(dp):.5f}m rotation={np.linalg.norm(dr):.5f}rad')


def waypoint(start, position, orientation, fraction):
    """Smooth Cartesian translation and shortest-path rotation interpolation."""
    u = float(fraction)
    if not math.isfinite(u) or not 0 <= u <= 1:
        raise ValueError('waypoint fraction outside [0,1]')
    u = u * u * (3 - 2 * u)
    R = rotation(orientation)
    delta = Rotation.from_matrix(R @ start[:3, :3].T).as_rotvec()
    return ((1 - u) * start[:3, 3] + u * numeric(position, (3,), 'position'),
            Rotation.from_rotvec(u * delta).as_matrix() @ start[:3, :3])
