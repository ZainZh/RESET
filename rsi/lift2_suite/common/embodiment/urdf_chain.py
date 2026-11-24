# Pure-numpy URDF kinematic chain.
# Pure-numpy URDF kinematics (no MuJoCo import). Robot-only declared model; never reads scene or object state. Units: metres, radians.
# v1: true SO(3) log orientation error (correct near 180 deg), analytic geometric Jacobian (world-frame, no factor 2).
import xml.etree.ElementTree as ET
import numpy as np


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _T(xyz, rpy):
    T = np.eye(4); T[:3, :3] = _rpy(*rpy); T[:3, 3] = xyz; return T


def _rot_axis(axis, q):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T = np.eye(4); T[:3, :3] = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K; return T


def so3_log(R):
    # Rotation vector (axis * angle, angle in [0, pi]) of a rotation matrix; valid at 0 and at 180 deg.
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    th = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if th < 1e-8:
        return 0.5 * w
    if np.pi - th < 1e-4:
        A = (R + np.eye(3)) / 2.0; i = int(np.argmax(np.diag(A)))
        v = A[:, i] / np.sqrt(max(A[i, i], 1e-12)); v = v / np.linalg.norm(v)
        if np.dot(v, w) < 0: v = -v
        return th * v
    return th / (2.0 * np.sin(th)) * w


def _rotvec_err(Rg, R):
    # World-frame rotation vector that takes R to Rg.
    return so3_log(Rg @ R.T)


class Chain:
    # Serial chain base -> tip from a URDF; only revolute/continuous joints are actuated.
    def __init__(self, urdf_path, base, tip):
        root = ET.parse(urdf_path).getroot(); self.joints = {}
        for j in root.findall('joint'):
            o = j.find('origin')
            xyz = [float(x) for x in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
            rpy = [float(x) for x in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
            ax = j.find('axis'); lim = j.find('limit')
            self.joints[j.find('child').get('link')] = dict(
                name=j.get('name'), type=j.get('type'), parent=j.find('parent').get('link'), T=_T(xyz, rpy),
                axis=[float(x) for x in ax.get('xyz').split()] if ax is not None else [0, 0, 1],
                lo=float(lim.get('lower', -np.inf)) if lim is not None else -np.inf,
                hi=float(lim.get('upper', np.inf)) if lim is not None else np.inf)
        path, link = [], tip
        while link != base:
            j = self.joints[link]; path.append(j); link = j['parent']
        self.path = path[::-1]
        self.act = [j for j in self.path if j['type'] in ('revolute', 'continuous')]
        self.names = [j['name'] for j in self.act]
        self.lo = np.array([j['lo'] for j in self.act]); self.hi = np.array([j['hi'] for j in self.act])
        self.tool = np.eye(4)

    def set_tool(self, xyz, rpy=(0, 0, 0)):
        self.tool = _T(xyz, rpy)

    def fk(self, q):
        T = np.eye(4); k = 0
        for j in self.path:
            T = T @ j['T']
            if j['type'] in ('revolute', 'continuous'):
                T = T @ _rot_axis(j['axis'], q[k]); k += 1
        return T @ self.tool

    def jacobian(self, q):
        # Analytic geometric Jacobian: rows 0-2 linear velocity of the tool point, rows 3-5 angular velocity, both in the base frame.
        T = np.eye(4); k = 0; axes = []; origins = []
        for j in self.path:
            T = T @ j['T']
            if j['type'] in ('revolute', 'continuous'):
                a = np.asarray(j['axis'], float); a = a / np.linalg.norm(a)
                axes.append(T[:3, :3] @ a); origins.append(T[:3, 3].copy())
                T = T @ _rot_axis(j['axis'], q[k]); k += 1
        p = (T @ self.tool)[:3, 3]; J = np.zeros((6, len(axes)))
        for i, (z, o) in enumerate(zip(axes, origins)):
            J[:3, i] = np.cross(z, p - o); J[3:, i] = z
        return J

    def ik(self, T_goal, q0, w_rot=0.5, iters=300, tol_p=1e-4, tol_r=1e-3, step=0.3):
        # Damped least squares to a full pose. Returns (q, ok, pos_err_m, rot_err_rad).
        q = np.clip(np.array(q0, float), self.lo + 1e-6, self.hi - 1e-6)
        for _ in range(iters):
            T = self.fk(q); dp = T_goal[:3, 3] - T[:3, 3]; dr = _rotvec_err(T_goal[:3, :3], T[:3, :3])
            if np.linalg.norm(dp) < tol_p and np.linalg.norm(dr) < tol_r:
                return q, True, float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
            J = self.jacobian(q); J[3:] *= w_rot; e = np.r_[dp, dr * w_rot]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + dq * min(1.0, step / max(np.abs(dq).max(), 1e-12)), self.lo + 1e-6, self.hi - 1e-6)
        T = self.fk(q)
        return q, False, float(np.linalg.norm(T_goal[:3, 3] - T[:3, 3])), float(np.linalg.norm(_rotvec_err(T_goal[:3, :3], T[:3, :3])))
