"""Public RGB-D perception for controllers: colour + depth → object point clouds.

Uses only the observation (RGB, depth, K, camera pose). Table plane is fit from
depth. Robot pixels are excluded by a height/colour band and by distance to the
current TCPs. Returns per-object centroid, top height, xy extent and yaw.
"""
import numpy as np
import cv2

HSV = {  # OpenCV hue 0-180
    'red': [((0, 120, 60), (8, 255, 255)), ((170, 120, 60), (180, 255, 255))],
    'orange': [((5, 150, 120), (18, 255, 255))],
    'yellow': [((20, 120, 120), (35, 255, 255))],
    'green': [((40, 80, 40), (85, 255, 255))],
    'blue': [((90, 120, 60), (125, 255, 255))],
    'purple': [((125, 60, 30), (165, 255, 255))],
    'beige': [((8, 20, 120), (30, 110, 255))],
    'wood': [((8, 50, 90), (25, 200, 255))],
    'grey_blue': [((95, 15, 60), (130, 120, 220))],
    'carton': [((10, 100, 120), (35, 255, 255)), ((40, 80, 60), (85, 255, 255))],
}


def points(obs, mask=None):
    d = obs['depth']; K = obs['K']; T = obs['T_world_from_camera']
    ys, xs = np.where((d > .1) & (d < 2.) if mask is None else mask & (d > .1) & (d < 2.))
    z = d[ys, xs]
    c = np.column_stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z])
    return c @ T[:3, :3].T + T[:3, 3], ys, xs


def table_height(obs):
    p, _, _ = points(obs)
    p = p[(p[:, 1] < -.25) & (p[:, 1] > -.8) & (np.abs(p[:, 0]) < .35)]
    h = np.histogram(p[:, 2], bins=np.arange(-.05, .3, .002))
    return float(h[1][np.argmax(h[0])] + .001)


def locate(obs, color, tcps=(), min_pixels=25, above=.004, max_height=.35, exclude=None):
    hsv = cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], bool)
    for lo, hi in HSV[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi)) > 0
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    p, ys, xs = points(obs, mask)
    if len(p) < min_pixels:
        return None
    z0 = table_height(obs)
    keep = (p[:, 2] > z0 + above) & (p[:, 2] < z0 + max_height)
    for t in tcps:  # drop gripper pixels: within a vertical cylinder around each TCP, above its tip
        t = np.asarray(t, float)
        keep &= ~((np.linalg.norm(p[:, :2] - t[:2], axis=1) < .09) & (p[:, 2] > t[2] - .02))
    if exclude is not None:  # drop points inside a known container footprint
        keep &= np.linalg.norm(p[:, :2] - np.asarray(exclude[0])[:2], axis=1) > exclude[1]
    p = p[keep]
    if len(p) < min_pixels:
        return None
    # Largest XY cluster.
    n, lab = cv2.connectedComponents(_raster(p))
    cell = _cells(p)
    if n > 2:
        ids = lab[cell[:, 1], cell[:, 0]]; best = np.bincount(ids[ids > 0]).argmax(); p = p[ids == best]
    xy = p[:, :2]; c = xy.mean(0)
    w, v = np.linalg.eigh(np.cov((xy - c).T)) if len(xy) > 3 else (np.ones(2), np.eye(2))
    yaw = float(np.arctan2(v[1, -1], v[0, -1]))
    return {'center': np.r_[c, np.median(p[:, 2])], 'top': float(np.percentile(p[:, 2], 95)),
            'min': p.min(0), 'max': p.max(0), 'yaw': yaw, 'n': int(len(p)), 'table': z0, 'points': p}


GRID = .004


def _cells(p):
    return np.floor((p[:, :2] - np.array([-.6, -1.2])) / GRID).astype(int)


def _raster(p):
    c = _cells(p); img = np.zeros((int(1.2 / GRID) + 2, int(1.2 / GRID) + 2), np.uint8)
    c = np.clip(c, 0, np.array(img.shape[::-1]) - 1)
    img[c[:, 1], c[:, 0]] = 1
    return cv2.dilate(img, np.ones((3, 3), np.uint8))


def fit_sphere(pts, radius):
    """Centre of a sphere of known radius from visible surface points."""
    from scipy.optimize import least_squares
    c0 = pts.mean(0) - [0, 0, radius * .5]
    r = least_squares(lambda c: np.linalg.norm(pts - c, axis=1) - radius, c0, loss='soft_l1', f_scale=.003)
    return r.x


def fit_upright_cylinder(pts, radius=None):
    """Axis xy of an upright round object (bowl/cup) from visible rim/wall points."""
    from scipy.optimize import least_squares
    xy = pts[:, :2]; c0 = xy.mean(0)
    if radius is None:
        r = least_squares(lambda v: np.linalg.norm(xy - v[:2], axis=1) - v[2], np.r_[c0, .05], loss='soft_l1', f_scale=.003)
        return r.x[:2], r.x[2]
    r = least_squares(lambda c: np.linalg.norm(xy - c, axis=1) - radius, c0, loss='soft_l1', f_scale=.003)
    return r.x, radius
