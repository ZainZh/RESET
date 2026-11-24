"""Colour-free, tilt-aware perception helpers (new module; rsi/perception.py is untouched).

Findings it addresses (real demo frames vs sim): the real table plane is tilted ~2.5 deg, so a single table height
misreads the +x side of the table as objects; ~14% of real points fall outside the 302x302 raster canvas of
perception.locate and make it raise IndexError. Here points are cropped to the canvas first and heights are measured
from a fitted plane. Uses only the observation (RGB-D, K, camera pose).
"""
import numpy as np
import cv2
from rsi import perception as P

FIT_X = (-.1, .45)
FIT_Y = (-.8, -.25)


def in_canvas(p):
    return (p[:, 0] > -.6) & (p[:, 0] < .6) & (p[:, 1] > -1.2) & (p[:, 1] < 0)


def safe_points(obs, mask=None):
    p, ys, xs = P.points(obs, mask)
    k = in_canvas(p)
    return p[k], ys[k], xs[k]


def fit_plane(obs, iters=4, band=.008):
    """Robust table plane (centre, unit normal with +z) from points near the modal table height."""
    p, _, _ = safe_points(obs)
    z0 = P.table_height(obs)
    w = p[(p[:, 0] > FIT_X[0]) & (p[:, 0] < FIT_X[1]) & (p[:, 1] > FIT_Y[0]) & (p[:, 1] < FIT_Y[1])]
    c = np.array([0., -.5, z0]); n = np.array([0., 0., 1.])
    sel = np.abs(w[:, 2] - z0) < .015
    for _ in range(iters):
        near = w[sel]
        if len(near) < 200:
            break
        c = near.mean(0)
        _, _, vt = np.linalg.svd(near[::5] - c)
        n = vt[-1] * np.sign(vt[-1][2])
        sel = np.abs((w - c) @ n) < band
    return c, n


def height(p, plane):
    c, n = plane
    return (np.asarray(p) - c) @ n


def find_blocks(obs, plane=None, lo=.012, hi=.05, min_n=40, max_extent=.12, exclude=()):
    """Colour-free small-object blobs: points lo..hi above the fitted plane, grouped by XY connectivity.
    exclude: [(xy, radius)] discs to drop (e.g. a located bowl). Blobs wider than max_extent are dropped."""
    plane = plane or fit_plane(obs)
    p, _, _ = safe_points(obs)
    k = (p[:, 0] > -.35) & (p[:, 0] < .45) & (p[:, 1] > -.8) & (p[:, 1] < -.25)
    p = p[k]
    h = height(p, plane)
    p = p[(h > lo) & (h < hi)]
    for xy, r in exclude:
        p = p[np.linalg.norm(p[:, :2] - np.asarray(xy)[:2], axis=1) > r]
    if len(p) < min_n:
        return []
    img = P._raster(p)
    n, lab = cv2.connectedComponents(img)
    cell = np.clip(P._cells(p), 0, np.array(img.shape[::-1]) - 1)
    ids = lab[cell[:, 1], cell[:, 0]]
    out = []
    for i in range(1, n):
        q = p[ids == i]
        if len(q) < min_n:
            continue
        ext = q[:, :2].max(0) - q[:, :2].min(0)
        if ext.max() > max_extent:
            continue
        out.append({'center': np.r_[q[:, :2].mean(0), np.median(q[:, 2])], 'n': int(len(q)), 'extent': ext,
                    'top': float(height(q, plane).max()), 'points': q})
    return sorted(out, key=lambda b: -b['n'])
