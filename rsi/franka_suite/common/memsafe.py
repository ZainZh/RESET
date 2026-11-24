'''Process-local, numerically equivalent replacement for rsi.perception_geo.fit_plane (memory fix; the shared file is NOT edited).

The shared fit_plane calls np.linalg.svd(near[::5] - c) with the default full_matrices=True, which allocates an N x N U matrix
(15-18 GB per process at 640x480; more at the Franka 1920x1080 camera). Only vt[-1] is used; full_matrices=False returns the same
3x3 vt. install() rebinds rsi.perception_geo.fit_plane in the CURRENT process only (perception_sam3 calls it through the module
attribute G.fit_plane, so every caller sees the replacement). The body below is the shared function verbatim except for that flag.
The trusted Franka runner calls install() in every episode child before the controller is loaded, for every controller (baseline
and branches alike), and records it in result.json.
'''
import numpy as np


def fit_plane(obs, iters=4, band=.008):
    from rsi import perception_geo as G, perception as P
    p, _, _ = G.safe_points(obs)
    z0 = P.table_height(obs)
    w = p[(p[:, 0] > G.FIT_X[0]) & (p[:, 0] < G.FIT_X[1]) & (p[:, 1] > G.FIT_Y[0]) & (p[:, 1] < G.FIT_Y[1])]
    c = np.array([0., -.5, z0]); n = np.array([0., 0., 1.])
    sel = np.abs(w[:, 2] - z0) < .015
    for _ in range(iters):
        near = w[sel]
        if len(near) < 200:
            break
        c = near.mean(0)
        _, _, vt = np.linalg.svd(near[::5] - c, full_matrices=False)
        n = vt[-1] * np.sign(vt[-1][2])
        sel = np.abs((w - c) @ n) < band
    return c, n


def install():
    from rsi import perception_geo as G
    G._shared_fit_plane = getattr(G, '_shared_fit_plane', G.fit_plane)
    G.fit_plane = fit_plane
    return 'rsi.perception_geo.fit_plane -> rsi.franka_suite.common.memsafe.fit_plane (full_matrices=False)'
