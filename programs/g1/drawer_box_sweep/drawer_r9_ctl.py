# drawer_r9_ctl (r9 = r8 + opened check from the cabinet mask's front extent (closed face x minus 3rd-percentile x of the re-perceived 'wooden cabinet' mask, >= 4.5 cm) instead of the
# re-detected knob: with the clean park pose (no table penetration) the knob re-detection hit the opened drawer's inner wall (z .072) and read 1.6 cm while the drawer was 12.1 cm open
# (drawer_r8_park2_diag_s0). The returned "knob" for the drop-point logic is the closed knob shifted by the measured opening. Everything else = r8.
import numpy as np
import drawer_r8_ctl as D

_ko = D.knob_from_depth


def open_drawer(io, knob, cab_prompt='wooden cabinet'):
    k0 = knob.copy()
    obs = io.observe()
    cab, plane = D.cabinet(io, obs, cab_prompt)
    _, face0, _ = _ko(io, obs, cab, plane)
    front0 = float(np.percentile(cab['points'][:, 0], 3))
    tried = 0
    for attempt in range(3):
        ranked = []
        for i, (side, nm, R) in enumerate(D.KNOB_SIDE):
            g = knob + R[:, 1] * D.THUMB_OFF
            a = R[:, 2]
            w = D.margin_plan(io, side, [g - a * .07, g, g + [-D.PULL, 0, 0], g + [-D.PULL, 0, .07]], R)
            if w is not None and w >= D.MIN_MARGIN:
                ranked.append((i, w, side, nm, R))
        io.log('knob_ranked', cands=[(x[3], round(float(x[1]), 3)) for x in ranked])
        if len(ranked) <= attempt:
            break
        i, w, side, nm, R = ranked[attempt]
        try:
            D.pull_once(io, knob, nm, R, side)
        except Exception as e:
            if io.error is not None:
                raise
            io.log('pull_failed', cand=nm, err=str(e)[:80])
            try:
                io.grip(side, 1., .3)
                io.home(side, 2.0)
            except Exception:
                if io.error is not None:
                    raise
        tried += 1
        obs = io.observe()
        cab, plane = D.cabinet(io, obs, cab_prompt)
        front = float(np.percentile(cab['points'][:, 0], 3))
        opened = front0 - front
        io.log('drawer_check', cand=nm, front0=front0, front=front, opened_m=opened)
        if opened >= D.OPEN_MIN:
            return k0 + np.array([-opened, 0., 0.])
    raise RuntimeError('drawer not opened after %d pulls' % tried)


D.open_drawer = open_drawer


def run(io):
    io.log('variant', name='drawer_r9')
    D.run(io)
