"""sweep s12 = s11 + high transit (one mechanism). Evidence: s10 and s11 DEV0-19 13/20 each; all 7 failures (DEV1, 8, 10,
13, 14, 16, 17) have the cube 90-180 mm BEYOND the brush tip along the brush axis (grasp_choice cube_along < -0.09), so the brush must turn by
a large angle to point at the pan, and the move to the start hover (only 20 mm above sweep height, the cube is 32 mm tall) sweeps the head
through the cube: brush push 7-29 mm, cube knocked aside / touched by the gripper. Change: every sweep first lifts the held brush to
TRANSIT_Z = 60 mm above sweep height in place, moves/turns to above the start at that height, then descends to the usual 20 mm hover and the
start. TRANSIT_Z = 0 restores s11 exactly. s11 docstring follows.
sweep s11 = s9 + OVERSHOOT .030 -> .000. Evidence: s9 DEV0-19 10/20; every success ends the cube at pan-local y ~-0.063
(needs only >= ~-0.100), while all 10 failures are right_arm_link6 vs left_arm_link5/6 contacts at the END of the first sweep (t 11.3-12.3 s;
5 robot_robot invalid, 5 with the cube slipping off the tip and touched by the gripper first). A shallower end keeps the right wrist
further from the left wrist. s9 docstring follows.
sweep s9 = s4 + END_PUSH (one mechanism: push with the head's END face, brush axis along the push).
Evidence: w2 (head across the push, contact near one end): the long side hits the pan side wall; s3/s4 (long side trailing at 30 deg): the
slanted face drifts the cube 30-40 mm onto the side wall; s1/s5/s6 (moving the push path laterally) and s7/s8 (sliding the held pan) cause
wrist collisions / finger-limit violations. With END_PUSH the head is turned so its axis points back along -d (tip forward, handle behind,
like poking with a stick): nothing sticks out sideways, the 26 mm tip face pushes the cube straight along the push line toward the pan
centre, and the right gripper (95 mm behind the tip, GRIP_FAR) stays outside the pan, away from the left wrist.
Contact = the perceived head tip (contact_s 0, face offset 0 instead of the 13 mm half bristle depth). OVERSHOOT .030 as s4.
END_PUSH = False restores s4 exactly.
s4 docstring follows.
sweep s4 = s3 + OVERSHOOT .010 -> .030 (planned cube centre 40 mm instead of 20 mm past the mouth estimate; the scorer
needs the centre >= ~18 mm inside the floor edge). s3 docstring follows.
sweep s3 = w2 + trailing head yaw (one mechanism). Evidence: w2 DEV0/13 diagnosis:
the cube is pushed near one END of the 116 mm head, the other 90 mm stick out sideways and hit the pan side-wall front (dustpan_collision_43/44)
at the mouth, so the cube stops at pan-local y -0.106..-0.113. s1 (head centred in the pan) made the right wrist collide with the left wrist
(no clear plan, 2/20). Change: during every sweep the head is yawed by TRAIL_DEG about the cube contact point so that its long protruding
side trails behind the push (rotated toward -d); only the short contact end enters the mouth and the slanted face also deflects the cube
toward the pan centre. TRAIL_DEG = 0 restores w2 exactly. Planned with the same robot-only check.
w2 docstring follows.
sweep w2 = w1 + INSIDE_MIN .010 -> .030 (re-push unless the perceived cube centre is 30 mm past the mouth). Evidence: w1 smoke
DEV0/1/2 ended with the cube on the mouth lip (scorer pan-local y -0.111/-0.104) while the re-observation read 13-20 mm past the mouth, so no re-push;
the passing DEV3 read 33 mm.
w1: sweep w1 (Franka, simulation only) = LIFT2 sw5 with ONLY the embodiment layer swapped: the right-arm vs rest-of-robot
clearance check uses the Franka planning MJCF (rsi/franka_suite/model/franka_planning.xml, joint vectors only, its desk box disabled
so the check stays robot-vs-robot as in sw5). Shared skill (w9/sw5 unchanged): held-pan frame from the left TCP, SAM3 brush/cube masks, grasp
choice + full right-arm plan, sweep law, re-observation and re-push. Start pose: Franka left gripper around the dustpan handle at the source
contact-home TCP (franka.HOME_TASK, same for every seed). sw5 docstring follows.
sweep sw5 = sw3 run on setup sweep_g4 (left contact home tilted -35 deg about the pinch axis) with CLEAR_M 0.012 -> 0.004, the value used by the
offline plan evaluation (20/20 clear plans; 0/20 on sweep_g3). No other logic change.
sweep sw3 = sw2 + hand-side choice: the sweep head axis may be the w9 sign (axis_new. ax >= 0) or the opposite sign
(brush turned 180 deg about vertical, hand on the other side of the push line). Evidence: sw1 on sweep_g3 DEV0/1 and smoke DEV7 all ended with
robot_robot right_link27 vs left_d405_link at t~11 s (end of first sweep), i.e. the hand is on the left-wrist side; flipping the grasp rotation
alone (sw2) does not move the hand. The planned side (sign) is used by every sweep/re-push of the episode.
sweep sw2 = sw1 + pre-grasp robot-only plan of the whole right-arm motion. Evidence: sw1 on sweep_g3 DEV0 ended with
NumericalFailure robot_robot (right_link27 finger vs left_d405_link) at the end of the first sweep. Before the brush grasp, for each grasp
option (w9 choice first, then the swapped grip/push ends if the grip point stays >= 45 mm from the perceived cube along the head) and each
grasp rotation (demo-derived, 180 deg flip), the predicted TCP path (hover, grasp, lift, sweep start, every sweep step, retract) is IK-chained
from the current encoders and checked with the LIFT2 robot-only model (joint vectors + static robot MJCF, no objects): clearance >= CLEAR_M
between any right-arm body and the left arm / head / base. The first clear option is executed; if none is clear the w9 choice is used.
sweep sw1 (LIFT2, smooth_table_v1 + one constant rigid group transform, setup sweep_g2) = source w9 with the world-axis
priors re-expressed relative to the held dustpan (LIFT2-specific plan; the shared skill, perception and sweep law are w9):
  P2' pan mouth  = left TCP + MOUTH_D * (-x_c), x_c = horizontal part of the left TCP canonical x axis (source: -x_c == world -y, so P2 is
                   unchanged on the source scene); sweep target = mouth + FLOOR_IN * x_c (source: [pl.x, mouth.y + FLOOR_IN]).
  P3' head-end fallback = the end along -y_c of the left TCP (source: -y_c == world -x, i.e. P3 unchanged on the source scene).
  Heights follow the perceived table plane: grasp z = demo z + plane(grasp xy) - plane(demo xy); sweep z follows the plane along the path
  (the source scene table is tilted ~2.2 deg; the same plane, different xy).
  Right grasp: demo rotation turned by the brush yaw change (w9), plus a 180 deg flip about the approach axis tried when IK fails (LIFT2).
w9 docstring follows.
sweep w9 (from w7): closed-loop re-observation. After each sweep the head is lifted off the cube and backed away,
the cube is re-perceived (SAM3 mask on a fresh RGB-D frame) and, if it is not past the pan mouth by INSIDE_MIN, the
same sweep is repeated from behind the newly perceived cube (at most MAX_REPUSH times). Lifting also stops the head
from pressing the cube during the scorer dwell. Perception, grasp and the sweep motion itself = w7.
sweep w7 (from w6): sweep 10 mm further through the target and 0.7 s per 4 cm step (w6: cubes stopped short of the mouth).
sweep w6 (from w5): learned-mask perception. Brush and cube identities from SAM3 text prompts
on the observed RGB; geometry (backprojected depth, PCA axis, width profile) only inside those masks. The previous global
height/crop/connected-component brush detector is removed (no fallback). Motion = w5 unchanged.
w5 (from w4): fingers at one end of the head (20 or 95 mm from the near end), cube pushed by the other end (70 mm away).
w4/w2/w0: see their NOTES.md.

Observations only: io.observe() RGB-D + K + camera pose, encoders/TCP, io.demo priors, SAM3 (shared service).
Perception sources (logged per episode in 'perceived' / 'reobserved'):
  brush  SAM3 'orange stick' -> highest-score instance whose masked 3-D points span > 0.15 m; axis = PCA of masked xy.
  head end: within-mask width profile (head ~30 mm vs handle ~18 mm wide, the wider third is the head) when the two
  ends differ by > 4 mm; otherwise prior P3 (head is the -x end). The source used is logged.
  cube   SAM3 'orange cube' -> highest-score instance whose masked points span < 0.06 m; centre = masked point median.
  plane  local robust table plane fit (same algorithm as rsi.perception_geo.fit_plane, SVD with full_matrices=False).
Geometry-based (not learned) parts and why: table plane (no object identity involved), axis/width/extent inside a learned mask
(metric pose needs depth), pan mouth from the left TCP (P2: the pan is held, its pose is defined by the gripper).
Demo-derived priors (listed in NOTES.md): P1 left grasp 20 mm below demo TCP; P2 mouth 0.138 m from left TCP along -y;
P3 head is the -x end (fallback only), grasp at demo close height -5 mm and demo close rotation turned by the brush yaw change.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from PIL import Image
from rsi import perception as P, perception_geo as G, perception_sam3 as S
import mujoco
from rsi.franka_suite.common import franka as ROBOT  # Franka robot constants only (planning MJCF path, joint names)

CLEAR_M = .004


class RobotCheck:
    """sw2: right-arm vs rest-of-robot clearance from joint vectors only (static LIFT2 robot MJCF, objects disabled, no live sim data)."""
    def __init__(self):
        m = self.m = mujoco.MjModel.from_xml_path(str(ROBOT.PLANNING_MJCF)); self.d = mujoco.MjData(m)
        m.geom_margin[:] = CLEAR_M
        for g in range(m.ngeom):
            if m.body(int(m.geom_bodyid[g])).name.startswith('object') or m.geom(g).name == 'table':
                m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0
        self.adr = {a: np.array([m.joint(j).qposadr[0] for j in ROBOT.ARMS[a]['joints']]) for a in ('left', 'right')}
        self.right = {i for i in range(m.nbody) if m.body(i).name.startswith('right_')}

    def clear(self, q_left, q_right):
        m, d = self.m, self.d
        d.qpos[self.adr['left']] = q_left; d.qpos[self.adr['right']] = q_right; mujoco.mj_forward(m, d)
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 in self.right) != (b2 in self.right) and c.dist < CLEAR_M:
                return False, '%s|%s:%.4f' % (m.body(b1).name, m.body(b2).name, c.dist)
        return True, None


CHK = None

UP = np.array([0., 0., 1.])
LEFT_DZ = -.020   # P1
MOUTH_D = .138           # P2' (along -x_c of the left TCP)
FLOOR_IN = .055
DEMO_AXIS_YAW = np.radians(-13.4)   # P3
SWEEP_Z = .002
OVERSHOOT = .000   # s11 (s4/s9: .030)
SWEEP_STEP_SEC = .7
GRASP_DZ = -.005
HEAD_C = .0575
GRIP_NEAR, GRIP_FAR = .020, .095
PUSH_NEAR, PUSH_FAR = .025, .090
BRUSH_PROMPT, CUBE_PROMPT = 'orange stick', 'orange cube'
RETRACT_UP, RETRACT_BACK = .040, .060
INSIDE_MIN = .030   # perceived cube centre past the mouth line (masked centres read ~15 mm short: camera-side faces)
REOBS_SCORE, REOBS_N, REOBS_SPAN = .10, 100, .075   # re-observation only: cube partly occluded by the head or tilted on the pan
MAX_REPUSH = 2
TRAIL_DEG = 30.   # s3
END_PUSH = True   # s9
TRANSIT_Z = .060   # s12
FACE = 0. if END_PUSH else .013   # contact-face offset from the contact centre line


def yaw_of(v):
    return float(np.arctan2(v[1], v[0]))


def rotz(a):
    return Rot.from_euler('z', a).as_matrix()


def fit_plane(obs, iters=4, band=.008):
    """Copy of rsi.perception_geo.fit_plane with a reduced SVD (identical Vt; avoids an N x N U matrix)."""
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


def masked_points(obs, mask, plane, lo=.003, hi=.08):
    p, ys, xs = G.safe_points(obs)
    h = G.height(p, plane)
    k = mask[ys, xs] & (h > lo) & (h < hi)
    return p[k]


def instances(proc, obs, prompt):
    state = proc.set_image(Image.fromarray(obs['rgb']))
    masks, scores = S._prompt(proc, state, prompt)
    order = np.argsort(-np.asarray(scores)) if len(scores) else []
    return [(masks[i], float(scores[i])) for i in order]


def find_cube(obs, plane, proc, min_score=.4, min_n=20, max_span=.06):
    for m, s in instances(proc, obs, CUBE_PROMPT):
        q = masked_points(obs, m, plane, hi=.06)
        if s >= min_score and len(q) >= min_n and np.ptp(q[:, :2], 0).max() < max_span:
            return {'center': np.r_[np.median(q[:, :2], 0), np.median(q[:, 2])], 'score': s, 'n': len(q)}
    return None


def perceive_cube(io):
    """w9 re-observation: cube centre from a fresh RGB-D frame + SAM3 mask, or None when not found."""
    obs = io.observe()
    return find_cube(obs, fit_plane(obs), S.get_proc(), REOBS_SCORE, REOBS_N, REOBS_SPAN)


HEAD_DIR = np.array([-1., 0, 0])  # set in run(): -y_c of the left contact-home TCP (P3')


def perceive(io):
    obs = io.observe()
    plane = fit_plane(obs)
    proc = S.get_proc()
    cube = find_cube(obs, plane, proc)
    if cube is None:
        raise RuntimeError('perception: SAM3 cube not found')
    brush = None
    for m, s in instances(proc, obs, BRUSH_PROMPT):
        q = masked_points(obs, m, plane)
        if len(q) < 80: continue
        xy = q[:, :2] - q[:, :2].mean(0); _, v = np.linalg.eigh(np.cov(xy.T)); ax = v[:, -1]
        pr = xy @ ax
        if np.ptp(np.percentile(pr, [1, 99])) > .15 and s >= .2:
            brush = (q, ax, s); break
    if brush is None:
        raise RuntimeError('perception: SAM3 brush not found')
    q, ax, bscore = brush
    c = q[:, :2].mean(0); pr = (q[:, :2] - c) @ ax; perp = (q[:, :2] - c) @ np.array([-ax[1], ax[0]])
    lo, hi = np.percentile(pr, [1, 99]); third = (hi - lo) / 3

    def width(sel):
        return float(np.ptp(np.percentile(perp[sel], [5, 95]))) if sel.sum() > 15 else 0.
    w_lo, w_hi = width(pr < lo + third), width(pr > hi - third)
    if abs(w_lo - w_hi) > .004:
        head_at_lo = w_lo > w_hi; source = 'mask_width_profile'
    else:
        head_at_lo = float(ax[:2] @ HEAD_DIR[:2]) < 0; source = 'prior_P3_head_along_minus_y_c'
    if not head_at_lo:
        ax = -ax; pr = -pr; lo, hi = -hi, -lo
    near = c + ax * lo
    io.log('perceived', cube=cube['center'], cube_score=cube['score'], brush_near=near, brush_axis=ax, brush_len=hi - lo,
        brush_score=bscore, n_brush=len(q), head_end_source=source, width_head_end=max(w_lo, w_hi), width_other_end=min(w_lo, w_hi),
        prompts={'brush': BRUSH_PROMPT, 'cube': CUBE_PROMPT})
    return cube['center'], near, ax, plane


def plane_z(plane, xy):
    c, n = plane
    return float(c[2] - (n[0] * (xy[0] - c[0]) + n[1] * (xy[1] - c[1])) / n[2])


def trail_yaw(axis_new, d, contact_s):
    """s3: extra yaw about the contact point that moves the long side of the head (from the contact toward the far head end) back along -d."""
    if not TRAIL_DEG: return 0.
    p = axis_new if contact_s < HEAD_C else -axis_new
    th = np.radians(TRAIL_DEG); pn = p * np.cos(th) - d * np.sin(th)
    return float(np.arctan2(p[0] * pn[1] - p[1] * pn[0], p @ pn))


def sweep_path(cube_xy, target, ax, tcp0, Rt, head_in_tcp, plane, dz0, sign=1., contact_s=HEAD_C):
    """Same geometry as sweep(): list of (tcp position, rotation) for start-hover, start, every step, lift, back."""
    d = target - cube_xy; dist = float(np.linalg.norm(d)); d /= dist
    axis_new = np.array([d[1], -d[0]])
    if axis_new @ ax[:2] < 0: axis_new = -axis_new
    axis_new = sign * axis_new
    if END_PUSH: axis_new = -d
    Rs = rotz(yaw_of(axis_new) - yaw_of(ax[:2]) + (0. if END_PUSH else trail_yaw(axis_new, d, contact_s))) @ Rt

    def tcp_for(head_xy, dz):
        xy = head_xy - (Rs @ head_in_tcp)[:2]
        return np.r_[xy, plane_z(plane, xy) + dz0 + dz]
    start = cube_xy - d * (.016 + FACE + .012); end = cube_xy + d * (dist - .016 - FACE + OVERSHOOT)
    n = max(2, int(np.ceil(np.linalg.norm(end - start) / .04)))
    pts = ([tcp_for(start, SWEEP_Z + TRANSIT_Z)] if TRANSIT_Z else []) + [tcp_for(start, SWEEP_Z + .02), tcp_for(start, SWEEP_Z)] + [tcp_for(start + (end - start) * k / n, SWEEP_Z) for k in range(1, n + 1)]
    up = pts[-1] + RETRACT_UP * UP
    return [(p, Rs) for p in pts] + [(up, Rs), (up - np.r_[d, 0.] * RETRACT_BACK, Rs)]


def plan_right(io, legs):
    """IK-chain the right arm through legs (5 interpolation steps each, from encoders) with the robot-only clearance check."""
    from twin.practice.motion import waypoint
    K = io._w.kinematics; enc = io._w.encoders(); ql = enc[0]; q = enc[1]; T = K.fk('right', q)
    for p, R in legs:
        for k in range(1, 6):
            a, b = waypoint(T, p, R, k / 5)
            try:
                q = K.ik('right', a, b, q, max_iterations=300)
            except Exception:
                return False, 'ik'
            ok, why = CHK.clear(ql, q)
            if not ok:
                return False, why
        T = K.fk('right', q)
    return True, None


def run(io):
    global HEAD_DIR, CHK
    CHK = RobotCheck()
    ev = io.demo['events']
    close_r = next(e for e in ev if e['arm'] == 'right' and e['kind'] == 'close')
    g_demo, R_demo = np.asarray(close_r['position']), np.asarray(close_r['rotation'])
    # 1 hold the dustpan (P1)
    L = io.tcp('left'); pl = L[:3, 3].copy()
    xc = L[:3, 0].copy(); xc[2] = 0; xc /= np.linalg.norm(xc); yc = L[:3, 1].copy(); yc[2] = 0; yc /= np.linalg.norm(yc)
    HEAD_DIR = -yc
    io.move('left', pl + LEFT_DZ * UP, L[:3, :3], 1.0)
    io.grip('left', 0., .8)
    mouth = pl + MOUTH_D * (-xc)
    target = (mouth + FLOOR_IN * xc)[:2]
    io.log('pan_frame', mouth=mouth, target=target, x_c=xc, y_c=yc)
    # 2 perceive cube + brush (SAM3 masks)
    cube, near, ax, plane = perceive(io)
    dyaw = yaw_of(ax) - DEMO_AXIS_YAW
    along = float((cube[:2] - near[:2]) @ ax[:2])
    first = (GRIP_FAR, PUSH_NEAR) if along < HEAD_C else (GRIP_NEAR, PUSH_FAR)
    options = [first] + [o for o in ((GRIP_FAR, PUSH_NEAR), (GRIP_NEAR, PUSH_FAR)) if o != first and abs(o[0] - along) >= .045]
    if END_PUSH: first = (GRIP_FAR, 0.); options = [first]   # s9: grip 95 mm behind the tip, push with the tip face
    io.grip('right', .6, .4)
    chosen, tried = None, []
    for grasp_s, contact_s in options:
        grasp_xy = near[:2] + ax[:2] * grasp_s
        g = np.r_[grasp_xy, g_demo[2] + GRASP_DZ + plane_z(plane, grasp_xy) - plane_z(plane, g_demo[:2])]
        for flip, sign in ((0., 1.), (np.pi, 1.), (0., -1.), (np.pi, -1.)):
            Rc = rotz(dyaw) @ R_demo @ Rot.from_euler('z', flip).as_matrix()
            head_c = near[:2] + ax[:2] * contact_s
            hit = Rc.T @ (np.r_[head_c, g[2]] - g); dz0 = g[2] - plane_z(plane, g[:2])
            legs = [(g + .08 * UP, Rc), (g, Rc), (g + .02 * UP, Rc)] + sweep_path(cube[:2], target, ax, g, Rc, hit, plane, dz0, sign, contact_s)
            ok, why = plan_right(io, legs); tried.append((grasp_s, round(float(np.degrees(flip))), sign, why))
            if ok:
                chosen = (grasp_s, contact_s, g, Rc, sign); break
        if chosen: break
    io.log('right_plan', tried=tried, chosen=None if chosen is None else (chosen[0], chosen[1], chosen[4]))
    SIGN = 1. if chosen is None else chosen[4]
    if chosen is None:
        grasp_s, contact_s = first; grasp_xy = near[:2] + ax[:2] * grasp_s
        g = np.r_[grasp_xy, g_demo[2] + GRASP_DZ + plane_z(plane, grasp_xy) - plane_z(plane, g_demo[:2])]; Rg = None
        for flip in (0., np.pi):
            Rc = rotz(dyaw) @ R_demo @ Rot.from_euler('z', flip).as_matrix()
            try:
                io.move('right', g + .08 * UP, Rc, 2.0); Rg = Rc; break
            except Exception as e:
                if type(e).__name__ in ('NumericalFailure', 'Limit'): raise
        if Rg is None:
            raise RuntimeError('no reachable brush grasp rotation')
    else:
        grasp_s, contact_s, g, Rg, _ = chosen
        io.move('right', g + .08 * UP, Rg, 2.0)
    io.log('grasp_choice', grasp_s=grasp_s, contact_s=contact_s, cube_along=along)
    io.move('right', g, Rg, 1.2)
    io.grip('right', 0., .8)
    T = io.tcp('right'); tcp0 = T[:3, 3].copy(); Rt = T[:3, :3].copy()
    head_c = near[:2] + ax[:2] * contact_s
    head_in_tcp = Rt.T @ (np.r_[head_c, tcp0[2]] - tcp0)
    dz0 = tcp0[2] - plane_z(plane, tcp0[:2])
    io.move('right', tcp0 + .02 * UP, Rt, .6)

    # 4 sweep: head long axis perpendicular to the push direction, start behind the cube (z follows the table plane)
    def sweep(cube_xy, label):
        d = target - cube_xy; dist = float(np.linalg.norm(d)); d /= dist
        axis_new = np.array([d[1], -d[0]])
        if axis_new @ ax[:2] < 0: axis_new = -axis_new
        axis_new = SIGN * axis_new
        if END_PUSH: axis_new = -d
        turn = yaw_of(axis_new) - yaw_of(ax[:2]) + (0. if END_PUSH else trail_yaw(axis_new, d, contact_s)); Rs = rotz(turn) @ Rt

        def tcp_for(head_xy, dz):
            xy = head_xy - (Rs @ head_in_tcp)[:2]
            return np.r_[xy, plane_z(plane, xy) + dz0 + dz]
        start = cube_xy - d * (.016 + FACE + .012)
        if TRANSIT_Z:   # s12: lift in place, turn/move above the start, then descend
            T0 = io.tcp('right'); zt = max(T0[2, 3], tcp_for(start, SWEEP_Z + TRANSIT_Z)[2])
            io.move('right', np.r_[T0[:2, 3], zt], T0[:3, :3], .8)
            io.move('right', np.r_[tcp_for(start, SWEEP_Z + TRANSIT_Z)[:2], zt], Rs, 1.5)
            io.move('right', tcp_for(start, SWEEP_Z + .02), Rs, .8)
        else:
            io.move('right', tcp_for(start, SWEEP_Z + .02), Rs, 1.5)
        io.move('right', tcp_for(start, SWEEP_Z), Rs, .6)
        end = cube_xy + d * (dist - .016 - FACE + OVERSHOOT)
        n = max(2, int(np.ceil(np.linalg.norm(end - start) / .04)))
        for k in range(1, n + 1):
            io.move('right', tcp_for(start + (end - start) * k / n, SWEEP_Z), Rs, SWEEP_STEP_SEC)
        io.wait(.3)
        io.log('swept', label=label, start=start, end=end, target=target)
        T = io.tcp('right'); up = T[:3, 3] + RETRACT_UP * UP
        io.move('right', up, T[:3, :3], .8)
        io.move('right', up - np.r_[d, 0.] * RETRACT_BACK, T[:3, :3], .8)
        io.wait(.3)

    sweep(cube[:2], 'first')
    # 5 w9: re-observe the cube (SAM3) and push again while it is not past the mouth by INSIDE_MIN (measured along x_c)
    for k in range(MAX_REPUSH):
        c2 = perceive_cube(io)
        past = None if c2 is None else float((c2['center'][:2] - mouth[:2]) @ xc[:2])
        io.log('reobserved', cube=None if c2 is None else c2['center'], cube_score=None if c2 is None else c2['score'],
            past_mouth=past, mouth=mouth[:2], repush=k)
        if c2 is None or past >= INSIDE_MIN:
            break
        try:
            sweep(c2['center'][:2], f'repush_{k}')
        except Exception as e:
            io.log('repush_failed', error=f'{type(e).__name__}: {e}'); break
