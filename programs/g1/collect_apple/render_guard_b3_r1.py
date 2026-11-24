# render_guard_cubes_r3 (r2 accepted two identical BLANK frames: readbacks can lag several renders and start blank in a fresh context): TRUSTED-SIDE observation integrity for G1 b2 runs on a shared GPU 0.
# Finding: with other CUDA/EGL jobs on GPU 0, head-cam readbacks were intermittently corrupted
# (vertically flipped, 1-byte shifted, NaN depth) and returned arrays were later overwritten in place -> SAM3 masks did not line up with depth.
# Guard: (1) every Renderer.render reads into a persistent ring buffer and calls glFinish first, so a late write never lands in freed memory;
# (2) per observation it renders at least MIN_RENDERS times and accepts only when the LAST 3 renders of the same (unstepped) physics state agree exactly (rgb) / to 1e-6 (depth)
# and pass a scene-free sanity check (rgb not blank: mean > 20 and std > 5; depth median in 0.2..2.0 m); otherwise keeps re-rendering (up to MAX_TRIES). Uses no object state, scorer or contacts; it only re-renders the same frame. Counts are logged.
import numpy as np
import mujoco as mj

RING = 96
MIN_RENDERS = 6
MAX_TRIES = 30
STATS = {'observe_calls': 0, 'renders': 0, 'invalid_frames': 0, 'extra_renders_beyond_min': 0, 'backend_faults': 0, 'max_tries_used': 0, 'gave_up': 0}
_ring = []


def _gl_finish():
    try:
        from OpenGL import GL
        GL.glFinish()
    except Exception:
        pass


def install_render_ring():
    if getattr(mj.Renderer, '_rsi_ring_installed', False):
        return
    orig = mj.Renderer.render

    def render(self, *, out=None):
        if out is None:
            shape = (self._height, self._width) if self._depth_rendering else (self._height, self._width, 3)
            out = np.zeros(shape, np.float32 if self._depth_rendering else np.uint8)
            _ring.append(out)
            if len(_ring) > RING:
                _ring.pop(0)
        if self._gl_context:
            self._gl_context.make_current()
        _gl_finish()
        r = orig(self, out=out)
        _gl_finish()
        return r
    mj.Renderer.render = render
    mj.Renderer._rsi_ring_installed = True


def guard_observe(cb):
    orig = cb.observe

    def one():
        try:
            o = orig()
        except Exception as e:
            if type(e).__name__ != 'BackendFault':
                raise
            STATS['backend_faults'] += 1
            return None
        o = dict(o)
        o['rgb'] = np.array(o['rgb'], copy=True)
        o['depth'] = np.array(o['depth'], copy=True)
        return o

    def valid(o):
        if o is None:
            return False
        r, d = o['rgb'], o['depth']
        ok = float(r.mean()) > 20 and float(r.std()) > 5 and 0.2 < float(np.median(d)) < 2.0
        if not ok:
            STATS['invalid_frames'] += 1
        return ok

    def same(a, b):
        return np.array_equal(a['rgb'], b['rgb']) and np.allclose(a['depth'], b['depth'], atol=1e-6, rtol=0)

    def observe():
        STATS['observe_calls'] += 1
        hist = []
        for k in range(1, MAX_TRIES + 1):
            cur = one()
            STATS['renders'] += 1
            hist.append(cur if valid(cur) else None)
            hist = hist[-3:]
            if k >= MIN_RENDERS and len(hist) == 3 and all(x is not None for x in hist) and same(hist[0], hist[1]) and same(hist[1], hist[2]):
                STATS['max_tries_used'] = max(STATS['max_tries_used'], k)
                STATS['extra_renders_beyond_min'] += k - MIN_RENDERS
                return hist[-1]
        STATS['gave_up'] += 1
        raise RuntimeError('render_guard: no stable valid frame in %d renders' % MAX_TRIES)
    cb.observe = observe
    return STATS
