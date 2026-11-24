# Scorer wiring for the G1 suite (r2). Trusted side only. Uses the accepted role-only scorers in ../scoring; no scorer logic is reimplemented here.
# Families: generic (inside/nested/on/in_drawer), box (strict v2), sweep (v3 with per-step Monitor). Caps, 11-sample dwell and force checks are preserved by the scorers/run_episode.
import os

FAMILY = {'lift_box': 'box', 'sweep_objects': 'sweep'}


def family(task):
    return FAMILY.get(task, 'generic')


def set_sam_env():
    # must run BEFORE any source policy import; reuses the existing service, never starts or restarts it
    os.environ['RSI_SAM3_REMOTE'] = '1'
    os.environ['RSI_SAM3_ADDR'] = __import__('tempfile').gettempdir() + '/sam3.sock'


class CompositeGuard:
    # pre/post step hooks for make_public_backend: non-robot guard always; sweep Monitor when the family needs it
    def __init__(self, guard, monitor, model, data):
        self.guard, self.monitor, self.model, self.data = guard, monitor, model, data

    def pre_step(self):
        self.guard.pre_step()
        if self.monitor is not None:
            self.monitor.pre_step(self.model, self.data)

    def post_step(self):
        self.guard.post_step()
        if self.monitor is not None:
            self.monitor.update(self.model, self.data)


class Scoring:
    def __init__(self, task, handle, record=None):
        from ..scoring import g1_scorer as S, g1_roles
        self.S, self.task, self.h = S, task, handle
        self.fam = family(task)
        self.roles = g1_roles.roles(handle.model)
        self.monitor = S.Monitor(handle.model, handle.data, self.roles) if self.fam == 'sweep' else None
        self.task_obj = None
        if self.fam != 'sweep':
            from rsi.tasks import get
            self.task_obj = get(task)
        self.record = record

    def guard_for(self):
        return CompositeGuard(self.h.guard, self.monitor, self.h.model, self.h.data)

    def snapshot(self):
        m, d, S = self.h.model, self.h.data, self.S
        if self.fam == 'generic':
            return S.generic_snapshot(self.task_obj, m, d, self.roles['robot'])
        if self.fam == 'box':
            return S.box_snapshot(self.task_obj, m, d, self.roles['robot'])
        return S.sweep_snapshot(m, d, self.roles)

    def evaluate(self, res):
        S = self.S
        nv, ce = bool(res['numerical_invalid']), res['controller_error']
        vetoed = bool(res['latched_error'] or res['time_limit'] or nv or ce or not res['dwell_ok'])
        if self.fam == 'generic':
            r = S.generic_evaluate(res['history']) if res['dwell_ok'] else None
        elif self.fam == 'box':
            r = S.box_evaluate(res['history'], nv, ce) if res['dwell_ok'] else None
        else:
            scored = S.sweep_evaluate(res['history'], self.monitor.summary(), nv) if res['dwell_ok'] else None
            r = S.sweep_final_success(scored, ce, nv) if scored is not None else None
        return {'family': self.fam, 'vetoed': vetoed, 'scorer_result': r}
