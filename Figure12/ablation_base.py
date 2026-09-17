"""Original screening and continuation primitives used by Figure 12 ablations.

Algorithm bodies are preserved from the original run_ablations.py. Only source
paths and experiment-driver code were removed for a portable, readable module.
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from numba import njit, types
from numba.typed import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation"))
from enhanced_dp import EnhancedPlanner, EnhancedContinuation, enhanced_value, grid
from run_mean_workload_dp import SPEC, BASE_POLICY as POLICY
from batch_adapter import iteration_sim, iteration_v2, robust_core
from root_calibrated_core import (_maximum_capacity, _lognormal_cdf,
                                  _truncated_lognormal_first_moment)

def screen_rows(base, state, spec=SPEC):
    """Exact frozen U9 screen statistics, before resource-grid rounding."""
    ft, fe = base.bundle.iteration.flight_time, base.bundle.iteration.flight_energy_kj
    rows = []
    for task in range(1, len(base.weights)):
        if not state.remaining_mask & (1 << (task - 1)):
            continue
        if not iteration_v2.visit_ok_v2(state.node, state.time_s, state.energy_kj,
                task, ft, fe, base.deadline, base.pars):
            continue
        p, st, se = robust_core.calibrated_task_stats(state.node, state.time_s,
            state.energy_kj, task, base.points, base.probabilities, base.mu,
            base.sigma, ft, fe, base.deadline, base.pars, base.aa, base.bb,
            base.dd, base.ee, 0)
        cost = ((ft[state.node, task] + st) / max(base.pars[0] - state.time_s, 1e-12)
              + (fe[state.node, task] + se) / max(base.pars[1] - state.energy_kj, 1e-12))
        score = float(base.weights[task] * p) / max(cost, 1e-12) ** spec.cost_power
        rows.append((score, -task))
    return rows


class ShortlistContinuation(EnhancedContinuation):
    """Same DP call as EnhancedContinuation, with an explicit ordering hook."""
    ordering = 'score'

    def __init__(self, base, spec):
        super().__init__(base, spec)
        self.shortlist_calls = self.shortlist_changed_calls = 0
        self.shortlist_set_changed_calls = 0
        self.last_tasks = []

    def selected_tasks(self, state, scores):
        original = [-item[1] for item in sorted(scores, reverse=True)[:self.spec.shortlist]]
        if self.ordering == 'distance':
            ft = self.base.bundle.iteration.flight_time
            chosen = sorted((-item[1] for item in scores),
                key=lambda task: (float(ft[state.node, task]), task))[:self.spec.shortlist]
        else:
            chosen = original
        self.shortlist_calls += 1
        self.shortlist_changed_calls += int(chosen != original)
        self.shortlist_set_changed_calls += int(set(chosen) != set(original))
        self.last_tasks = chosen
        return np.asarray(chosen, dtype=np.int64)

    def plan(self, state):
        key = (state.node, state.time_s, state.energy_kj, state.remaining_mask)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        b, s = self.base, self.spec
        ft, fe = b.bundle.iteration.flight_time, b.bundle.iteration.flight_energy_kj
        scores = screen_rows(b, state, s)
        tasks = self.selected_tasks(state, scores)
        tailmask = sum(1 << (-item[1] - 1) for item in scores)
        for task in tasks:
            tailmask &= ~(1 << (int(task) - 1))
        cache = Dict.empty(types.int64, types.Tuple((types.float64, types.int64)))
        tc = Dict.empty(types.int64, types.Tuple((types.float64, types.int64)))
        nt = math.ceil(b.pars[0] / s.time_step) + 3
        ne = math.ceil(b.pars[1] / s.energy_step) + 3
        value, task = enhanced_value(state.node, grid(state.time_s, s.time_step, s.rounding),
            grid(state.energy_kj, s.energy_step, s.rounding), (1 << len(tasks)) - 1,
            tasks, np.uint64(tailmask), s.time_step, s.energy_step, nt, ne, cache, tc,
            self.moments, b.mu, b.sigma, ft, fe, b.weights, b.deadline, b.pars,
            b.aa, b.bb, b.dd, b.ee, s.tail_weight, s.tail_depth, s.cost_power,
            s.rounding, s.integration)
        self.expansions += len(cache)
        self.cache[key] = int(task), float(value)
        return self.cache[key]


@njit(cache=True)
def immediate_value(node, it, ie, tasks, moments, mu, sigma, ft, fe, weights,
                    deadline, pars, aa, bb, dd, ee, time_step, energy_step):
    """Frozen G=3 U9 root integration with both successor values set to zero."""
    t, e = it * time_step, ie * energy_step
    if t + ft[node, 0] > pars[0] + 1e-9 or e + fe[node, 0] > pars[1] + 1e-9:
        return 0., 0, 0
    best, best_task, branches = 0., 0, 0
    bins = moments.shape[1] + 1
    for task in tasks:
        if not iteration_v2.visit_ok_v2(node, t, e, task, ft, fe, deadline, pars):
            continue
        at, ae, dt, de = iteration_v2.resource_limits_v2(node, t, e, task, ft, fe, deadline, pars)
        cap = _maximum_capacity(task, dt, de, pars, aa, bb, dd, ee)
        p = _lognormal_cdf(mu[task], sigma[task], cap)
        cap_moment = _truncated_lognormal_first_moment(mu[task], sigma[task], cap)
        value, previous = 0., 0.
        for g in range(bins):
            pr = min(p, (g + 1) / bins) - g / bins
            if pr <= 1e-14:
                break
            upper = cap_moment if p <= (g + 1) / bins else moments[task, g]
            workload = min(cap, (upper - previous) / pr)
            previous = upper
            ok, st, se, mode, freq, cost = robust_core.execute_virtual(workload, dt, de,
                pars, aa[task], bb[task], dd[task], ee[task], 0)
            value += pr * (weights[task] if ok else 0.)
            branches += 1
        if value > best + 1e-12 or (abs(value - best) <= 1e-12 and best_task > 0 and task < best_task):
            best, best_task = value, task
    return best, best_task, branches


class MyopicContinuation(ShortlistContinuation):
    def __init__(self, base, spec):
        super().__init__(base, spec)
        self.one_step_branches = self.zero_future_calls = 0

    def plan(self, state):
        key = (state.node, state.time_s, state.energy_kj, state.remaining_mask)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        b, s = self.base, self.spec
        tasks = self.selected_tasks(state, screen_rows(b, state, s))
        value, task, branches = immediate_value(state.node,
            grid(state.time_s, s.time_step, s.rounding),
            grid(state.energy_kj, s.energy_step, s.rounding), tasks,
            self.moments, b.mu, b.sigma, b.bundle.iteration.flight_time,
            b.bundle.iteration.flight_energy_kj, b.weights, b.deadline, b.pars,
            b.aa, b.bb, b.dd, b.ee, s.time_step, s.energy_step)
        self.one_step_branches += branches
        self.cache[key] = int(task), float(value)
        return self.cache[key]

    def value(self, state, immediate):
        self.calls += 1
        self.zero_future_calls += 1
        return float(immediate)


class ReferencePlanner(EnhancedPlanner):
    enhanced_spec = SPEC


class MeanInformationPlanner(EnhancedPlanner):
    # One information change, represented consistently throughout prediction.
    enhanced_spec = replace(SPEC, quadrature=1)


def validate_trace(bundle, row):
    """Reconstruct geometry, execution, deadlines and direct-return reserves."""
    mission = bundle.iteration
    cfg = mission.cfg
    position = mission.base_xy.copy()
    t = e = reward = 0.
    seen = set()
    time_error = energy_error = 0.
    margin_t = margin_e = margin_deadline = float('inf')
    for event in json.loads(row['trace']):
        i = int(event['task'])
        if i in seen:
            raise AssertionError('Repeated task in mission trace.')
        seen.add(i)
        task = mission.tasks[i - 1]
        c = float(task.workload_gcy)
        assert abs(c - float(event['revealed_workload_gcy'])) <= 1e-8
        flight = float(np.linalg.norm(task.xy - position)) / float(cfg.speed_mps)
        t += flight
        e += flight * float(cfg.flight_power_w) / 1000.
        st = se = 0.
        if event['mode'] == 'local':
            f = float(event['frequency_ghz'])
            assert cfg.local_f_min_ghz - 1e-9 <= f <= cfg.local_f_max_ghz + 1e-9
            st = c / f
            se = cfg.kappa_kj_per_gcycle_ghz2 * c * f * f + cfg.hover_power_w * st / 1000.
        elif event['mode'] == 'mec':
            m = int(event['mec'])
            ru, rd = float(task.ul_rates[m]), float(task.dl_rates[m])
            assert ru >= cfg.ul_min_mbps - 1e-9 and rd >= cfg.dl_min_mbps - 1e-9
            tu, td = task.data_mbit / ru, task.data_mbit * task.output_ratio / rd
            st = tu + c / cfg.mec_cpu_ghz[m] + td
            se = (cfg.tx_power_w * tu + cfg.rx_power_w * td + cfg.hover_power_w * st) / 1000.
        else:
            assert event['mode'] == 'skip'
        t += st
        e += se
        position = task.xy
        if event['mode'] != 'skip':
            margin_deadline = min(margin_deadline, task.deadline_s - t)
            assert task.deadline_s - t >= -1e-6
            reward += task.weight
        ret = float(np.linalg.norm(position - mission.base_xy)) / float(cfg.speed_mps)
        margin_t = min(margin_t, cfg.t_max - t - ret)
        margin_e = min(margin_e, cfg.e_max_kj - e - ret * cfg.flight_power_w / 1000.)
        assert margin_t >= -1e-6 and margin_e >= -1e-6
        time_error = max(time_error, abs(t - float(event['time_s'])))
        energy_error = max(energy_error, abs(e - float(event['energy_kj'])))
        assert time_error < 1e-6 and energy_error < 1e-6
    ret = float(np.linalg.norm(position - mission.base_xy)) / float(cfg.speed_mps)
    t += ret
    e += ret * cfg.flight_power_w / 1000.
    assert abs(t - row['final_time_s']) < 1e-6 and abs(e - row['final_energy_kj']) < 1e-6
    assert abs(reward - row['weighted_completed']) < 1e-9
    assert abs(reward / sum(task.weight for task in mission.tasks) - row['safe_mcr']) < 1e-9
    assert row['return_success'] == 1
    return {'trace_verified': True, 'verified_actions': len(seen),
        'max_time_reconstruction_error_s': time_error,
        'max_energy_reconstruction_error_kj': energy_error,
        'min_direct_return_time_margin_s': margin_t,
        'min_direct_return_energy_margin_kj': margin_e,
        'min_completion_deadline_margin_s': margin_deadline}
