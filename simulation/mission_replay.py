"""Re-run the actual nominal mission used in the map and decision example.

The original EnhancedPlanner performs all task and execution decisions.
AuditPlanner records its root alternatives and post-arrival alternatives without
changing its decisions. Results are checked against mission feasibility and
accounting invariants, without stored mission outcomes.

Dependencies: numpy, scipy, numba. Input tables are generated locally from
the task attributes in generate_nominal_case.py before each run.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
from numba import types
from numba.typed import Dict
import planner
from batch_adapter import BatchCase, iteration_v2, robust_core
from config import load_unified_policies
from enhanced_dp import EnhancedPlanner, EnhancedSpec, enhanced_value, grid
from generate_nominal_case import generate
from root_calibrated_core import (
    _maximum_capacity, _lognormal_cdf, _truncated_lognormal_first_moment,
)

HERE = Path(__file__).resolve().parent
NOMINAL_CASE = HERE / 'runtime' / 'nominal_case'
SEED = 2026092000
SPEC = EnhancedSpec(shortlist=9, time_step=25.0, energy_step=7.5, quadrature=3,
                    tail_weight=0.0, tail_depth=6, cost_power=0.5, rounding=1,
                    mode='both', arrival_rule='original', integration=0)


def root_values(continuation, state):
    """Expose only the root alternatives of the unchanged recursive DP.

    The recursion, discretization, workload binning, and shortlist are the same
    as enhanced_dp.py. Assertions compare the root maximum to the native plan.
    """
    b, s = continuation.base, continuation.spec
    ft, fe = b.bundle.iteration.flight_time, b.bundle.iteration.flight_energy_kj
    scores = []
    for task in range(1, len(b.weights)):
        if not state.remaining_mask & (1 << (task - 1)):
            continue
        if not iteration_v2.visit_ok_v2(state.node, state.time_s, state.energy_kj, task, ft, fe, b.deadline, b.pars):
            continue
        p, st, se = robust_core.calibrated_task_stats(state.node, state.time_s, state.energy_kj, task,
            b.points, b.probabilities, b.mu, b.sigma, ft, fe, b.deadline, b.pars,
            b.aa, b.bb, b.dd, b.ee, 0)
        cost = ((ft[state.node, task] + st) / max(b.pars[0] - state.time_s, 1e-12)
                + (fe[state.node, task] + se) / max(b.pars[1] - state.energy_kj, 1e-12))
        scores.append((float(b.weights[task] * p) / max(cost, 1e-12) ** s.cost_power, -task))
    ranked = sorted(scores, reverse=True)
    tasks = np.array([-x[1] for x in ranked[:s.shortlist]], np.int64)
    tailmask = sum(1 << (-item[1] - 1) for item in scores)
    for task in tasks:
        tailmask &= ~(1 << (int(task) - 1))
    cache = Dict.empty(types.int64, types.Tuple((types.float64, types.int64)))
    tail_cache = Dict.empty(types.int64, types.Tuple((types.float64, types.int64)))
    nt = math.ceil(b.pars[0] / s.time_step) + 3
    ne = math.ceil(b.pars[1] / s.energy_step) + 3
    common = (tasks, np.uint64(tailmask), s.time_step, s.energy_step, nt, ne, cache, tail_cache,
        continuation.moments, b.mu, b.sigma, ft, fe, b.weights, b.deadline,
        b.pars, b.aa, b.bb, b.dd, b.ee, s.tail_weight, s.tail_depth,
        s.cost_power, s.rounding, s.integration)
    t = grid(state.time_s, s.time_step, s.rounding) * s.time_step
    e = grid(state.energy_kj, s.energy_step, s.rounding) * s.energy_step
    mask = (1 << len(tasks)) - 1
    values = []
    for index, raw_task in enumerate(tasks):
        task = int(raw_task)
        if not iteration_v2.visit_ok_v2(state.node, t, e, task, ft, fe, b.deadline, b.pars):
            values.append({'task': task, 'expected_value': None, 'rounded_state_feasible': False})
            continue
        at, ae, dt, de = iteration_v2.resource_limits_v2(state.node, t, e, task, ft, fe, b.deadline, b.pars)
        rem = mask & ~(1 << index)
        sit, sie = grid(at, s.time_step, s.rounding), grid(ae, s.energy_step, s.rounding)
        skip = enhanced_value(task, sit, sie, rem, *common)[0]
        cap = _maximum_capacity(task, dt, de, b.pars, b.aa, b.bb, b.dd, b.ee)
        p = _lognormal_cdf(b.mu[task], b.sigma[task], cap)
        cap_moment = _truncated_lognormal_first_moment(b.mu[task], b.sigma[task], cap)
        bins = continuation.moments.shape[1] + 1
        value, previous = (1-p) * skip, 0.0
        for g in range(bins):
            pr = min(p, (g+1) / bins) - g / bins
            if pr <= 1e-14:
                break
            upper = cap_moment if p <= (g+1) / bins else continuation.moments[task, g]
            c = min(cap, (upper-previous) / pr)
            previous = upper
            ok, st, se, _, _, _ = robust_core.execute_virtual(c, dt, de, b.pars,
                b.aa[task], b.bb[task], b.dd[task], b.ee[task], 0)
            v = skip
            if ok:
                cit, cie = grid(at+st, s.time_step, s.rounding), grid(ae+se, s.energy_step, s.rounding)
                future = enhanced_value(task, cit, cie, rem, *common)[0]
                v = max(v, b.weights[task] + future)
            value += pr * v
        values.append({'task': task, 'expected_value': float(value), 'rounded_state_feasible': True})
    return {'screening': [{'rank': rank+1, 'task': -task, 'score': score,
                           'retained': rank < s.shortlist} for rank, (score, task) in enumerate(ranked)],
            'retained_tasks': [int(x) for x in tasks], 'candidate_values': values}


class AuditPlanner(EnhancedPlanner):
    enhanced_spec = SPEC
    top_records = []
    sub_records = []

    def select_task(self, state):
        task, detail = super().select_task(state)
        ps = self._plan_state(state)
        audit = root_values(self.continuation, ps)
        root_max = max([0.] + [x['expected_value'] for x in audit['candidate_values'] if x['expected_value'] is not None])
        assert math.isclose(root_max, detail['selected_value'], abs_tol=1e-8), (root_max, detail)
        audit.update(epoch=len(self.top_records), current_node=int(ps.node),
                     time_s=float(ps.time_s), energy_kj=float(ps.energy_kj),
                     remaining_tasks=[i for i in range(1, self.n_tasks+1) if ps.remaining_mask & (1 << (i-1))],
                     selected_task=int(task), selected_value=float(detail['selected_value']))
        self.top_records.append(audit)
        return task, detail

    def _best_action(self, task, workload, arrival, remaining, price_t, price_e):
        selected, value = super()._best_action(task, workload, arrival, remaining, price_t, price_e)
        at, ae, available_t, available_e = arrival
        alternatives = []
        for action in (self._skip_action(), *self._completion_actions(task, workload, available_t, available_e, price_t, price_e)):
            successor = self._successor(task, at, ae, remaining, action)
            future = self.continuation.plan(successor)[1]
            immediate = 0.0 if action.mode == 'skip' else float(self.weights[task])
            alternatives.append({**asdict(action), 'immediate_reward': immediate,
                'future_value': float(future), 'total_value': immediate+float(future),
                'resource_cost': price_t * action.service_time_s + price_e * action.service_energy_kj,
                'successor': asdict(successor)})
        cap = _maximum_capacity(task, available_t, available_e, self.pars, self.aa, self.bb, self.dd, self.ee)
        self.sub_records.append({'epoch': len(self.sub_records), 'task': int(task),
            'priority': float(self.weights[task]), 'observed_workload_gcy': float(workload),
            'workload_mean_gcy': float(math.exp(self.mu[task]+0.5*self.sigma[task]**2)),
            'mu': float(self.mu[task]), 'sigma': float(self.sigma[task]),
            'arrival_time_s': float(at), 'arrival_energy_kj': float(ae),
            'execution_time_available_s': float(available_t), 'execution_energy_available_kj': float(available_e),
            'maximum_feasible_workload_gcy': float(cap),
            'shadow_price_time': float(price_t), 'shadow_price_energy': float(price_e),
            'completion_feasible': len(alternatives) > 1, 'alternatives': alternatives,
            'selected_action': asdict(selected), 'selected_value': float(value)})
        return selected, value


def mission_geometry(case_path):
    """Return the geographical markers from the actual experiment input."""
    with (case_path / 'nodes.csv').open(newline='', encoding='utf-8') as stream:
        nodes = list(csv.DictReader(stream))
    base = next(item for item in nodes if item['node_type'] == 'base')
    return {
        'base': {'name': 'Base', 'lon': float(base['lon']), 'lat': float(base['lat'])},
        'mec': [
            {'name': f"MEC {int(item['mec_id'])}",
             'lon': float(item['lon']), 'lat': float(item['lat'])}
            for item in nodes if item['node_type'] == 'mec'
        ],
        'tasks': [
            {'idx': int(item['task_id']),
             'lon': float(item['lon']), 'lat': float(item['lat'])}
            for item in nodes if item['node_type'] == 'task'
        ],
    }


def run(seed=SEED, case_path=None):
    """Execute one mission and return data that both figure plotters accept.

    Generate the selected seed from the public task attributes without bundled
    input tables. Calls should be sequential because the original runner selects
    the planner through a module class reference. That reference is restored
    before this function returns.
    """
    started = time.perf_counter()
    seed = int(seed)
    case_path = generate(NOMINAL_CASE if case_path is None else case_path,
                         seed_start=seed, seed_count=1)
    case = BatchCase(case_path)
    bundle = case.build_bundle(seed)
    policy = load_unified_policies(HERE / 'policies.json')['CMSACR']
    AuditPlanner.top_records = []
    AuditPlanner.sub_records = []
    original_planner = planner.ConsistentACARPlanner
    try:
        planner.ConsistentACARPlanner = AuditPlanner
        result = planner.run_unified_mission(bundle, policy, record_trace=True)
    finally:
        planner.ConsistentACARPlanner = original_planner

    trace = json.loads(result.pop('trace'))
    metadata = case.metadata(seed)
    metadata.update(result)
    metadata.update(method='Proposed', config_id='U9')
    visited = [int(item['task']) for item in trace]
    completed = [item for item in trace if item['mode'] != 'skip']
    completed_weight = sum(bundle.iteration.tasks[int(item['task']) - 1].weight
                           for item in completed)
    validations = {
        'return_within_budgets': bool(result['return_success'])
            and 0.0 <= result['final_time_s'] <= metadata['t_max_s'] + 1e-7
            and 0.0 <= result['final_energy_kj'] <= metadata['e_max_kj'] + 1e-7,
        'unique_visits': len(visited) == len(set(visited)) == result['visited_tasks'],
        'completion_counts': len(completed) == result['completed_tasks']
            == result['local_actions'] + result['mec_actions']
            and len(trace) - len(completed) == result['skipped_tasks'],
        'completed_weight': math.isclose(completed_weight,
            result['weighted_completed'], abs_tol=1e-8, rel_tol=0.0),
        'completed_deadlines': all(item['time_s'] <=
            bundle.iteration.tasks[int(item['task']) - 1].deadline_s + 1e-7
            for item in completed),
        'route_matches_trace': result['route'] == '0-' +
            '-'.join(str(task) for task in visited) + '-0',
    }
    failed = [name for name, passed in validations.items() if not passed]
    if failed:
        raise AssertionError('Mission validation failed: ' + ', '.join(failed))
    return {
        'description': 'Fresh execution of the nominal mission and its Top/Sub decisions.',
        'seed': seed,
        'spec': asdict(SPEC),
        'policy': asdict(policy),
        'input_case': str(case_path),
        'mec_indexing': 'Internal MEC 0,1,2 correspond to map MEC 1,2,3; -1 denotes local/skip.',
        'value_definition': 'Expected sum of task priority; completion total = immediate reward + continuation value; skip total = continuation value.',
        'feasibility_note': 'An infeasible completion has no comparison value and is not plotted as a zero-value completion.',
        'validation': validations,
        'mission_result': metadata,
        'mission_geometry': mission_geometry(case_path),
        'trace': trace,
        'top_decisions': AuditPlanner.top_records,
        'sub_decisions': AuditPlanner.sub_records,
        'audit_elapsed_s': time.perf_counter() - started,
    }
