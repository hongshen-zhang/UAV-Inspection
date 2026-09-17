"""Reproduce Proposed and the five original comparison algorithms.

run(public_arrays, hidden_workloads, method='Proposed', L=9) returns one
mission's metrics. Workload vectors include an unused base entry at index 0.
All units are seconds, kJ, Gcycles, and GHz. Public arrays contain no sampled
workloads. The driver reveals one scalar only after the next task is chosen.
"""
from __future__ import annotations

import json
from time import perf_counter
import numpy as np
from .controller import (BasePlanner, PlanState, PriorityPlanner,
    MeanWorkloadPlanner, MyopicPlanner, NearestNeighborPlanner)

METHODS = ('Proposed', 'Weight Greedy', 'Mean-workload Greedy',
           'Mean-workload DP', 'Distribution-aware Myopic', 'Nearest Neighbor')


class NormalizedExecutor(BasePlanner):
    """Execution-only ablation; the original stochastic DP is unchanged."""
    def _prices(self, task, arrival_t, arrival_e, remaining):
        available_t = min(self.deadline[task] - arrival_t,
            self.pars[0] - arrival_t - self.mission.flight_time[task, 0])
        available_e = (self.pars[1] - arrival_e
            - self.mission.flight_energy_kj[task, 0])
        return 1.0 / available_t, 1.0 / available_e


def make_planner(public_arrays, method='Proposed', L=9, **options):
    """Construct a controller from public constants and distributions only."""
    from .ablations import (NoPriorityPlanner, TopMyopicPlanner,
                            NoProactiveSkipPlanner)
    classes = {
        'Proposed': BasePlanner,
        'Weight Greedy': PriorityPlanner,
        'Priority Greedy': PriorityPlanner,
        'Mean-workload Greedy': MeanWorkloadPlanner,
        'Mean-workload DP': BasePlanner,
        'Distribution-aware Myopic': MyopicPlanner,
        'Nearest Neighbor': NearestNeighborPlanner,
        'Normalized resource cost': NormalizedExecutor,
        'No priority factor': NoPriorityPlanner,
        'Myopic Top': TopMyopicPlanner,
        'No proactive skip': NoProactiveSkipPlanner,
    }
    if method not in classes:
        raise ValueError(f'Unknown method: {method}')
    return classes[method](public_arrays, mean=method == 'Mean-workload DP',
                           L=L, **options)


def run(public_arrays, hidden_workloads, method='Proposed', L=9, *,
        record_trace=True, seed=None, **options):
    """Run one paired mission with the original deterministic tie rules.

    Optional numerical settings: time_step=25, energy_step=7.5,
    quadrature=3 (DP bins), support_order=9 (GH nodes). Timing excludes
    controller construction and can include JIT warm-up on the first call.
    Callers must warm up before using runtime values as measurements.
    """
    controller = make_planner(public_arrays, method, L, **options)
    n = controller.n_tasks
    state = PlanState(0, 0.0, 0.0, (1 << n) - 1)
    reward = 0.0
    completed = skipped = local = mec = 0
    trace, route, selection_ms, execution_ms = [], [0], [], []
    started = perf_counter()
    for epoch in range(n):
        tick = perf_counter()
        task, detail = controller.select_task(state)
        selection_ms.append((perf_counter() - tick) * 1000.0)
        task = int(task)
        if task == 0:
            break
        assert state.remaining_mask & (1 << (task - 1))
        assert controller._arrival(state, task) is not None
        # The actual workload is read only after the route choice. It is never
        # stored in the planner or passed to its task-selection method.
        workload = float(hidden_workloads(task) if callable(hidden_workloads)
                         else hidden_workloads[task])
        tick = perf_counter()
        action = controller.choose_actual_action(state, task, workload)
        execution_ms.append((perf_counter() - tick) * 1000.0)
        next_t = state.time_s + float(controller.mission.flight_time[state.node, task])
        next_e = state.energy_kj + float(controller.mission.flight_energy_kj[state.node, task])
        next_t += float(action.service_time_s)
        next_e += float(action.service_energy_kj)
        state = PlanState(task, next_t, next_e,
                          state.remaining_mask & ~(1 << (task - 1)))
        route.append(task)
        if action.mode == 'skip':
            skipped += 1
        else:
            completed += 1
            reward += float(controller.weights[task])
            local += int(action.mode == 'local')
            mec += int(action.mode == 'mec')
            assert state.time_s <= controller.deadline[task] + 1e-7
        assert state.time_s + controller.mission.flight_time[task, 0] <= controller.pars[0] + 1e-7
        assert state.energy_kj + controller.mission.flight_energy_kj[task, 0] <= controller.pars[1] + 1e-7
        if record_trace:
            trace.append(dict(epoch=epoch, task=task,
                revealed_workload_gcy=workload, mode=action.mode,
                mec=int(action.mec), frequency_ghz=float(action.frequency_ghz),
                time_s=float(state.time_s), energy_kj=float(state.energy_kj),
                service_time_s=float(action.service_time_s),
                service_energy_kj=float(action.service_energy_kj), **detail))
    final_t = state.time_s + float(controller.mission.flight_time[state.node, 0])
    final_e = state.energy_kj + float(controller.mission.flight_energy_kj[state.node, 0])
    route.append(0)
    residual = max(0.0, final_t - float(controller.pars[0]),
                   final_e - float(controller.pars[1]))
    total = float(controller.weights.sum())
    result = dict(method=method, wcr=reward / total,
        safe_mcr=reward / total if residual <= 1e-7 else 0.0,
        weighted_completed=reward, total_weight=total,
        completed_tasks=completed, visited_tasks=completed + skipped,
        skipped_tasks=skipped, local_actions=local, mec_actions=mec,
        final_time_s=final_t, final_energy_kj=final_e,
        return_success=int(residual <= 1e-7), max_constraint_residual=residual,
        route='-'.join(str(task) for task in route),
        runtime_ms=(perf_counter() - started) * 1000.0,
        planner_runtime_ms=sum(selection_ms) + sum(execution_ms),
        selection_times_ms=selection_ms, execution_times_ms=execution_ms,
        trace=json.dumps(trace, separators=(',', ':')) if record_trace else '',
        continuation_expansions=int(controller.continuation.expansions))
    if seed is not None:
        result['seed'] = int(seed)
    return result
