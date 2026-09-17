#!/usr/bin/env python3
"""Run the six interpretable paired baselines on a formal N=20 case."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESEARCH = HERE

import planner  # type: ignore
from batch_adapter import BatchCase, robust_core  # type: ignore
from config import load_unified_policies  # type: ignore
from study_policies import HeuristicPlanner  # type: ignore
from batch_adapter import iteration_v2  # type: ignore
from enhanced_dp import EnhancedPlanner, EnhancedSpec  # type: ignore


BASE_CLASS = planner.ConsistentACARPlanner
POLICIES = load_unified_policies(RESEARCH / "policies.json")
BASE_POLICY = POLICIES["CMSACR"]


class PriorityPlanner(HeuristicPlanner):
    def select_task(self, state):
        self.stats["route_calls"] += 1
        plan = self._plan_state(state)
        best = 0
        best_key = None
        for task in range(1, self.n_tasks + 1):
            if not plan.remaining_mask & (1 << (task - 1)):
                continue
            if self._arrival(plan, task) is None:
                continue
            key = (-float(self.weights[task]), float(self.mission.flight_time[plan.node, task]), task)
            if best_key is None or key < best_key:
                best, best_key = task, key
        return int(best), {"selection_mode": "priority_greedy"}


def _mean_completion(plan, task, planner_obj):
    """Return the deterministic arithmetic-mean workload service tuple.

    Cost and Mean-workload Greedy use exactly one workload value (the
    arithmetic mean) for route ranking.  The physical feasibility check and
    normalized service cost are still evaluated with the same execution model
    as the online planner.
    """
    arrival = planner_obj._arrival(plan, task)
    if arrival is None:
        return None
    _at, _ae, dt, de = arrival
    mean_c = float(np.exp(planner_obj.mu[task] + 0.5 * planner_obj.sigma[task] ** 2))
    ok, st, se, _mode, _freq, _cost = iteration_v2.execute_virtual_v2(
        mean_c, dt, de, planner_obj.pars, planner_obj.aa[task],
        planner_obj.bb[task], planner_obj.dd[task], planner_obj.ee[task],
        0, 1.0, 1.0, 0.0,
    )
    if not ok:
        return None
    normalized = (
        float(st) / max(float(dt), 1e-12)
        + float(se) / max(float(de), 1e-12)
    )
    return float(st), float(se), float(normalized)


class CostPlanner(HeuristicPlanner):
    def select_task(self, state):
        self.stats["route_calls"] += 1
        plan = self._plan_state(state)
        best = 0
        best_key = None
        for task in range(1, self.n_tasks + 1):
            if not plan.remaining_mask & (1 << (task - 1)):
                continue
            arrival = self._arrival(plan, task)
            if arrival is None:
                continue
            mean_service = _mean_completion(plan, task, self)
            if mean_service is None:
                continue
            service_t, service_e, _ = mean_service
            # Normalize by the fixed total mission budgets.  This keeps the
            # route score a task-level cost, independent of the current state.
            cost = ((self.mission.flight_time[plan.node, task] + service_t)
                    / max(float(self.pars[0]), 1e-12)
                    + (self.mission.flight_energy_kj[plan.node, task] + service_e)
                    / max(float(self.pars[1]), 1e-12))
            key = (float(cost), float(self.mission.flight_time[plan.node, task]), task)
            if best_key is None or key < best_key:
                best, best_key = task, key
        return int(best), {"selection_mode": "cost_greedy"}


class MeanWorkloadPlanner(HeuristicPlanner):
    """Greedy route ranked by priority divided by mean-workload cost."""

    def select_task(self, state):
        self.stats["route_calls"] += 1
        plan = self._plan_state(state)
        best = 0
        best_key = None
        for task in range(1, self.n_tasks + 1):
            if not plan.remaining_mask & (1 << (task - 1)):
                continue
            arrival = self._arrival(plan, task)
            if arrival is None:
                continue
            mean_service = _mean_completion(plan, task, self)
            if mean_service is None:
                continue
            service_t, service_e, _ = mean_service
            cost = (
                (self.mission.flight_time[plan.node, task] + service_t)
                / max(float(self.pars[0]), 1e-12)
                + (self.mission.flight_energy_kj[plan.node, task] + service_e)
                / max(float(self.pars[1]), 1e-12)
            )
            score = float(self.weights[task]) / max(float(cost), 1e-12)
            key = (-score, float(self.mission.flight_time[plan.node, task]), task)
            if best_key is None or key < best_key:
                best, best_key = task, key
        return int(best), {"selection_mode": "mean_workload_greedy"}


class MyopicPlanner(HeuristicPlanner):
    def select_task(self, state):
        self.stats["route_calls"] += 1
        plan = self._plan_state(state)
        best = 0
        best_key = None
        for task in range(1, self.n_tasks + 1):
            if not plan.remaining_mask & (1 << (task - 1)):
                continue
            arrival = self._arrival(plan, task)
            if arrival is None:
                continue
            p, _service_t, _service_e = robust_core.calibrated_task_stats(
                plan.node, plan.time_s, plan.energy_kj, task,
                self.price_points, self.price_probabilities, self.mu, self.sigma,
                self.mission.flight_time, self.mission.flight_energy_kj,
                self.deadline, self.pars, self.aa, self.bb, self.dd, self.ee, 0,
            )
            key = (-float(self.weights[task] * p),
                   float(self.mission.flight_time[plan.node, task]), task)
            if best_key is None or key < best_key:
                best, best_key = task, key
        return int(best), {"selection_mode": "distribution_aware_myopic"}


class LocalOnlyPlanner(EnhancedPlanner):
    """U9 planner with Local as the only admissible execution mode."""

    enhanced_spec = EnhancedSpec(
        shortlist=9, time_step=25.0, energy_step=7.5, quadrature=3,
        tail_weight=0.0, tail_depth=6, cost_power=0.5, rounding=1,
        mode="local", arrival_rule="original", integration=0,
    )


class MECOnlyPlanner(EnhancedPlanner):
    """U9 planner with MEC as the only admissible execution mode."""

    enhanced_spec = EnhancedSpec(
        shortlist=9, time_step=25.0, energy_step=7.5, quadrature=3,
        tail_weight=0.0, tail_depth=6, cost_power=0.5, rounding=1,
        mode="mec", arrival_rule="original", integration=0,
    )


