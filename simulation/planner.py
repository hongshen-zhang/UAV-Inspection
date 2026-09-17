"""Unified route and arrival controller for the consistent ACAR redesign.

The same ``ContinuationEvaluator`` object is called in both stages.  The
current task reward is always added with coefficient one.  The guidance weight
can only act on the LP opportunity value inside the continuation evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr, ndtri


HERE = Path(__file__).resolve().parent
SHARED_SRC = HERE
if str(SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(SHARED_SRC))

from batch_adapter import (  # noqa: E402
    iteration_choice,
    iteration_sim,
    iteration_v2,
    optimized_core,
    robust_core,
    visited_mask,
)
import calibrated_beam_core  # noqa: E402
import executable_rollout_core  # noqa: E402
import root_calibrated_core  # noqa: E402

from config import PolicySpec  # noqa: E402


EPS = 1e-12


@dataclass(frozen=True)
class PlanState:
    node: int
    time_s: float
    energy_kj: float
    remaining_mask: int


@dataclass(frozen=True)
class ExecutionAction:
    mode: str
    mec: int
    frequency_ghz: float
    service_time_s: float
    service_energy_kj: float
    occupation: float


def immediate_reward(priority: float, action: ExecutionAction) -> float:
    """Return the original MTE reward without a configurable coefficient."""

    return float(priority) if action.mode != "skip" else 0.0


def route_choice_is_better(
    candidate_value: float,
    candidate_task: int,
    incumbent_value: float,
    incumbent_task: int,
) -> bool:
    """Compare a task with the zero-value stop action and a current incumbent."""

    if float(candidate_value) > float(incumbent_value) + EPS:
        return True
    if abs(float(candidate_value) - float(incumbent_value)) > EPS:
        return False
    return int(incumbent_task) != 0 and int(candidate_task) < int(incumbent_task)


def _all_mask(n_tasks: int) -> int:
    return (1 << int(n_tasks)) - 1


def _visited_mask(state: Any) -> int:
    result = 0
    for task in state.visited:
        result |= 1 << (int(task) - 1)
    return result


def deterministic_mean_support(
    mu: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse each LogNormal workload to its arithmetic mean.

    The returned ``mu`` and ``sigma`` describe a degenerate LogNormal at
    ``E[C]``.  Passing those arrays to the analytic completion-mass code is
    essential: changing only the quadrature points would still let the planner
    use the original distribution through its exact CDF.
    """

    mean = np.exp(
        np.asarray(mu, dtype=np.float64)
        + np.square(np.asarray(sigma, dtype=np.float64)) / 2.0
    )
    points = mean[:, np.newaxis]
    probabilities = np.ones(1, dtype=np.float64)
    deterministic_mu = np.log(np.maximum(mean, np.finfo(np.float64).tiny))
    deterministic_sigma = np.zeros_like(deterministic_mu)
    return points, probabilities, deterministic_mu, deterministic_sigma


class ContinuationEvaluator:
    """One shared deterministic continuation evaluator.

    It uses exact LogNormal completion masses inside the continuation Beam and
    deterministic GH or midpoint-quantile support for resource moments.  Route
    and arrival calls share this same object and therefore the same support,
    Beam width, lookahead depth, and guidance rule.
    """

    def __init__(self, bundle: Any, policy: PolicySpec) -> None:
        self.bundle = bundle
        self.spec = policy.continuation
        (
            self.weights,
            self.deadline,
            self.mu,
            self.sigma,
            self.pars,
            self.aa,
            self.bb,
            self.dd,
            self.ee,
        ) = bundle.robust_arrays
        if self.spec.support_rule == "gh":
            points, probabilities = optimized_core.gh_workload_points(
                self.mu, self.sigma, int(self.spec.support_order)
            )
            nodes, _weights = hermgauss(int(self.spec.support_order))
            self.conditional_uniform = np.asarray(
                ndtr(np.sqrt(2.0) * nodes), dtype=np.float64
            )
        elif self.spec.support_rule == "quantile":
            points, probabilities = iteration_v2.quantile_workload_points(
                self.mu, self.sigma, int(self.spec.support_order)
            )
            self.conditional_uniform = np.asarray(
                (
                    np.arange(int(self.spec.support_order), dtype=float)
                    + (1.0 / 2.0)
                )
                / float(self.spec.support_order),
                dtype=np.float64,
            )
        else:
            (
                points,
                probabilities,
                self.mu,
                self.sigma,
            ) = deterministic_mean_support(self.mu, self.sigma)
            self.conditional_uniform = np.asarray([1.0 / 2.0], dtype=np.float64)
        self.points = np.asarray(points, dtype=np.float64)
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.calls = 0
        self.cache_hits = 0
        self.expansions = 0
        self._value_cache: dict[tuple[int, float, float, int, float], float] = {}
        self.scenarios, self.scenario_probabilities = self._scenario_bank()

    def _scenario_bank(self) -> tuple[np.ndarray, np.ndarray]:
        """Build a deterministic equal weight Latin hypercube scenario bank."""

        if self.spec.evaluator != "executable_rollout":
            return np.zeros((0, len(self.mu)), dtype=np.float64), np.zeros(
                0, dtype=np.float64
            )
        order = int(self.spec.support_order)
        profiles = int(self.spec.rollout_profiles)
        scenarios = np.zeros(
            (order * profiles, len(self.mu)), dtype=np.float64
        )
        row = 0
        for profile in range(profiles):
            multiplier = 2 * profile + 1
            while math.gcd(multiplier, order) != 1:
                multiplier += 1
            for stratum in range(order):
                for task in range(1, len(self.mu)):
                    offset = (
                        task * (profile + 1) + task * task + profile * profile
                    ) % order
                    index = (multiplier * stratum + offset) % order
                    scenarios[row, task] = float(self.points[task, index])
                row += 1
        probabilities = np.full(
            order * profiles,
            1.0 / float(order * profiles),
            dtype=np.float64,
        )
        return scenarios, probabilities

    @property
    def signature(self) -> str:
        return self.spec.fingerprint

    def value(self, state: PlanState, immediate: float) -> float:
        """Evaluate one successor with the common continuation definition."""

        self.calls += 1
        key = (
            int(state.node),
            float(state.time_s),
            float(state.energy_kj),
            int(state.remaining_mask),
            float(immediate),
        )
        cached = self._value_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return float(cached)
        if int(state.remaining_mask) == 0:
            result = float(immediate)
            self._value_cache[key] = result
            return result
        if self.spec.evaluator == "executable_rollout":
            value, expanded = executable_rollout_core.rollout_value(
                int(state.node),
                float(state.time_s),
                float(state.energy_kj),
                np.uint64(state.remaining_mask),
                np.asarray(self.scenarios, dtype=np.float64),
                np.asarray(self.scenario_probabilities, dtype=np.float64),
                np.asarray(self.points, dtype=np.float64),
                np.asarray(self.probabilities, dtype=np.float64),
                self.mu,
                self.sigma,
                self.bundle.robust.flight_time,
                self.bundle.robust.flight_energy_kj,
                self.weights,
                self.deadline,
                self.pars,
                self.aa,
                self.bb,
                self.dd,
                self.ee,
                float(self.spec.base_cost_floor),
                float(self.spec.base_cost_power),
                float(self.spec.base_cluster_weight),
                float(self.spec.base_deadline_weight),
                int(self.spec.rollout_steps),
            )
        else:
            kernel = (
                calibrated_beam_core.beam_future_calibrated_cached
                if self.spec.cache_expected_stats
                else calibrated_beam_core.beam_future_calibrated
            )
            value, expanded = kernel(
                int(state.node),
                float(state.time_s),
                float(state.energy_kj),
                np.uint64(state.remaining_mask),
                np.asarray(self.points, dtype=np.float64),
                np.asarray(self.probabilities, dtype=np.float64),
                self.mu,
                self.sigma,
                self.bundle.robust.flight_time,
                self.bundle.robust.flight_energy_kj,
                self.weights,
                self.deadline,
                self.pars,
                self.aa,
                self.bb,
                self.dd,
                self.ee,
                int(self.spec.beam_width),
                float(self.spec.guidance.effective_weight),
                int(self.spec.lookahead_depth),
            )
        self.expansions += int(expanded)
        result = float(immediate) + float(value)
        self._value_cache[key] = result
        return result

    def candidate_tasks(self, state: PlanState) -> tuple[int, ...]:
        """Return all tasks or the fixed rollout policy shortlist."""

        if self.spec.evaluator != "executable_rollout":
            return tuple(
                task
                for task in range(1, len(self.weights))
                if state.remaining_mask & (1 << (task - 1))
            )
        tasks, _scores, count = executable_rollout_core.candidate_shortlist(
            int(state.node),
            float(state.time_s),
            float(state.energy_kj),
            np.uint64(state.remaining_mask),
            np.asarray(self.points, dtype=np.float64),
            np.asarray(self.probabilities, dtype=np.float64),
            self.mu,
            self.sigma,
            self.bundle.robust.flight_time,
            self.bundle.robust.flight_energy_kj,
            self.weights,
            self.deadline,
            self.pars,
            self.aa,
            self.bb,
            self.dd,
            self.ee,
            int(self.spec.candidate_width),
            float(self.spec.base_cost_floor),
            float(self.spec.base_cost_power),
            float(self.spec.base_cluster_weight),
            float(self.spec.base_deadline_weight),
        )
        return tuple(int(task) for task in tasks[: int(count)])


class ConsistentACARPlanner:
    """ACAR planner with one route/arrival continuation contract."""

    def __init__(self, bundle: Any, policy: PolicySpec) -> None:
        self.bundle = bundle
        self.policy = policy
        self.mission = bundle.iteration
        (
            self.weights,
            self.deadline,
            self.mu,
            self.sigma,
            self.pars,
            self.aa,
            self.bb,
            self.dd,
            self.ee,
        ) = bundle.iteration_arrays
        self.n_tasks = len(self.weights) - 1
        self.continuation = ContinuationEvaluator(bundle, policy)
        # Every planning component, including optional analytic root
        # calibration, must see the same workload representation.  For the
        # matched-mean control these are the collapsed deterministic arrays.
        self.mu = self.continuation.mu
        self.sigma = self.continuation.sigma
        # The shadow price LP uses the same workload support as continuation.
        self.price_points = self.continuation.points
        self.price_probabilities = self.continuation.probabilities
        self.stats: dict[str, int] = {
            "route_calls": 0,
            "arrival_calls": 0,
            "dual_price_calls": 0,
            "dual_price_cache_hits": 0,
            "action_value_calls": 0,
            "anchor_calls": 0,
            "anchor_retained": 0,
            "anchor_replaced": 0,
        }
        self._price_cache: dict[
            tuple[int, float, float, int], tuple[float, float]
        ] = {}

    @property
    def route_continuation_signature(self) -> str:
        return self.continuation.signature

    @property
    def arrival_continuation_signature(self) -> str:
        # Deliberately return the same evaluator signature. There is no second
        # route or arrival continuation configuration in this design.
        return self.continuation.signature

    def _plan_state(self, state: Any) -> PlanState:
        remaining = _all_mask(self.n_tasks) & (~_visited_mask(state))
        return PlanState(
            int(state.node),
            float(state.time_s),
            float(state.energy_kj),
            int(remaining),
        )

    def _arrival(
        self, plan: PlanState, task: int
    ) -> tuple[float, float, float, float] | None:
        if not iteration_v2.visit_ok_v2(
            int(plan.node),
            float(plan.time_s),
            float(plan.energy_kj),
            int(task),
            self.mission.flight_time,
            self.mission.flight_energy_kj,
            self.deadline,
            self.pars,
        ):
            return None
        arrival_t, arrival_e, available_t, available_e = (
            iteration_v2.resource_limits_v2(
                int(plan.node),
                float(plan.time_s),
                float(plan.energy_kj),
                int(task),
                self.mission.flight_time,
                self.mission.flight_energy_kj,
                self.deadline,
                self.pars,
            )
        )
        if available_t <= 0.0 or available_e <= 0.0:
            return None
        return (
            float(arrival_t),
            float(arrival_e),
            float(available_t),
            float(available_e),
        )

    def _prices(
        self, task: int, arrival_t: float, arrival_e: float, remaining: int
    ) -> tuple[float, float]:
        if self.policy.action_rule != "mode_separated" or int(remaining) == 0:
            return 0.0, 0.0
        cache_key = (
            int(task),
            float(arrival_t),
            float(arrival_e),
            int(remaining),
        )
        cached = self._price_cache.get(cache_key)
        if cached is not None:
            self.stats["dual_price_cache_hits"] += 1
            return cached
        if self.policy.continuation.analytic_mass_calibration:
            price_t, price_e, _ = self._analytic_prices(
                int(task),
                float(arrival_t),
                float(arrival_e),
                int(remaining),
            )
        else:
            price_t, price_e, _ = root_calibrated_core.state_lp_dual_prices_v2(
                int(task),
                float(arrival_t),
                float(arrival_e),
                np.uint64(remaining),
                np.asarray(self.price_points, dtype=np.float64),
                np.asarray(self.price_probabilities, dtype=np.float64),
                self.mission.flight_time,
                self.mission.flight_energy_kj,
                self.weights,
                self.deadline,
                self.pars,
                self.aa,
                self.bb,
                self.dd,
                self.ee,
            )
        self.stats["dual_price_calls"] += 1
        result = max(0.0, float(price_t)), max(0.0, float(price_e))
        self._price_cache[cache_key] = result
        return result

    def _analytic_prices(
        self, node: int, time_s: float, energy_kj: float, remaining: int
    ) -> tuple[float, float, float]:
        """Solve the two resource LP with analytic completion masses."""

        values = np.zeros(self.n_tasks, dtype=np.float64)
        time_cost = np.zeros(self.n_tasks, dtype=np.float64)
        energy_cost = np.zeros(self.n_tasks, dtype=np.float64)
        count = 0
        for candidate in range(1, self.n_tasks + 1):
            bit = 1 << (candidate - 1)
            if (int(remaining) & bit) == 0:
                continue
            probability, expected_time, expected_energy = (
                robust_core.calibrated_task_stats(
                    int(node),
                    float(time_s),
                    float(energy_kj),
                    int(candidate),
                    np.asarray(self.price_points, dtype=np.float64),
                    np.asarray(self.price_probabilities, dtype=np.float64),
                    self.mu,
                    self.sigma,
                    self.mission.flight_time,
                    self.mission.flight_energy_kj,
                    self.deadline,
                    self.pars,
                    self.aa,
                    self.bb,
                    self.dd,
                    self.ee,
                    0,
                )
            )
            value = float(self.weights[candidate]) * float(probability)
            if value <= 1e-14:
                continue
            incoming_time = float(self.mission.flight_time[node, candidate])
            incoming_energy = float(
                self.mission.flight_energy_kj[node, candidate]
            )
            for predecessor in range(1, self.n_tasks + 1):
                if predecessor == candidate:
                    continue
                predecessor_bit = 1 << (predecessor - 1)
                if (int(remaining) & predecessor_bit) == 0:
                    continue
                incoming_time = min(
                    incoming_time,
                    float(self.mission.flight_time[predecessor, candidate]),
                )
                incoming_energy = min(
                    incoming_energy,
                    float(
                        self.mission.flight_energy_kj[predecessor, candidate]
                    ),
                )
            values[count] = value
            time_cost[count] = incoming_time + float(expected_time)
            energy_cost[count] = incoming_energy + float(expected_energy)
            count += 1

        minimum_return_time = float(self.mission.flight_time[node, 0])
        minimum_return_energy = float(self.mission.flight_energy_kj[node, 0])
        for candidate in range(1, self.n_tasks + 1):
            bit = 1 << (candidate - 1)
            if (int(remaining) & bit) == 0:
                continue
            minimum_return_time = min(
                minimum_return_time,
                float(self.mission.flight_time[candidate, 0]),
            )
            minimum_return_energy = min(
                minimum_return_energy,
                float(self.mission.flight_energy_kj[candidate, 0]),
            )
        capacity_time = float(self.pars[0]) - float(time_s) - minimum_return_time
        capacity_energy = (
            float(self.pars[1]) - float(energy_kj) - minimum_return_energy
        )
        result = root_calibrated_core._lp_dual_prices(
            float(capacity_time),
            float(capacity_energy),
            values,
            time_cost,
            energy_cost,
            int(count),
        )
        return float(result[0]), float(result[1]), float(result[2])

    @staticmethod
    def _local_weights(
        price_t: float, price_e: float, available_t: float, available_e: float
    ) -> tuple[float, float]:
        scaled_t = max(0.0, float(price_t)) * max(float(available_t), 0.0)
        scaled_e = max(0.0, float(price_e)) * max(float(available_e), 0.0)
        total = scaled_t + scaled_e
        if total <= 1e-14:
            return 1.0, 1.0
        return (
            max(EPS, 2.0 * scaled_t / total),
            max(EPS, 2.0 * scaled_e / total),
        )

    def _completion_actions(
        self,
        task: int,
        workload: float,
        available_t: float,
        available_e: float,
        price_t: float,
        price_e: float,
    ) -> tuple[ExecutionAction, ...]:
        if self.policy.action_rule == "mode_separated":
            time_weight, energy_weight = self._local_weights(
                price_t, price_e, available_t, available_e
            )
        else:
            time_weight, energy_weight = 1.0, 1.0

        actions: list[ExecutionAction] = []
        local_ok, local_t, local_e, _mode, local_f, _cost = (
            iteration_v2.execute_virtual_v2(
                float(workload),
                float(available_t),
                float(available_e),
                self.pars,
                self.aa[int(task)],
                self.bb[int(task)],
                self.dd[int(task)],
                self.ee[int(task)],
                1,
                float(time_weight),
                float(energy_weight),
                0.0,
            )
        )
        if local_ok:
            actions.append(
                ExecutionAction(
                    "local",
                    -1,
                    float(local_f),
                    float(local_t),
                    float(local_e),
                    float(local_t) / max(float(available_t), EPS)
                    + float(local_e) / max(float(available_e), EPS),
                )
            )

        for mec in range(self.aa.shape[1]):
            if float(self.aa[int(task), mec]) >= 1e90:
                continue
            service_t = float(
                self.aa[int(task), mec]
                + self.bb[int(task), mec] * float(workload)
            )
            service_e = float(
                self.dd[int(task), mec]
                + self.ee[int(task), mec] * float(workload)
            )
            if service_t > float(available_t) + 1e-9:
                continue
            if service_e > float(available_e) + 1e-9:
                continue
            actions.append(
                ExecutionAction(
                    "mec",
                    int(mec),
                    float(self.mission.cfg.mec_cpu_ghz[mec]),
                    service_t,
                    service_e,
                    service_t / max(float(available_t), EPS)
                    + service_e / max(float(available_e), EPS),
                )
            )

        if self.policy.action_rule == "mode_separated" and actions:
            selected = min(
                actions,
                key=lambda action: (
                    float(price_t) * action.service_time_s
                    + float(price_e) * action.service_energy_kj,
                    action.occupation,
                    action.service_time_s,
                    action.service_energy_kj,
                    action.mec,
                ),
            )
            return (selected,)
        return tuple(actions)

    @staticmethod
    def _skip_action() -> ExecutionAction:
        return ExecutionAction("skip", -1, 0.0, 0.0, 0.0, 0.0)

    def _successor(
        self,
        task: int,
        arrival_t: float,
        arrival_e: float,
        remaining: int,
        action: ExecutionAction,
    ) -> PlanState:
        return PlanState(
            int(task),
            float(arrival_t) + float(action.service_time_s),
            float(arrival_e) + float(action.service_energy_kj),
            int(remaining),
        )

    def _action_value(
        self,
        task: int,
        arrival_t: float,
        arrival_e: float,
        remaining: int,
        action: ExecutionAction,
    ) -> float:
        self.stats["action_value_calls"] += 1
        successor = self._successor(
            task, arrival_t, arrival_e, remaining, action
        )
        reward = immediate_reward(float(self.weights[int(task)]), action)
        return self.continuation.value(successor, reward)

    def _best_action(
        self,
        task: int,
        workload: float,
        arrival: tuple[float, float, float, float],
        remaining: int,
        price_t: float,
        price_e: float,
    ) -> tuple[ExecutionAction, float]:
        arrival_t, arrival_e, available_t, available_e = arrival
        completions = self._completion_actions(
            task,
            workload,
            available_t,
            available_e,
            price_t,
            price_e,
        )
        actions = (self._skip_action(), *completions)
        scored = [
            (
                self._action_value(
                    task, arrival_t, arrival_e, remaining, action
                ),
                action,
            )
            for action in actions
        ]
        value, selected = max(
            scored,
            key=lambda item: (
                item[0],
                1 if item[1].mode != "skip" else 0,
                -item[1].occupation,
                -item[1].service_time_s,
                -item[1].service_energy_kj,
                -item[1].mec,
            ),
        )
        return selected, float(value)

    def select_task(self, state: Any) -> tuple[int, dict[str, Any]]:
        """Choose the next task without reading any realized hidden workload."""

        self.stats["route_calls"] += 1
        plan = self._plan_state(state)
        if plan.remaining_mask == 0:
            return 0, {}
        best_task = 0
        best_value = 0.0
        candidate_values: dict[str, float] = {}
        candidates = list(self.continuation.candidate_tasks(plan))
        anchor_task = 0
        anchor_score = 0.0
        if self.policy.continuation.anchor_rule == "conditioned":
            anchor_task, anchor_score, _elapsed = iteration_choice(
                self.bundle, state, visited_mask(state)
            )
            self.stats["anchor_calls"] += 1
            if int(anchor_task) > 0 and int(anchor_task) not in candidates:
                candidates.append(int(anchor_task))
        for task in candidates:
            bit = 1 << (task - 1)
            if (plan.remaining_mask & bit) == 0:
                continue
            arrival = self._arrival(plan, task)
            if arrival is None:
                continue
            remaining = plan.remaining_mask & (~bit)
            price_t, price_e = self._prices(
                task, arrival[0], arrival[1], remaining
            )
            if self.policy.continuation.analytic_mass_calibration:
                capacity = float(
                    root_calibrated_core._maximum_capacity(
                        int(task),
                        float(arrival[2]),
                        float(arrival[3]),
                        self.pars,
                        self.aa,
                        self.bb,
                        self.dd,
                        self.ee,
                    )
                )
                feasible_mass = float(
                    root_calibrated_core._lognormal_cdf(
                        float(self.mu[task]),
                        float(self.sigma[task]),
                        float(capacity),
                    )
                )
                skip_value = self._action_value(
                    task,
                    arrival[0],
                    arrival[1],
                    remaining,
                    self._skip_action(),
                )
                feasible_value = 0.0
                if feasible_mass > 1e-14:
                    for branch, probability in enumerate(
                        self.continuation.probabilities
                    ):
                        quantile = float(
                            np.clip(
                                feasible_mass
                                * self.continuation.conditional_uniform[branch],
                                1e-14,
                                1.0 - 1e-14,
                            )
                        )
                        workload = float(
                            np.exp(
                                float(self.mu[task])
                                + float(self.sigma[task]) * float(ndtri(quantile))
                            )
                        )
                        workload = min(workload, capacity)
                        _action, branch_value = self._best_action(
                            task,
                            workload,
                            arrival,
                            remaining,
                            price_t,
                            price_e,
                        )
                        feasible_value += float(probability) * float(branch_value)
                expected_value = (
                    feasible_mass * feasible_value
                    + (1.0 - feasible_mass) * float(skip_value)
                )
            else:
                expected_value = 0.0
                for branch, probability in enumerate(
                    self.continuation.probabilities
                ):
                    workload = float(self.continuation.points[task, branch])
                    _action, branch_value = self._best_action(
                        task,
                        workload,
                        arrival,
                        remaining,
                        price_t,
                        price_e,
                    )
                    expected_value += float(probability) * float(branch_value)
            candidate_values[str(task)] = float(expected_value)
            if route_choice_is_better(
                expected_value, task, best_value, best_task
            ):
                best_task = int(task)
                best_value = float(expected_value)
        unconstrained_task = int(best_task)
        unconstrained_value = float(best_value)
        if int(anchor_task) > 0 and str(anchor_task) in candidate_values:
            anchor_value = float(candidate_values[str(anchor_task)])
            required = anchor_value + float(
                self.policy.continuation.improvement_margin
            )
            if (
                int(best_task) != int(anchor_task)
                and float(best_value) < required + EPS
            ):
                best_task = int(anchor_task)
                best_value = anchor_value
                self.stats["anchor_retained"] += 1
            elif int(best_task) != int(anchor_task):
                self.stats["anchor_replaced"] += 1
            else:
                self.stats["anchor_retained"] += 1
        return best_task, {
            "selection_mode": (
                "rollout_policy_improvement"
                if self.policy.continuation.evaluator == "executable_rollout"
                else "consistent_acar"
            ),
            "candidate_values": candidate_values,
            "selected_value": float(best_value),
            "unconstrained_task": int(unconstrained_task),
            "unconstrained_value": float(unconstrained_value),
            "anchor_task": int(anchor_task),
            "anchor_native_score": float(anchor_score),
            "improvement_margin": float(
                self.policy.continuation.improvement_margin
            ),
            "continuation_signature": self.continuation.signature,
            "guidance_rule": self.policy.continuation.guidance.rule,
            "guidance_weight": float(
                self.policy.continuation.guidance.effective_weight
            ),
            "continuation_evaluator": self.policy.continuation.evaluator,
        }

    def choose_actual_action(
        self, state: Any, task: int, workload: float
    ) -> ExecutionAction:
        """Choose recourse using the same evaluator used by ``select_task``."""

        self.stats["arrival_calls"] += 1
        plan = self._plan_state(state)
        arrival = self._arrival(plan, int(task))
        if arrival is None:
            return self._skip_action()
        remaining = plan.remaining_mask & (~(1 << (int(task) - 1)))
        price_t, price_e = self._prices(
            int(task), arrival[0], arrival[1], remaining
        )
        selected, _value = self._best_action(
            int(task),
            float(workload),
            arrival,
            remaining,
            price_t,
            price_e,
        )
        return selected


def _apply_action(
    mission: Any, state: Any, task: int, action: ExecutionAction
) -> None:
    old_node = int(state.node)
    state.time_s += float(mission.flight_time[old_node, int(task)])
    state.energy_kj += float(mission.flight_energy_kj[old_node, int(task)])
    state.node = int(task)
    state.visited.add(int(task))
    state.route.append(int(task))
    if action.mode == "skip":
        state.skipped += 1
        return
    state.time_s += float(action.service_time_s)
    state.energy_kj += float(action.service_energy_kj)
    state.completed.add(int(task))
    if action.mode == "local":
        state.local_actions += 1
    elif action.mode == "mec":
        state.mec_actions += 1


def run_unified_mission(
    bundle: Any, policy: PolicySpec, *, record_trace: bool = False
) -> dict[str, Any]:
    """Run one mission while enforcing the reveal-at-arrival information order."""

    mission = bundle.iteration
    planner = ConsistentACARPlanner(bundle, policy)
    state = iteration_sim.State()
    trace: list[dict[str, Any]] = []
    planner_seconds = 0.0
    started = perf_counter()
    for epoch in range(int(mission.cfg.n_tasks)):
        planning_started = perf_counter()
        task, detail = planner.select_task(state)
        planner_seconds += perf_counter() - planning_started
        if int(task) == 0:
            break

        # The realized workload is read only after the route decision.
        workload = float(mission.tasks[int(task) - 1].workload_gcy)
        recourse_started = perf_counter()
        action = planner.choose_actual_action(state, int(task), workload)
        planner_seconds += perf_counter() - recourse_started
        _apply_action(mission, state, int(task), action)
        if record_trace:
            trace.append(
                {
                    "epoch": int(epoch),
                    "task": int(task),
                    "revealed_workload_gcy": workload,
                    "mode": action.mode,
                    "mec": int(action.mec),
                    "frequency_ghz": float(action.frequency_ghz),
                    "time_s": float(state.time_s),
                    "energy_kj": float(state.energy_kj),
                    "route_continuation_signature": (
                        planner.route_continuation_signature
                    ),
                    "arrival_continuation_signature": (
                        planner.arrival_continuation_signature
                    ),
                    **detail,
                }
            )

    iteration_sim.finish_return(mission, state)
    runtime = perf_counter() - started
    total_weight = float(sum(task.weight for task in mission.tasks))
    completed_weight = float(
        sum(mission.tasks[index - 1].weight for index in state.completed)
    )
    residual = max(
        0.0,
        float(state.time_s) - float(mission.cfg.t_max),
        float(state.energy_kj) - float(mission.cfg.e_max_kj),
    )
    stats = dict(planner.stats)
    stats["continuation_calls"] = int(planner.continuation.calls)
    stats["continuation_cache_hits"] = int(planner.continuation.cache_hits)
    stats["continuation_expansions"] = int(planner.continuation.expansions)
    return {
        "method_key": policy.method_key,
        "method": policy.label,
        "seed": int(mission.seed),
        "mcr": completed_weight / max(total_weight, EPS),
        "safe_mcr": (
            completed_weight / max(total_weight, EPS)
            if residual <= 1e-7
            else 0.0
        ),
        "weighted_completed": completed_weight,
        "total_weight": total_weight,
        "completed_tasks": int(len(state.completed)),
        "visited_tasks": int(len(state.visited)),
        "skipped_tasks": int(state.skipped),
        "local_actions": int(state.local_actions),
        "mec_actions": int(state.mec_actions),
        "final_time_s": float(state.time_s),
        "final_energy_kj": float(state.energy_kj),
        "return_success": int(residual <= 1e-7),
        "max_constraint_residual": float(residual),
        "runtime_ms": float(runtime * 1000.0),
        "planner_runtime_ms": float(planner_seconds * 1000.0),
        "route": "0-" + "-".join(str(task) for task in state.route) + "-0",
        "continuation_signature": planner.continuation.signature,
        "route_arrival_signature_match": int(
            planner.route_continuation_signature
            == planner.arrival_continuation_signature
        ),
        "guidance_rule": policy.continuation.guidance.rule,
        "guidance_weight": float(
            policy.continuation.guidance.effective_weight
        ),
        "cache_expected_stats": int(
            policy.continuation.cache_expected_stats
        ),
        "planner_stats": json.dumps(stats, sort_keys=True, separators=(",", ":")),
        "trace": json.dumps(trace, separators=(",", ":")) if record_trace else "",
    }
