"""One-step Monte Carlo rollout adapted from Novoa and Storer (2009).

Source: EJOR 196(2), 509--515, doi:10.1016/j.ejor.2008.03.023,
Sections 5.1, 5.2 and 5.4.  Preserved components are the NN + 2-opt
base tour, cyclic base-tour update, and sampled full continuation.
Reward maximization, UAV constraints and Local/MEC execution are adaptations.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def closed_tour_length(order: np.ndarray, distance: np.ndarray) -> float:
    """Length of depot -> task order -> depot; tasks are indexed from one."""
    if len(order) == 0:
        return 0.0
    previous = 0
    total = 0.0
    for task in order:
        total += float(distance[previous, int(task)])
        previous = int(task)
    return total + float(distance[previous, 0])


def build_base_tour(distance: np.ndarray) -> np.ndarray:
    """Deterministic nearest neighbor followed by first-improvement 2-opt.

    The matrix must be symmetric.  The depot is node zero.  Priorities and
    workloads are intentionally absent: this is the geometric n2 base tour.
    Flight-time distances give the same tour under the fixed UAV speed.
    """
    distance = np.asarray(distance, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("distance must contain finite nonnegative values")
    if not np.allclose(distance, distance.T, rtol=1e-10, atol=1e-10):
        raise ValueError("the geometric 2-opt base requires symmetric distances")
    unvisited = set(range(1, len(distance)))
    order: list[int] = []
    previous = 0
    while unvisited:
        following = min(unvisited, key=lambda task: (distance[previous, task], task))
        order.append(following)
        unvisited.remove(following)
        previous = following

    # Reverse segments between two nonadjacent tour edges.  Strict
    # improvement prevents cycling; scan order and restarting are fixed.
    changed = True
    while changed:
        changed = False
        for start in range(len(order) - 1):
            left = 0 if start == 0 else order[start - 1]
            first = order[start]
            for end in range(start + 1, len(order)):
                last = order[end]
                right = 0 if end == len(order) - 1 else order[end + 1]
                original = distance[left, first] + distance[last, right]
                swapped = distance[left, last] + distance[first, right]
                if swapped < original - 1e-12:
                    order[start : end + 1] = reversed(order[start : end + 1])
                    changed = True
                    break
            if changed:
                break
    return np.asarray(order, dtype=np.int64)


def cyclic_order(base_tour: np.ndarray, first: int, remaining_mask: int) -> np.ndarray:
    """Rotate at a candidate and delete already visited tasks."""
    indices = np.flatnonzero(np.asarray(base_tour) == int(first))
    if len(indices) != 1 or not (int(remaining_mask) & (1 << (int(first) - 1))):
        raise ValueError("first must be one unvisited task in the base tour")
    start = int(indices[0])
    rotated = np.concatenate((base_tour[start:], base_tour[:start]))
    return np.asarray(
        [task for task in rotated if int(remaining_mask) & (1 << (int(task) - 1))],
        dtype=np.int64,
    )


class NovoaRollout:
    """Full-candidate one-step rollout with a fixed geometric base policy.

    ``env`` contains public physical parameters and workload priors only.
    Actual workload realizations must never be attached to this policy.
    """

    def __init__(self, env: Any, seed: int, samples: int = 100) -> None:
        if int(samples) != samples or int(samples) < 1:
            raise ValueError("samples must be a positive integer")
        self.env = env
        self.samples = int(samples)
        self.seed = int(seed)
        self.rng = np.random.default_rng(np.random.SeedSequence([970029, self.seed]))
        self.base_tour = build_base_tour(env.flight_time)
        self.positions = np.zeros(len(self.base_tour) + 1, dtype=np.int64)
        for position, task in enumerate(self.base_tour):
            self.positions[int(task)] = position
        self._kernel = None
        if all(hasattr(env, name) for name in ("pars", "aa", "bb", "dd", "ee")):
            from .novoa_kernel import rollout_values_kernel
            self._kernel = rollout_values_kernel
        self.diagnostics: dict[str, Any] = {
            "source": "Novoa and Storer (2009), doi:10.1016/j.ejor.2008.03.023",
            "adaptation": "priority-reward UAV mission; common safe Local/MEC executor",
            "lookahead_steps": 1,
            "samples": self.samples,
            "planning_seed": self.seed,
            "seed_namespace": 970029,
            "base_tour": self.base_tour.tolist(),
            "base_tour_rule": "depot nearest-neighbor then deterministic first-improvement 2-opt",
            "base_tour_updates": "cyclic rotation; remove visited tasks; fixed throughout mission",
            "candidate_rule": "all feasible unvisited tasks plus zero-reward STOP",
            "decisions": 0,
            "candidate_evaluations": 0,
            "scenario_rollouts": 0,
            "sampled_transitions": 0,
            "evaluation_backend": "shared_numba_transition" if self._kernel else "reference_python",
        }

    def _visit_feasible(self, state: Any, task: int) -> bool:
        """Same exact visit, deadline and direct-return guard as env.tasks."""
        env = self.env
        arrival_t = float(state.time_s) + float(env.flight_time[int(state.node), task])
        arrival_e = float(state.energy_kj) + float(env.flight_energy[int(state.node), task])
        return bool(
            arrival_t + float(env.flight_time[task, 0]) <= float(env.t_max) + 1e-9
            and arrival_e + float(env.flight_energy[task, 0]) <= float(env.e_max) + 1e-9
            and float(env.deadline[task]) - arrival_t > 0.0
        )

    def _candidate_value(self, state: Any, first: int, scenarios: np.ndarray) -> float:
        order = cyclic_order(self.base_tour, first, int(state.remaining_mask))
        total = 0.0
        transitions = 0
        for workloads in scenarios:
            current = state  # State is frozen; env.step returns a new state.
            reward = 0.0
            for task_value in order:
                task = int(task_value)
                if not self._visit_feasible(current, task):
                    continue
                transition = self.env.step(current, task, float(workloads[task]))
                current = transition.state
                reward += float(transition.reward)
                transitions += 1
            total += reward
        self.diagnostics["sampled_transitions"] += transitions
        return total / float(len(scenarios))

    def select_task(self, state: Any) -> int:
        candidates = tuple(sorted(int(task) for task in self.env.tasks(state)))
        self.diagnostics["decisions"] += 1
        if not candidates:
            self.diagnostics["last_candidate_values"] = {}
            return 0
        scenarios = self.env.sample_workloads(self.rng, self.samples)
        best_task = 0
        best_value = 0.0
        values: dict[str, float] = {}
        if self._kernel is None:
            estimates = [self._candidate_value(state, task, scenarios) for task in candidates]
        else:
            env = self.env
            estimates, transitions = self._kernel(
                int(state.node), float(state.time_s), float(state.energy_kj),
                int(state.remaining_mask), np.asarray(candidates, dtype=np.int64),
                self.base_tour, self.positions, scenarios, env.flight_time,
                env.flight_energy, env.weights, env.deadline, env.pars,
                env.aa, env.bb, env.dd, env.ee,
            )
            self.diagnostics["sampled_transitions"] += int(transitions)
        for task, estimate in zip(candidates, estimates):
            value = float(estimate)
            values[str(task)] = value
            # Sorted candidates make equal positive scores deterministic.
            # A zero score never displaces the zero-reward STOP action.
            if value > best_value + 1e-12:
                best_task, best_value = task, value
        self.diagnostics["candidate_evaluations"] += len(candidates)
        self.diagnostics["scenario_rollouts"] += len(candidates) * self.samples
        self.diagnostics["last_candidate_values"] = values
        self.diagnostics["last_selected_value"] = best_value
        return best_task


def make_policy(env: Any, seed: int, samples: int = 100, **kwargs: Any) -> NovoaRollout:
    """Factory used by the common experiment runner."""
    if kwargs:
        raise TypeError(f"unknown Novoa rollout options: {sorted(kwargs)}")
    return NovoaRollout(env, seed, samples=samples)
