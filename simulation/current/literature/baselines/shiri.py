"""Shiri et al. (2024) iterative optimization, adapted to the UAV model.

Core: predicted deterministic prize-routing MILP, observation on arrival,
residual MILP, and adoption only when its predicted prize is at least the
retained route's prize (Procedure 2).  This module never receives actual
unvisited workloads.  Fixed time limits make this an incumbent-based IO
adaptation; no source competitive guarantee or exact-optimality claim applies.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import hashlib
import json
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

VERSION = "shiri-io-uav-v2"
_INITIAL_CACHE: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class RoutingResult:
    route: tuple[int, ...]
    reward: float
    status: int
    solver_reward: float | None
    solver_gap: float | None
    wall_s: float
    source: str
    candidate_count: int


class ShiriIOPolicy:
    def __init__(
        self, env: Any, seed: int, *, initial_time_limit: float = 5.0,
        online_time_limit: float = 0.25, cache_initial: bool = True,
    ) -> None:
        if initial_time_limit <= 0 or online_time_limit <= 0:
            raise ValueError("MILP time limits must be positive")
        self.env, self.seed = env, int(seed)
        self.initial_time_limit = float(initial_time_limit)
        self.online_time_limit = float(online_time_limit)
        self.cache_initial = bool(cache_initial)
        self.route: list[int] = []
        self.initialized = False
        self.decision_log: list[dict[str, Any]] = []
        self.diagnostics: dict[str, Any] = {
            "method": "Shiri-IO-adapted", "version": VERSION,
            "source": "Shiri et al. (2024), doi:10.1016/j.trb.2024.102984, Procedure 2",
            "prediction": "prior mean workload; observed current workload exact",
            "resource_mapping": "common executor evaluated at direct arrival from residual origin; frozen within each MILP",
            "initial_time_limit_s": self.initial_time_limit,
            "online_time_limit_s": self.online_time_limit,
            "mip_relative_gap_target": 0.0,
            "milp_solver": "scipy.optimize.milp / HiGHS, one thread",
            "decisions": 0, "arrival_reoptimizations": 0,
            "accepted_rivals": 0, "retained_incumbents": 0,
            "proactive_skips": 0, "solver_calls": 0,
            "solver_time_limits": 0, "solver_feasible_results": 0,
            "milp_wall_s": 0.0, "initial_cache_hit": False,
        }

    def _predictions(self, state: Any, observed: dict[int, float]) -> dict[int, tuple[float, float]]:
        """Map public mean/current-observed workloads to fixed node forecasts."""
        result = {}
        for task in self.env.tasks(state):
            workload = observed.get(int(task), float(self.env.mean_workloads[task]))
            transition = self.env.step(state, int(task), float(workload))
            if transition.completion_feasible:
                result[int(task)] = (float(transition.service_time_s), float(transition.service_energy_kj))
        return result

    def _route_feasible(
        self, state: Any, route: tuple[int, ...] | list[int],
        costs: dict[int, tuple[float, float]], current: int | None,
    ) -> bool:
        if len(set(route)) != len(route):
            return False
        if current is not None and current in route and route[0] != current:
            return False
        node, t, e = int(state.node), float(state.time_s), float(state.energy_kj)
        for task in route:
            if task not in costs:
                return False
            service_t, service_e = costs[task]
            t += float(self.env.flight_time[node, task]) + service_t
            e += float(self.env.flight_energy[node, task]) + service_e
            if t > float(self.env.deadline[task]) + 1e-7:
                return False
            node = task
        return (
            t + float(self.env.flight_time[node, 0]) <= self.env.t_max + 1e-7
            and e + float(self.env.flight_energy[node, 0]) <= self.env.e_max + 1e-7
        )

    def _feasible_incumbent(
        self, state: Any, costs: dict[int, tuple[float, float]],
        current: int | None, old_route: list[int],
    ) -> tuple[int, ...]:
        """A deterministic feasible fallback; not a replacement for the MILP.

        SciPy exposes no MIP-start argument. Retain any feasible old route and
        also build a cheapest-insertion route, then compare these with the
        actual solver incumbent. Every residual optimization still calls MILP.
        """
        route: list[int] = []
        remaining = set(costs)
        ht = max(self.env.t_max - state.time_s, 1e-12)
        he = max(self.env.e_max - state.energy_kj, 1e-12)
        while remaining:
            best_key = None
            best_route = None
            best_task = None
            for task in sorted(remaining):
                for position in range(len(route) + 1):
                    trial = route[:position] + [task] + route[position:]
                    if not self._route_feasible(state, trial, costs, current):
                        continue
                    left = state.node if position == 0 else route[position - 1]
                    right = 0 if position == len(route) else route[position]
                    dt = (self.env.flight_time[left, task] + self.env.flight_time[task, right]
                          - self.env.flight_time[left, right] + costs[task][0])
                    de = (self.env.flight_energy[left, task] + self.env.flight_energy[task, right]
                          - self.env.flight_energy[left, right] + costs[task][1])
                    key = (float(self.env.weights[task]) / max(dt / ht + de / he, 1e-12), -task, -position)
                    if best_key is None or key > best_key:
                        best_key, best_route, best_task = key, trial, task
            if best_route is None:
                break
            route = best_route
            remaining.remove(best_task)
        if self._route_feasible(state, old_route, costs, current):
            if self._reward(old_route) >= self._reward(route):
                route = list(old_route)
        return tuple(route)

    def _reward(self, route: tuple[int, ...] | list[int]) -> float:
        return float(sum(self.env.weights[task] for task in route))

    def _solve(
        self, state: Any, observed: dict[int, float], current: int | None,
        old_route: list[int], time_limit: float,
    ) -> RoutingResult:
        started = perf_counter()
        costs = self._predictions(state, observed)
        nodes = tuple(sorted(costs))
        fallback = self._feasible_incumbent(state, costs, current, old_route)
        k = len(nodes)
        if k == 0:
            return RoutingResult((), 0.0, 0, 0.0, 0.0, perf_counter() - started, "empty", 0)
        # Local node ids 0..k-1 are tasks; k is current physical origin and
        # k+1 is the terminal depot. If current task is retained, it must be
        # completed immediately, before departing; revisits are forbidden.
        start, goal = k, k + 1
        current_index = nodes.index(current) if current in nodes else None
        horizon = max(float(self.env.t_max - state.time_s), 1e-12)
        capacity = max(float(self.env.e_max - state.energy_kj), 1e-12)
        service_t = np.asarray([costs[i][0] / horizon for i in nodes])
        service_e = np.asarray([costs[i][1] / capacity for i in nodes])
        direct_t = np.asarray([self.env.flight_time[state.node, i] / horizon for i in nodes])
        direct_e = np.asarray([self.env.flight_energy[state.node, i] / capacity for i in nodes])
        return_t = np.asarray([self.env.flight_time[i, 0] / horizon for i in nodes])
        return_e = np.asarray([self.env.flight_energy[i, 0] / capacity for i in nodes])
        completion_deadlines = np.asarray([
            min(1.0 - return_t[i], float(self.env.deadline[task] - state.time_s) / horizon)
            for i, task in enumerate(nodes)
        ])
        arcs = [(start, goal)]
        arcs += [(start, j) for j in range(k)]
        arcs += [(i, goal) for i in range(k)]
        for i in range(k):
            for j in range(k):
                if i == j or j == current_index:
                    continue
                edge_t = self.env.flight_time[nodes[i], nodes[j]] / horizon
                edge_e = self.env.flight_energy[nodes[i], nodes[j]] / capacity
                # Triangle-inequality lower bounds: no feasible route can
                # contain an arc failing these two-node checks.
                if direct_t[i] + service_t[i] + edge_t + service_t[j] > completion_deadlines[j] + 1e-12:
                    continue
                if direct_e[i] + service_e[i] + edge_e + service_e[j] + return_e[j] > 1.0 + 1e-12:
                    continue
                arcs.append((i, j))
        a = len(arcs)
        y_start, time_start = a, a + k
        nvars = a + 2 * k
        travel_t, travel_e = [], []
        for i, j in arcs:
            source = int(state.node) if i == start else nodes[i]
            target = 0 if j == goal else nodes[j]
            travel_t.append(float(self.env.flight_time[source, target]) / horizon)
            travel_e.append(float(self.env.flight_energy[source, target]) / capacity)
        objective = np.zeros(nvars)
        objective[y_start:time_start] = -self.env.weights[list(nodes)]
        integrality = np.zeros(nvars, dtype=np.uint8)
        integrality[:time_start] = 1
        lower = np.zeros(nvars)
        upper = np.ones(nvars)
        rows, columns, data, lower_rows, upper_rows = [], [], [], [], []

        def add(entries, lo=-np.inf, hi=np.inf):
            row = len(lower_rows)
            for col, coefficient in entries:
                if coefficient:
                    rows.append(row); columns.append(col); data.append(float(coefficient))
            lower_rows.append(float(lo)); upper_rows.append(float(hi))

        add([(index, 1.0) for index, (i, _) in enumerate(arcs) if i == start], 1, 1)
        add([(index, 1.0) for index, (_, j) in enumerate(arcs) if j == goal], 1, 1)
        for i, task in enumerate(nodes):
            add([(index, 1.0) for index, (_, j) in enumerate(arcs) if j == i] + [(y_start + i, -1)], 0, 0)
            add([(index, 1.0) for index, (j, _) in enumerate(arcs) if j == i] + [(y_start + i, -1)], 0, 0)
            deadline = completion_deadlines[i]
            add([(time_start + i, 1.0), (y_start + i, service_t[i] - deadline)], hi=0)
            add([(time_start + i, 1.0), (y_start + i, -direct_t[i])], lo=0)
        # Resource rows also strengthen the arrival-time formulation.
        add(list(enumerate(travel_t)) + [(y_start + i, service_t[i]) for i in range(k)], hi=1)
        add(list(enumerate(travel_e)) + [(y_start + i, service_e[i]) for i in range(k)], hi=1)
        # A known feasible external incumbent is also a valid objective lower
        # bound; this strengthens search without dropping any better route.
        add([(y_start + i, self.env.weights[task]) for i, task in enumerate(nodes)], lo=self._reward(fallback))
        for index, (i, j) in enumerate(arcs):
            if i == start and j == goal:
                continue
            big_m = 3.0 + travel_t[index]
            if i == start:
                add([(time_start + j, 1), (index, -big_m)], lo=travel_t[index] - big_m)
            elif j == goal:
                add([(time_start + i, 1), (y_start + i, service_t[i]), (index, big_m)],
                    hi=1 - travel_t[index] + big_m)
            else:
                add([(time_start + j, 1), (time_start + i, -1),
                     (y_start + i, -service_t[i]), (index, -big_m)],
                    lo=travel_t[index] - big_m)
        matrix = coo_matrix((data, (rows, columns)), shape=(len(lower_rows), nvars)).tocsc()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected:.*")
            answer = milp(
                objective, integrality=integrality, bounds=Bounds(lower, upper),
                constraints=LinearConstraint(matrix, lower_rows, upper_rows),
                options={"time_limit": float(time_limit), "mip_rel_gap": 0.0,
                         "presolve": True, "threads": 1, "random_seed": 0},
            )
        self.diagnostics["solver_calls"] += 1
        self.diagnostics["solver_time_limits"] += int(answer.status == 1)
        chosen, source = fallback, "external_feasible_incumbent"
        solver_reward = None
        if answer.x is not None:
            adjacency = {i: j for (i, j), value in zip(arcs, answer.x[:a]) if value > 0.5}
            route, visited, current_vertex = [], set(), start
            for _ in range(k + 1):
                next_vertex = adjacency.get(current_vertex)
                if next_vertex == goal:
                    break
                if next_vertex is None or next_vertex in visited or next_vertex == start:
                    route = None
                    break
                visited.add(next_vertex)
                route.append(nodes[next_vertex])
                current_vertex = next_vertex
            else:
                route = None
            selected = {nodes[i] for i in range(k) if answer.x[y_start + i] > 0.5}
            if route is not None and set(route) == selected and self._route_feasible(state, route, costs, current):
                solver_reward = self._reward(route)
                self.diagnostics["solver_feasible_results"] += 1
                if solver_reward >= self._reward(chosen) - 1e-9:
                    chosen, source = tuple(route), "milp_incumbent"
        elapsed = perf_counter() - started
        self.diagnostics["milp_wall_s"] += elapsed
        gap = getattr(answer, "mip_gap", None)
        return RoutingResult(
            tuple(chosen), self._reward(chosen), int(answer.status), solver_reward,
            None if gap is None else float(gap), elapsed, source, k,
        )

    def _cache_key(self, state: Any) -> str:
        digest = hashlib.sha256()
        digest.update(VERSION.encode())
        digest.update(repr((self.initial_time_limit, state)).encode())
        for name in ("weights", "deadline", "mu", "sigma", "pars", "aa", "bb", "dd", "ee", "flight_time", "flight_energy"):
            digest.update(name.encode()); digest.update(np.ascontiguousarray(getattr(self.env, name)).tobytes())
        return digest.hexdigest()

    def _initialize(self, state: Any) -> None:
        key = self._cache_key(state)
        cache_dir = Path(__file__).resolve().parents[1] / "runtime/shiri_initial_cache"
        cache_path = cache_dir / f"{key}.json"
        cached = _INITIAL_CACHE.get(key) if self.cache_initial else None
        if cached is None and self.cache_initial and cache_path.exists():
            cached = json.loads(cache_path.read_text())
        if cached is not None:
            costs = self._predictions(state, {})
            if not self._route_feasible(state, cached["route"], costs, None):
                raise RuntimeError("cached initial Shiri route fails public-model validation")
            self.route = list(cached["route"])
            self.diagnostics["initial_cache_hit"] = True
            self.diagnostics["initial_optimization"] = cached
        else:
            result = self._solve(state, {}, None, [], self.initial_time_limit)
            cached = dict(result.__dict__)
            self.route = list(result.route)
            self.diagnostics["initial_optimization"] = cached
            if self.cache_initial:
                _INITIAL_CACHE[key] = cached
                cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(f".{self.seed}.tmp")
                temporary.write_text(json.dumps(cached, indent=2) + "\n")
                temporary.replace(cache_path)
        self.initialized = True

    def select_task(self, state: Any) -> int:
        self.diagnostics["decisions"] += 1
        if not self.initialized:
            self._initialize(state)
        self.route = [task for task in self.route if state.remaining_mask & (1 << (task - 1))]
        legal = set(self.env.tasks(state))
        while self.route and self.route[0] not in legal:
            self.route.pop(0)
        return int(self.route[0]) if self.route else 0

    def arrival_decision(self, state_before: Any, task: int, revealed_c: float) -> bool:
        """Reoptimize after observation and before servicing the arrived task.

        False requests a proactive skip: the runner must retain the flight cost
        and remove the visited task while spending no service resources.
        True requests the common executor, which may still skip infeasibility.
        """
        if not self.initialized:
            raise RuntimeError("select_task must initialize the policy before arrival")
        self.route = [i for i in self.route if state_before.remaining_mask & (1 << (i - 1))]
        old_reward = self._reward(self.route)
        arrived = type(state_before)(
            node=int(task),
            time_s=float(state_before.time_s + self.env.flight_time[state_before.node, task]),
            energy_kj=float(state_before.energy_kj + self.env.flight_energy[state_before.node, task]),
            remaining_mask=int(state_before.remaining_mask),
        )
        rival = self._solve(arrived, {int(task): float(revealed_c)}, int(task), list(self.route), self.online_time_limit)
        accepted = rival.reward >= old_reward - 1e-9
        if accepted:
            self.route = list(rival.route)
        complete = not accepted or int(task) in self.route
        self.route = [i for i in self.route if i != int(task)]
        self.diagnostics["arrival_reoptimizations"] += 1
        self.diagnostics["accepted_rivals"] += int(accepted)
        self.diagnostics["retained_incumbents"] += int(not accepted)
        self.diagnostics["proactive_skips"] += int(not complete)
        record = {
            "task": int(task), "incumbent_reward": old_reward,
            "rival_reward": rival.reward, "accepted": bool(accepted),
            "complete_requested": bool(complete), "rival": dict(rival.__dict__),
        }
        self.decision_log.append(record)
        self.diagnostics["last_arrival"] = record
        return bool(complete)


def make_policy(env: Any, seed: int, **kwargs: Any) -> ShiriIOPolicy:
    return ShiriIOPolicy(env, seed, **kwargs)
