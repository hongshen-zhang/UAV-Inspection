"""ADAPT-IACS known-prior UAV/MEC routing adaptation (Qian et al., 2025).

Source: IEEE IoTJ 12(10), 13988--14001, doi:10.1109/JIOT.2025.3525985.
Author source commit: e9967bf0d7a42c9051bdb1759bebeeac42d30681.
Preserves offline ACS, online multi-quantile IACS, inherited incumbent,
pheromone transition/updates, drop/add/2-opt, and safety/prize scoring.
Known workload quantiles replace the source's Bayesian unknown-power model.
Scalar ACS cost is normalized time+energy; exact route feasibility always
uses the unchanged public step_kernel. Theta is NOT a joint risk guarantee.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
from numba import njit
from scipy.special import ndtri
from environment import step_kernel

SOURCE_COMMIT = "e9967bf0d7a42c9051bdb1759bebeeac42d30681"
DEFAULT_LEVELS = (.75, .775, .8, .825, .85, .875, .9, .925, .95, .975, .999)


@njit(cache=True)
def _route_key(route, length):
    # Exact injective encoding for up to 60 nodes numbered 1..60.
    blocks = np.zeros(6, dtype=np.int64)
    for k in range(length):
        blocks[k // 10] |= np.int64(route[k]) << (6 * (k % 10))
    return (blocks[0], blocks[1], blocks[2], blocks[3], blocks[4], blocks[5])


@njit(cache=True)
def _feasible(route, length, origin, t0, e0, mask0, workloads, ft, fe,
              weights, deadlines, pars, aa, bb, dd, ee, cache, counts):
    counts[0] += 1
    key = _route_key(route, length)
    if key in cache:
        counts[1] += 1
        return cache[key]
    node, t, e, mask = origin, t0, e0, mask0
    for k in range(length):
        task = route[k]
        nt, ne, nm, reward, ok, mode, freq, st, se = step_kernel(
            node, t, e, mask, task, workloads[task], ft, fe, weights,
            deadlines, pars, aa, bb, dd, ee)
        counts[2] += 1
        if not ok:
            cache[key] = False
            return False
        node, t, e, mask = task, nt, ne, nm
    ok = t + ft[node, 0] <= pars[0] + 1e-9 and e + fe[node, 0] <= pars[1] + 1e-9
    cache[key] = ok
    return ok


@njit(cache=True)
def _fitness(route, length, origin, arcs, service, weights):
    cost, prize, node = 0.0, 0.0, origin
    for k in range(length):
        task = route[k]
        cost += arcs[node, task] + service[task]
        prize += weights[task]
        node = task
    return prize, cost + arcs[node, 0]


@njit(cache=True)
def _remove(route, length, position):
    for k in range(position, length - 1):
        route[k] = route[k + 1]
    route[length - 1] = 0
    return length - 1


@njit(cache=True)
def _insert(route, length, position, task):
    for k in range(length, position, -1):
        route[k] = route[k - 1]
    route[position] = task
    return length + 1


@njit(cache=True)
def _drop(route, length, origin, t0, e0, mask, workloads, arcs, service,
          ft, fe, weights, deadlines, pars, aa, bb, dd, ee, cache, counts):
    while length and not _feasible(route, length, origin, t0, e0, mask,
                                    workloads, ft, fe, weights, deadlines,
                                    pars, aa, bb, dd, ee, cache, counts):
        best_pos, best_ratio = 0, math.inf
        for k in range(length):
            before = origin if k == 0 else route[k - 1]
            after = 0 if k == length - 1 else route[k + 1]
            task = route[k]
            saved = arcs[before, task] + arcs[task, after] - arcs[before, after] + service[task]
            ratio = weights[task] / max(saved, 1e-12)
            if ratio < best_ratio:
                best_pos, best_ratio = k, ratio
        length = _remove(route, length, best_pos)
        counts[5] += 1
    return length


@njit(cache=True)
def _insertion_positions(route, length, origin, task, arcs):
    # Author C++ Add operator: all predecessor/successor route edges
    # incident to the three nearest route nodes. The published pseudocode's
    # adjacent-pair shortcut is deliberately not used by the author code.
    near = np.full(3, -1, dtype=np.int64)
    distance = np.full(3, math.inf)
    for p in range(length + 2):
        node = origin if p == 0 else 0 if p == length + 1 else route[p - 1]
        d = arcs[task, node]
        for j in range(3):
            if d < distance[j]:
                for q in range(2, j, -1):
                    near[q], distance[q] = near[q - 1], distance[q - 1]
                near[j], distance[j] = p, d
                break
    positions = np.zeros(length + 1, dtype=np.bool_)
    for p in near:
        if p >= 0:
            if p > 0:
                positions[p - 1] = True
            if p <= length:
                positions[p] = True
    return positions


@njit(cache=True)
def _best_add_position(route, length, task, positions, origin, t0, e0, mask,
                       workloads, arcs, service, ft, fe, weights, deadlines,
                       pars, aa, bb, dd, ee, cache, counts):
    best_position, smallest = -1, math.inf
    for p in positions:
        if p < 0:
            continue
        before = origin if p == 0 else route[p - 1]
        after = 0 if p == length else route[p]
        delta = arcs[before, task] + arcs[task, after] - arcs[before, after] + service[task]
        if delta >= smallest:
            continue
        _insert(route, length, p, task)
        ok = _feasible(route, length + 1, origin, t0, e0, mask, workloads,
                       ft, fe, weights, deadlines, pars, aa, bb, dd, ee,
                       cache, counts)
        _remove(route, length + 1, p)
        if ok:
            best_position, smallest = p, delta
    return best_position, smallest


@njit(cache=True)
def _add(route, length, allowed, origin, t0, e0, mask, workloads, arcs,
         service, ft, fe, weights, deadlines, pars, aa, bb, dd, ee, cache, counts):
    # Initialize one feasible cheapest insertion for each unvisited task.
    # As in author _iterativeAddNodes, later steps retain that location (if
    # unaffected), comparing only the edges around the newly inserted task.
    used = np.zeros(len(weights), dtype=np.bool_)
    for k in range(length):
        used[route[k]] = True
    candidate_positions = np.full(len(weights), -1, dtype=np.int64)
    candidate_costs = np.full(len(weights), math.inf)
    for task in allowed:
        if used[task]:
            continue
        selected = _insertion_positions(route, length, origin, task, arcs)
        positions = np.flatnonzero(selected)
        candidate_positions[task], candidate_costs[task] = _best_add_position(
            route, length, task, positions, origin, t0, e0, mask, workloads,
            arcs, service, ft, fe, weights, deadlines, pars, aa, bb, dd, ee,
            cache, counts)
    while length < len(allowed):
        best_task, best_ratio = 0, -1.0
        for task in allowed:
            if candidate_positions[task] < 0:
                continue
            ratio = weights[task] / max(candidate_costs[task], 1e-12)
            if ratio > best_ratio:
                best_task, best_ratio = task, ratio
        if best_task == 0:
            break
        best_pos = candidate_positions[best_task]
        length = _insert(route, length, best_pos, best_task)
        candidate_positions[best_task] = -1
        counts[6] += 1
        for task in allowed:
            previous_pos = candidate_positions[task]
            if previous_pos < 0:
                continue
            positions = np.full(3, -1, dtype=np.int64)
            if previous_pos != best_pos and previous_pos != best_pos + 1:
                positions[0] = previous_pos + (1 if best_pos < previous_pos else 0)
            positions[1], positions[2] = best_pos, best_pos + 1
            candidate_positions[task], candidate_costs[task] = _best_add_position(
                route, length, task, positions, origin, t0, e0, mask, workloads,
                arcs, service, ft, fe, weights, deadlines, pars, aa, bb, dd, ee,
                cache, counts)
    return length


@njit(cache=True)
def _two_opt(route, length, max_changes, origin, t0, e0, mask, workloads,
             arcs, ft, fe, weights, deadlines, pars, aa, bb, dd, ee, cache, counts):
    changed, changes = True, 0
    currently_feasible = _feasible(route, length, origin, t0, e0, mask,
                                   workloads, ft, fe, weights, deadlines,
                                   pars, aa, bb, dd, ee, cache, counts)
    while changed and changes < max_changes:
        changed = False
        for i in range(length - 1):
            for j in range(i + 1, length):
                before = origin if i == 0 else route[i - 1]
                after = 0 if j == length - 1 else route[j + 1]
                # Public fixed-speed flight matrices are symmetric.
                delta = arcs[before, route[j]] + arcs[route[i], after] - arcs[before, route[i]] - arcs[route[j], after]
                if delta >= -1e-12:
                    continue
                route[i:j + 1] = route[i:j + 1][::-1]
                new_feasible = _feasible(route, length, origin, t0, e0, mask,
                                         workloads, ft, fe, weights, deadlines,
                                         pars, aa, bb, dd, ee, cache, counts)
                if currently_feasible and not new_feasible:
                    route[i:j + 1] = route[i:j + 1][::-1]
                else:
                    currently_feasible = new_feasible
                    changed = True
                    changes += 1
                    counts[7] += 1
                if changes >= max_changes:
                    break
            if changes >= max_changes:
                break


@njit(cache=True)
def _solve(origin, t0, e0, mask, workloads, previous, ft, fe, weights,
           deadlines, pars, aa, bb, dd, ee, num_ants, max_iters, max_no_impr,
           max_2opt, q0, alpha, beta, rho, tolerance, random_seed, offline):
    np.random.seed(random_seed)
    n = len(weights) - 1
    counts = np.zeros(8, dtype=np.int64)
    arcs = ft / max(pars[0] - t0, 1e-12) + fe / max(pars[1] - e0, 1e-12)
    service = np.zeros(n + 1)
    allowed_buffer = np.zeros(n, dtype=np.int64)
    num_allowed = 0
    for task in range(1, n + 1):
        if not mask & (1 << (task - 1)):
            continue
        nt, ne, nm, reward, ok, mode, freq, st, se = step_kernel(
            origin, t0, e0, mask, task, workloads[task], ft, fe, weights,
            deadlines, pars, aa, bb, dd, ee)
        if ok:
            service[task] = st / max(pars[0] - t0, 1e-12) + se / max(pars[1] - e0, 1e-12)
            allowed_buffer[num_allowed] = task
            num_allowed += 1
    allowed = allowed_buffer[:num_allowed]
    best = np.zeros(n, dtype=np.int64)
    if num_allowed == 0:
        return best, 0, 0.0, arcs[origin, 0], counts
    cache = {(0, 0, 0, 0, 0, 0): True}
    is_allowed = np.zeros(n + 1, dtype=np.bool_)
    for task in allowed:
        is_allowed[task] = True
    length = 0
    if offline:
        used = np.zeros(n + 1, dtype=np.bool_)
        node = origin
        for k in range(num_allowed):
            nearest, distance = 0, math.inf
            for task in allowed:
                if not used[task] and ft[node, task] < distance:
                    nearest, distance = task, ft[node, task]
            best[length] = nearest
            length += 1
            if not _feasible(best, length, origin, t0, e0, mask, workloads,
                             ft, fe, weights, deadlines, pars, aa, bb, dd, ee,
                             cache, counts):
                length = _remove(best, length, length - 1)
                break
            node, used[nearest] = nearest, True
    else:
        for task in previous:
            if is_allowed[task]:
                best[length] = task
                length += 1
    if not offline:
        length = _drop(best, length, origin, t0, e0, mask, workloads, arcs, service,
                       ft, fe, weights, deadlines, pars, aa, bb, dd, ee, cache, counts)
        length = _add(best, length, allowed, origin, t0, e0, mask, workloads, arcs,
                      service, ft, fe, weights, deadlines, pars, aa, bb, dd, ee, cache, counts)
    best_prize, best_cost = _fitness(best, length, origin, arcs, service, weights)
    best_length = length
    if best_length == 0:
        return best, 0, best_prize, best_cost, counts
    tau0 = best_prize / max(best_cost, 1e-12) / (best_length + 1)
    pheromone = np.full((n + 1, n + 1), tau0)
    if offline:
        # The source uses NN solely to initialize pheromone; the first ant
        # generation establishes the offline global incumbent.
        best[:] = 0
        best_length, best_prize, best_cost = 0, -1.0, math.inf
    stale = 0
    for iteration in range(max_iters):
        # Author C++ tests >, although prose says maximum of 25.
        if stale > max_no_impr:
            break
        counts[4] += 1
        local_best = np.zeros(n, dtype=np.int64)
        local_length, local_prize, local_cost = 0, -1.0, math.inf
        starts = np.empty(num_ants, dtype=np.int64)
        if num_ants >= num_allowed:
            for k in range(num_allowed):
                starts[k] = allowed[k]
            for k in range(num_allowed, num_ants):
                starts[k] = allowed[np.random.randint(num_allowed)]
        else:
            starts[:] = np.random.permutation(allowed)[:num_ants]
        for ant in range(num_ants):
            route = np.zeros(n, dtype=np.int64)
            used = np.zeros(n + 1, dtype=np.bool_)
            node, t, e, current_mask, length = origin, t0, e0, mask, 0
            following = starts[ant]
            for k in range(num_allowed):
                if k > 0:
                    products = np.zeros(num_allowed)
                    total, largest, chosen_index = 0.0, -1.0, -1
                    for a in range(num_allowed):
                        task = allowed[a]
                        if not used[task]:
                            eta = weights[task] / max(arcs[node, task] + service[task], 1e-12)
                            products[a] = pheromone[node, task] * eta ** beta
                            total += products[a]
                            if products[a] > largest:
                                largest, chosen_index = products[a], a
                    if chosen_index < 0:
                        break
                    if np.random.random() > q0:
                        threshold = np.random.random() * total
                        accum = 0.0
                        for a in range(num_allowed):
                            accum += products[a]
                            if products[a] > 0.0 and accum >= threshold:
                                chosen_index = a
                                break
                    following = allowed[chosen_index]
                    if pheromone[node, following] > tau0:
                        pheromone[node, following] -= rho * (pheromone[node, following] - tau0)
                route[length] = following
                length += 1
                used[following] = True
                nt, ne, nm, reward, ok, mode, freq, st, se = step_kernel(
                    node, t, e, current_mask, following, workloads[following],
                    ft, fe, weights, deadlines, pars, aa, bb, dd, ee)
                counts[2] += 1
                if not ok:
                    break
                node, t, e, current_mask = following, nt, ne, nm
            if length > 1 and max_2opt > 0:
                _two_opt(route, length, max_2opt, origin, t0, e0, mask,
                          workloads, arcs, ft, fe, weights, deadlines, pars,
                          aa, bb, dd, ee, cache, counts)
            length = _drop(route, length, origin, t0, e0, mask, workloads, arcs,
                           service, ft, fe, weights, deadlines, pars, aa, bb,
                           dd, ee, cache, counts)
            length = _add(route, length, allowed, origin, t0, e0, mask, workloads,
                          arcs, service, ft, fe, weights, deadlines, pars, aa,
                          bb, dd, ee, cache, counts)
            prize, cost = _fitness(route, length, origin, arcs, service, weights)
            counts[3] += 1
            if prize > local_prize + 1e-12 or (abs(prize - local_prize) <= 1e-12 and cost < local_cost):
                local_best[:] = route
                local_length, local_prize, local_cost = length, prize, cost
        improved = local_prize >= best_prize + tolerance or (abs(local_prize - best_prize) <= 1e-12 and local_cost <= best_cost - tolerance)
        if improved:
            best[:] = local_best
            best_length, best_prize, best_cost = local_length, local_prize, local_cost
            stale = 0
        else:
            stale += 1
        # Source evaporates active target-to-target arcs, then deposits on
        # every incumbent edge (current origin/end are masked depots).
        for left in allowed:
            for right in allowed:
                if left != right:
                    pheromone[left, right] *= (1.0 - alpha)
        node = origin
        deposit = alpha * best_prize / max(best_cost, 1e-12)
        for k in range(best_length + 1):
            task = best[k] if k < best_length else 0
            pheromone[node, task] += deposit
            node = task
    assert _feasible(best, best_length, origin, t0, e0, mask, workloads,
                     ft, fe, weights, deadlines, pars, aa, bb, dd, ee, cache, counts)
    return best, best_length, best_prize, best_cost, counts


class AdaptIACS:
    """Public-prior policy; no hidden realized workload table is accepted."""
    def __init__(self, env: Any, seed: int, *, ants: int = 40,
                 iterations: int = 250, no_improvement: int = 25,
                 two_opt: int = 100, q0: float = .9, alpha: float = .1,
                 beta: float = 2., rho: float = .1, tolerance: float = 1e-4,
                 belief_weight: float = .5, levels=DEFAULT_LEVELS,
                 offline_iterations: int = 500, offline_no_improvement: int = 250,
                 offline_two_opt: int = 150, offline_tolerance: float = .01):
        if not 1 <= env.n_tasks <= 60:
            raise ValueError("ADAPT exact route-cache encoding supports 1..60 tasks")
        if not np.allclose(env.flight_time, env.flight_time.T) or not np.allclose(env.flight_energy, env.flight_energy.T):
            raise ValueError("This adaptation requires the benchmark's symmetric flight matrices")
        self.env, self.seed = env, int(seed)
        self.levels = np.asarray(levels, dtype=np.float64)
        if len(self.levels) < 2 or np.any(np.diff(self.levels) <= 0) or self.levels[0] <= 0 or self.levels[-1] >= 1:
            raise ValueError("levels must contain ascending probabilities strictly between 0 and 1")
        if ants < 1 or iterations < 1 or no_improvement < 0 or two_opt < 0:
            raise ValueError("Invalid search effort parameters")
        self.settings = (int(ants), int(iterations), int(no_improvement), int(two_opt),
                         float(q0), float(alpha), float(beta), float(rho), float(tolerance))
        self.offline_settings = (int(ants), int(offline_iterations), int(offline_no_improvement),
                                 int(offline_two_opt), float(q0), float(alpha), float(beta),
                                 float(rho), float(offline_tolerance))
        self.belief_weight = float(belief_weight)
        self.rng = np.random.default_rng(np.random.SeedSequence([1302025, self.seed]))
        self.quantiles = np.exp(env.mu[None, :] + env.sigma[None, :] * ndtri(self.levels)[:, None])
        self.quantiles[:, 0] = 0.
        self.previous = np.empty(0, dtype=np.int64)
        self.decisions = 0
        self.diagnostics = {
            "source": "Qian et al. (2025), doi:10.1109/JIOT.2025.3525985",
            "source_commit": SOURCE_COMMIT,
            "adaptation": "known lognormal prior; two resource cost surrogate; exact shared executor route checks",
            "uncertainty_learning": "none: public priors are known; source Bayesian flight-power estimator is not transplanted",
            "theta_interpretation": "marginal workload quantile, not joint chance-constraint guarantee",
            "levels": self.levels.tolist(), "belief_weight": self.belief_weight,
            "ants": ants, "iterations": iterations, "no_improvement": no_improvement,
            "two_opt": two_opt, "q0": q0, "alpha": alpha, "beta": beta,
            "rho": rho, "tolerance": tolerance, "seed_namespace": 1302025,
            "offline_iterations": offline_iterations, "offline_no_improvement": offline_no_improvement,
            "offline_two_opt": offline_two_opt, "offline_tolerance": offline_tolerance,
            "inheritance": "repaired preceding selected route retained as incumbent; uniform pheromone prize/(cost*edges)",
            "phases": "offline known-mean ACS first action; multi-quantile IACS after every actual observation",
            "route_checks": 0, "route_cache_hits": 0, "planning_transitions": 0,
            "ant_constructions": 0, "search_iterations": 0, "drops": 0,
            "adds": 0, "two_opt_changes": 0, "decisions": 0,
        }

    def _search(self, state, workloads, offline):
        env = self.env
        route, length, prize, cost, counts = _solve(
            int(state.node), float(state.time_s), float(state.energy_kj),
            int(state.remaining_mask), workloads, self.previous,
            env.flight_time, env.flight_energy, env.weights, env.deadline,
            env.pars, env.aa, env.bb, env.dd, env.ee,
            *(self.offline_settings if offline else self.settings),
            int(self.rng.integers(0, 2**31 - 1)), bool(offline))
        names = ("route_checks", "route_cache_hits", "planning_transitions",
                 "ant_constructions", "search_iterations", "drops", "adds", "two_opt_changes")
        for name, count in zip(names, counts):
            self.diagnostics[name] += int(count)
        return route[:length].copy(), float(prize), float(cost)

    def select_task(self, state: Any) -> int:
        if not self.env.tasks(state):
            self.decisions += 1
            self.diagnostics['decisions'] = self.decisions
            self.diagnostics['last_candidates'] = []
            self.diagnostics['last_phase'] = 'stop'
            return 0
        if self.decisions == 0:
            route, prize, cost = self._search(state, self.env.mean_workloads, True)
            self.diagnostics["last_phase"] = "offline"
            self.diagnostics["last_candidates"] = [{"theta": None, "prize": prize, "cost": cost, "route": route.tolist()}]
        else:
            candidates = [self._search(state, work, False) for work in self.quantiles]
            prizes = np.asarray([entry[1] for entry in candidates])
            if float(np.max(prizes) - np.min(prizes)) <= 1e-12:
                chosen = len(candidates) - 1
                scores = self.levels.copy()
            else:
                scores = self.belief_weight * (self.levels - self.levels[0]) / (self.levels[-1] - self.levels[0]) + (1. - self.belief_weight) * (prizes - prizes.min()) / (prizes.max() - prizes.min())
                chosen = int(np.argmax(scores))
            route, prize, cost = candidates[chosen]
            self.diagnostics["last_phase"] = "online"
            self.diagnostics["last_selected_theta"] = float(self.levels[chosen])
            self.diagnostics["last_candidates"] = [
                {"theta": float(theta), "prize": item[1], "cost": item[2], "score": float(score), "route": item[0].tolist()}
                for theta, item, score in zip(self.levels, candidates, scores)]
        self.previous = route.copy()
        self.decisions += 1
        self.diagnostics["decisions"] = self.decisions
        return int(route[0]) if len(route) else 0


def make_policy(env: Any, seed: int, **kwargs: Any) -> AdaptIACS:
    return AdaptIACS(env, seed, **kwargs)
