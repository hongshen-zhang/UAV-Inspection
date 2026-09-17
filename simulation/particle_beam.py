"""A bounded common-route beam retaining deterministic workload trajectories.

The beam optimizes a common route prefix across quadrature scenarios. Each
trajectory can bypass an infeasible visit, and only reads the workload of the
task it reaches. Forecasts do not read the mission's realized workloads.
An LP is a pruning guide only; returned values are executable-route rewards.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np
from numba import njit
from scipy.special import ndtri
from scipy.stats import qmc

from planner import ConsistentACARPlanner, PlanState, route_choice_is_better
from batch_adapter import iteration_v2, robust_core
from calibrated_beam_core import _remaining_potential


@njit(cache=True)
def particle_beam(node0, t0, e0, mask0, scenarios, points, probs, mu, sigma,
                  ft, fe, weights, deadline, pars, aa, bb, dd, ee,
                  width, depth, guide_weight, admission_factor):
    n = len(weights) - 1
    count_s = len(scenarios)
    nodes = np.full((width, count_s), node0, dtype=np.int64)
    times = np.full((width, count_s), t0)
    energies = np.full((width, count_s), e0)
    rewards = np.zeros((width, count_s))
    masks = np.full(width, mask0, dtype=np.uint64)
    first = np.zeros(width, np.int64)
    count = 1
    best_value = 0.0
    best_first = 0
    expansions = 0
    for level in range(min(depth, n)):
        cap = width * n
        cn = np.zeros((cap, count_s), np.int64)
        ct = np.zeros((cap, count_s))
        ce = np.zeros((cap, count_s))
        cr = np.zeros((cap, count_s))
        cm = np.zeros(cap, np.uint64)
        cf = np.zeros(cap, np.int64)
        last = np.zeros(cap, np.int64)
        score = np.full(cap, -1e100)
        value = np.zeros(cap)
        nc = 0
        for b in range(count):
            for task in range(1, n + 1):
                bit = np.uint64(1) << np.uint64(task - 1)
                if (masks[b] & bit) == 0:
                    continue
                gain = 0.0
                for s in range(count_s):
                    node = nodes[b, s]
                    t = times[b, s]
                    e = energies[b, s]
                    r = rewards[b, s]
                    if iteration_v2.visit_ok_v2(node, t, e, task, ft, fe, deadline, pars):
                        at, ae, dt, de = iteration_v2.resource_limits_v2(
                            node, t, e, task, ft, fe, deadline, pars)
                        node = task
                        t, e = at, ae
                        ok, st, se, mode, freq, cost = robust_core.execute_virtual(
                            scenarios[s, task], dt, de, pars, aa[task], bb[task],
                            dd[task], ee[task], 0)
                        # Optional admission uses only known priorities and the
                        # just-revealed service demand, never future workloads.
                        if ok and admission_factor > 0.0:
                            other_w = 0.0
                            other_cost = 0.0
                            for j in range(1, n + 1):
                                bj = np.uint64(1) << np.uint64(j - 1)
                                if j == task or (masks[b] & bj) == 0:
                                    continue
                                p, et, en = robust_core.calibrated_task_stats(
                                    node, t, e, j, points, probs, mu, sigma,
                                    ft, fe, deadline, pars, aa, bb, dd, ee, 0)
                                other_w += weights[j] * p
                                other_cost += (ft[node, j] + et) / max(pars[0] - t, 1e-12)
                                other_cost += (fe[node, j] + en) / max(pars[1] - e, 1e-12)
                            unit_price = other_w / max(other_cost, 1e-12)
                            service_cost = st / max(pars[0] - t, 1e-12) + se / max(pars[1] - e, 1e-12)
                            ok = weights[task] >= admission_factor * unit_price * service_cost
                        if ok:
                            t += st
                            e += se
                            r += weights[task]
                            gain += weights[task]
                    cn[nc, s], ct[nc, s], ce[nc, s], cr[nc, s] = node, t, e, r
                if gain <= 1e-12:
                    continue
                cm[nc] = masks[b] & ~bit
                cf[nc] = task if level == 0 else first[b]
                last[nc] = task
                value[nc] = np.mean(cr[nc])
                if value[nc] > best_value + 1e-12 or (
                    abs(value[nc] - best_value) <= 1e-12 and
                    best_first > 0 and cf[nc] < best_first):
                    best_value, best_first = value[nc], cf[nc]
                # A mean-state LP guides pruning, but does not contribute to
                # final reward. Completed and skipped trajectories remain
                # distinct in all subsequent executable route extensions.
                potential = 0.0
                if guide_weight > 0.0 and cm[nc] != 0:
                    potential = _remaining_potential(
                        task, np.mean(ct[nc]), np.mean(ce[nc]), cm[nc], points,
                        probs, mu, sigma, ft, fe, weights, deadline, pars,
                        aa, bb, dd, ee)
                score[nc] = value[nc] + guide_weight * potential
                nc += 1
                expansions += 1
        if nc == 0:
            break
        order = np.argsort(-score[:nc])
        chosen = np.zeros(nc, np.bool_)
        seen = np.zeros(n + 1, np.bool_)
        count = 0
        for pass_id in range(2):
            for k in order:
                if chosen[k] or (pass_id == 0 and seen[last[k]]):
                    continue
                chosen[k] = True
                seen[last[k]] = True
                nodes[count], times[count], energies[count], rewards[count] = cn[k], ct[k], ce[k], cr[k]
                masks[count], first[count] = cm[k], cf[k]
                count += 1
                if count == width:
                    break
            if count == width:
                break
    return best_first, best_value, expansions


@dataclass(frozen=True)
class ParticleSpec:
    scenarios: int = 32
    beam_width: int = 32
    depth: int = 12
    guide_weight: float = 0.5
    admission_factor: float = 0.0


class ParticleContinuation:
    def __init__(self, original, spec):
        self.base = original
        self.spec = spec
        self.points, self.probabilities = original.points, original.probabilities
        self.mu, self.sigma = original.mu, original.sigma
        self.conditional_uniform = original.conditional_uniform
        m = int(np.log2(spec.scenarios))
        if 2 ** m != spec.scenarios:
            raise ValueError('scenario count must be a power of two')
        # Center the deterministic Sobol dyadic cells. No mission seed or
        # realized workload enters construction of these integration nodes.
        u = qmc.Sobol(d=len(self.mu)-1, scramble=False).random_base2(m)
        z = ndtri(u + 0.5 / spec.scenarios)
        self.scenarios = np.zeros((spec.scenarios, len(self.mu)))
        self.scenarios[:, 1:] = np.exp(self.mu[1:] + self.sigma[1:] * z)
        self.calls = self.cache_hits = self.expansions = 0
        self.cache = {}
        self.signature = hashlib.sha256(json.dumps(spec.__dict__, sort_keys=True).encode()).hexdigest()[:16]

    def plan(self, state):
        key = (state.node, state.time_s, state.energy_kj, state.remaining_mask)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        b = self.base
        task, value, expanded = particle_beam(
            state.node, state.time_s, state.energy_kj, np.uint64(state.remaining_mask),
            self.scenarios, b.points, b.probabilities, b.mu, b.sigma,
            b.bundle.robust.flight_time, b.bundle.robust.flight_energy_kj,
            b.weights, b.deadline, b.pars, b.aa, b.bb, b.dd, b.ee,
            self.spec.beam_width, self.spec.depth, self.spec.guide_weight,
            self.spec.admission_factor)
        self.expansions += expanded
        self.cache[key] = (int(task), float(value))
        return self.cache[key]

    def value(self, state, immediate):
        self.calls += 1
        return immediate + self.plan(state)[1]


class ParticlePlanner(ConsistentACARPlanner):
    particle_spec = ParticleSpec()

    def __init__(self, bundle, policy):
        super().__init__(bundle, policy)
        self.continuation = ParticleContinuation(self.continuation, self.particle_spec)

    def select_task(self, state):
        self.stats['route_calls'] += 1
        task, value = self.continuation.plan(self._plan_state(state))
        return task, {'selection_mode': 'particle_beam', 'selected_value': value,
                      'continuation_signature': self.continuation.signature}
