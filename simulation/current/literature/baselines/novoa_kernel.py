"""Compiled full-tour evaluator using the exact shared physical transition."""
from __future__ import annotations

# Environment configures the frozen research imports and local cache first.
from environment import step_kernel
import numpy as np
from numba import njit


@njit(cache=True)
def rollout_values_kernel(
    node, time_s, energy_kj, remaining_mask, candidates, base_tour, positions,
    scenarios, flight_time, flight_energy, weights, deadline, pars, aa, bb, dd, ee,
):
    values = np.zeros(len(candidates), dtype=np.float64)
    transitions = 0
    n = len(base_tour)
    for candidate_index in range(len(candidates)):
        first = candidates[candidate_index]
        start = positions[first]
        total = 0.0
        for scenario in range(len(scenarios)):
            current_node = node
            current_t = time_s
            current_e = energy_kj
            current_mask = remaining_mask
            reward = 0.0
            for offset in range(n):
                task = base_tour[(start + offset) % n]
                if not current_mask & (1 << (task - 1)):
                    continue
                nt, ne, nm, prize, ok, mode, freq, service_t, service_e = step_kernel(
                    current_node, current_t, current_e, current_mask, task,
                    scenarios[scenario, task], flight_time, flight_energy, weights,
                    deadline, pars, aa, bb, dd, ee,
                )
                # A visit that is unsafe or already expired is passed over
                # without moving. A safe skipped task DOES move and is removed.
                if mode == -2:
                    continue
                current_node = task
                current_t, current_e, current_mask = nt, ne, nm
                reward += prize
                transitions += 1
            total += reward
        values[candidate_index] = total / len(scenarios)
    return values, transitions
