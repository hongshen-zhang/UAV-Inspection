"""Original two-resource LP dual solver; function bodies are unchanged."""
from __future__ import annotations
import math
import numpy as np
from numba import njit
from . import physics as iteration_v2

@njit(cache=True)
def _dual_objective(y_t, y_e, cap_t, cap_e, values, time_cost, energy_cost, count):
    """Dual objective of the two resource fractional task relaxation."""

    result = cap_t * y_t + cap_e * y_e
    for index in range(count):
        residual = values[index] - time_cost[index] * y_t - energy_cost[index] * y_e
        if residual > 0.0:
            result += residual
    return result


@njit(cache=True)
def _lp_dual_prices(cap_t, cap_e, values, time_cost, energy_cost, count):
    """Return one deterministic optimal dual price pair and the LP value.

    The candidate vertices are identical to the exact two variable dual solver
    used by the existing LP guidance. If several vertices have the same dual
    value, the minimum capacity normalized price norm is selected. This is a
    deterministic secondary rule rather than a fitted coefficient.
    """

    if cap_t <= 0.0 or cap_e <= 0.0 or count == 0:
        return 0.0, 0.0, 0.0

    best_t = 0.0
    best_e = 0.0
    best_value = _dual_objective(
        best_t, best_e, cap_t, cap_e, values, time_cost, energy_cost, count
    )
    best_norm = 0.0

    for first in range(count):
        if values[first] <= 0.0:
            continue
        if time_cost[first] > 1e-12:
            candidate_t = values[first] / time_cost[first]
            candidate_e = 0.0
            value = _dual_objective(
                candidate_t,
                candidate_e,
                cap_t,
                cap_e,
                values,
                time_cost,
                energy_cost,
                count,
            )
            norm = (cap_t * candidate_t) ** 2 + (cap_e * candidate_e) ** 2
            if value < best_value - 1e-12 or (
                abs(value - best_value) <= 1e-12 and norm < best_norm - 1e-12
            ):
                best_t = candidate_t
                best_e = candidate_e
                best_value = value
                best_norm = norm
        if energy_cost[first] > 1e-12:
            candidate_t = 0.0
            candidate_e = values[first] / energy_cost[first]
            value = _dual_objective(
                candidate_t,
                candidate_e,
                cap_t,
                cap_e,
                values,
                time_cost,
                energy_cost,
                count,
            )
            norm = (cap_t * candidate_t) ** 2 + (cap_e * candidate_e) ** 2
            if value < best_value - 1e-12 or (
                abs(value - best_value) <= 1e-12 and norm < best_norm - 1e-12
            ):
                best_t = candidate_t
                best_e = candidate_e
                best_value = value
                best_norm = norm

    for first in range(count):
        for second in range(first + 1, count):
            determinant = (
                time_cost[first] * energy_cost[second]
                - time_cost[second] * energy_cost[first]
            )
            if abs(determinant) < 1e-12:
                continue
            candidate_t = (
                values[first] * energy_cost[second]
                - values[second] * energy_cost[first]
            ) / determinant
            candidate_e = (
                time_cost[first] * values[second]
                - time_cost[second] * values[first]
            ) / determinant
            if candidate_t < -1e-12 or candidate_e < -1e-12:
                continue
            candidate_t = max(0.0, candidate_t)
            candidate_e = max(0.0, candidate_e)
            value = _dual_objective(
                candidate_t,
                candidate_e,
                cap_t,
                cap_e,
                values,
                time_cost,
                energy_cost,
                count,
            )
            norm = (cap_t * candidate_t) ** 2 + (cap_e * candidate_e) ** 2
            if value < best_value - 1e-12 or (
                abs(value - best_value) <= 1e-12 and norm < best_norm - 1e-12
            ):
                best_t = candidate_t
                best_e = candidate_e
                best_value = value
                best_norm = norm

    return best_t, best_e, max(0.0, best_value)


@njit(cache=True)
def state_lp_dual_prices_v2(
    node,
    t,
    e,
    mask,
    points,
    probs,
    ft,
    fe,
    weights,
    deadline,
    pars,
    aa,
    bb,
    dd,
    ee,
):
    """Compute LP shadow prices from unrevealed task distributions only."""

    n = len(weights) - 1
    values = np.zeros(n)
    time_cost = np.zeros(n)
    energy_cost = np.zeros(n)
    count = 0
    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (mask & bit) == 0:
            continue
        probability, expected_time, expected_energy = iteration_v2.expected_task_stats_v2(
            node,
            t,
            e,
            task,
            points,
            probs,
            ft,
            fe,
            deadline,
            pars,
            aa,
            bb,
            dd,
            ee,
            0,
            1.0,
            1.0,
            0.0,
        )
        value = weights[task] * probability
        if value <= 1e-14:
            continue
        incoming_time = ft[node, task]
        incoming_energy = fe[node, task]
        for predecessor in range(1, n + 1):
            if predecessor == task:
                continue
            predecessor_bit = np.uint64(1) << np.uint64(predecessor - 1)
            if (mask & predecessor_bit) != 0:
                incoming_time = min(incoming_time, ft[predecessor, task])
                incoming_energy = min(incoming_energy, fe[predecessor, task])
        values[count] = value
        time_cost[count] = incoming_time + expected_time
        energy_cost[count] = incoming_energy + expected_energy
        count += 1

    minimum_return_time = ft[node, 0]
    minimum_return_energy = fe[node, 0]
    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (mask & bit) != 0:
            minimum_return_time = min(minimum_return_time, ft[task, 0])
            minimum_return_energy = min(minimum_return_energy, fe[task, 0])
    cap_t = pars[0] - t - minimum_return_time
    cap_e = pars[1] - e - minimum_return_energy
    return _lp_dual_prices(cap_t, cap_e, values, time_cost, energy_cost, count)

