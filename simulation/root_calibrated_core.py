"""Root probability calibration for the single policy AR MTE controller.

The kernels in this module leave the deterministic quantile continuation of
AR MTE unchanged.  They only use the analytic LogNormal completion mass at the
candidate root task.  This separates a small, auditable correction of the
discontinuous immediate reward from the more expensive CDF calibration used
at every hypothetical continuation state by DAAR.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from vendor.iteration.common import mte_v2 as iteration_v2


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


@njit(cache=True)
def _dual_execution_weights(price_t, price_e, dt, de):
    """Map absolute LP prices to the normalized MTE Sub weight interface."""

    scaled_t = max(0.0, price_t) * max(dt, 0.0)
    scaled_e = max(0.0, price_e) * max(de, 0.0)
    total = scaled_t + scaled_e
    if total <= 1e-14:
        return 1.0, 1.0
    return (
        max(1e-12, 2.0 * scaled_t / total),
        max(1e-12, 2.0 * scaled_e / total),
    )


@njit(cache=True)
def _reduced_cost_action(
    workload,
    dt,
    de,
    reward,
    price_t,
    price_e,
    pars,
    aa,
    bb,
    dd,
    ee,
):
    """Solve the one step Lagrangian execution decision.

    Skip has reduced contribution zero. A successful action contributes its
    current reward minus the shadow priced time and energy occupation. The
    Local frequency is optimized analytically by ``execute_virtual_v2`` after
    mapping the absolute shadow prices to its normalized weight interface.
    """

    time_weight, energy_weight = _dual_execution_weights(
        price_t, price_e, dt, de
    )
    best_gain = 0.0
    best_cost = 0.0
    best_occupation = 0.0
    best_time = 0.0
    best_energy = 0.0
    best_mode = -1
    best_frequency = 0.0

    local_ok, local_t, local_e, _mode, local_f, _cost = (
        iteration_v2.execute_virtual_v2(
            workload,
            dt,
            de,
            pars,
            aa,
            bb,
            dd,
            ee,
            1,
            time_weight,
            energy_weight,
            0.0,
        )
    )
    if local_ok:
        dual_cost = max(0.0, price_t) * local_t + max(0.0, price_e) * local_e
        gain = reward - dual_cost
        occupation = local_t / max(dt, 1e-12) + local_e / max(de, 1e-12)
        if gain > best_gain + 1e-12:
            best_gain = gain
            best_cost = dual_cost
            best_occupation = occupation
            best_time = local_t
            best_energy = local_e
            best_mode = 0
            best_frequency = local_f

    for mec in range(len(aa)):
        if aa[mec] >= 1e90:
            continue
        service_time = aa[mec] + bb[mec] * workload
        service_energy = dd[mec] + ee[mec] * workload
        if service_time > dt + 1e-9 or service_energy > de + 1e-9:
            continue
        dual_cost = (
            max(0.0, price_t) * service_time
            + max(0.0, price_e) * service_energy
        )
        gain = reward - dual_cost
        occupation = (
            service_time / max(dt, 1e-12)
            + service_energy / max(de, 1e-12)
        )
        better = gain > best_gain + 1e-12
        if (
            not better
            and gain > 1e-12
            and abs(gain - best_gain) <= 1e-12
            and occupation < best_occupation - 1e-12
        ):
            better = True
        if better:
            best_gain = gain
            best_cost = dual_cost
            best_occupation = occupation
            best_time = service_time
            best_energy = service_energy
            best_mode = mec + 1
            best_frequency = 0.0

    return (
        best_mode >= 0,
        best_time,
        best_energy,
        best_mode,
        best_frequency,
        best_gain,
        best_cost,
    )


@njit(cache=True)
def _shadow_cost_action(
    workload,
    dt,
    de,
    price_t,
    price_e,
    pars,
    aa,
    bb,
    dd,
    ee,
):
    """Return the feasible successful action with minimum shadow cost."""

    time_weight, energy_weight = _dual_execution_weights(
        price_t, price_e, dt, de
    )
    best_cost = 1e100
    best_occupation = 1e100
    best_time = 0.0
    best_energy = 0.0
    best_mode = -1
    best_frequency = 0.0

    local_ok, local_t, local_e, _mode, local_f, _cost = (
        iteration_v2.execute_virtual_v2(
            workload,
            dt,
            de,
            pars,
            aa,
            bb,
            dd,
            ee,
            1,
            time_weight,
            energy_weight,
            0.0,
        )
    )
    if local_ok:
        dual_cost = max(0.0, price_t) * local_t + max(0.0, price_e) * local_e
        occupation = local_t / max(dt, 1e-12) + local_e / max(de, 1e-12)
        best_cost = dual_cost
        best_occupation = occupation
        best_time = local_t
        best_energy = local_e
        best_mode = 0
        best_frequency = local_f

    for mec in range(len(aa)):
        if aa[mec] >= 1e90:
            continue
        service_time = aa[mec] + bb[mec] * workload
        service_energy = dd[mec] + ee[mec] * workload
        if service_time > dt + 1e-9 or service_energy > de + 1e-9:
            continue
        dual_cost = (
            max(0.0, price_t) * service_time
            + max(0.0, price_e) * service_energy
        )
        occupation = (
            service_time / max(dt, 1e-12)
            + service_energy / max(de, 1e-12)
        )
        if dual_cost < best_cost - 1e-12 or (
            abs(dual_cost - best_cost) <= 1e-12
            and occupation < best_occupation - 1e-12
        ):
            best_cost = dual_cost
            best_occupation = occupation
            best_time = service_time
            best_energy = service_energy
            best_mode = mec + 1
            best_frequency = 0.0

    return (
        best_mode >= 0,
        best_time,
        best_energy,
        best_mode,
        best_frequency,
        best_cost,
    )


@njit(cache=True)
def _pareto_shadow_actions(
    workload,
    dt,
    de,
    price_t,
    price_e,
    pars,
    aa,
    bb,
    dd,
    ee,
):
    """Return the finite nondominated Local/MEC execution set.

    The set contains the original occupation-minimizing Local action, the
    shadow-price Local action, and every feasible MEC action. An action is
    removed only when another retained action uses no more service time and no
    more onboard energy, with at least one strict improvement. Equal resource
    pairs are deterministically deduplicated.
    """

    capacity = len(aa) + 2
    times = np.zeros(capacity)
    energies = np.zeros(capacity)
    modes = np.zeros(capacity, dtype=np.int64)
    frequencies = np.zeros(capacity)
    count = 0

    base_ok, base_t, base_e, _base_mode, base_f, _base_cost = (
        iteration_v2.execute_virtual_v2(
            workload,
            dt,
            de,
            pars,
            aa,
            bb,
            dd,
            ee,
            1,
            1.0,
            1.0,
            0.0,
        )
    )
    if base_ok:
        times[count] = base_t
        energies[count] = base_e
        modes[count] = 0
        frequencies[count] = base_f
        count += 1

    time_weight, energy_weight = _dual_execution_weights(
        price_t, price_e, dt, de
    )
    dual_ok, dual_t, dual_e, _dual_mode, dual_f, _dual_cost = (
        iteration_v2.execute_virtual_v2(
            workload,
            dt,
            de,
            pars,
            aa,
            bb,
            dd,
            ee,
            1,
            time_weight,
            energy_weight,
            0.0,
        )
    )
    if dual_ok:
        duplicate = False
        for index in range(count):
            if (
                abs(times[index] - dual_t) <= 1e-9
                and abs(energies[index] - dual_e) <= 1e-9
            ):
                duplicate = True
                break
        if not duplicate:
            times[count] = dual_t
            energies[count] = dual_e
            modes[count] = 0
            frequencies[count] = dual_f
            count += 1

    for mec in range(len(aa)):
        if aa[mec] >= 1e90:
            continue
        service_time = aa[mec] + bb[mec] * workload
        service_energy = dd[mec] + ee[mec] * workload
        if service_time > dt + 1e-9 or service_energy > de + 1e-9:
            continue
        duplicate = False
        for index in range(count):
            if (
                abs(times[index] - service_time) <= 1e-9
                and abs(energies[index] - service_energy) <= 1e-9
            ):
                duplicate = True
                break
        if duplicate:
            continue
        times[count] = service_time
        energies[count] = service_energy
        modes[count] = mec + 1
        frequencies[count] = 0.0
        count += 1

    keep = np.ones(count, dtype=np.uint8)
    for candidate in range(count):
        for challenger in range(count):
            if challenger == candidate:
                continue
            weak_time = times[challenger] <= times[candidate] + 1e-10
            weak_energy = energies[challenger] <= energies[candidate] + 1e-10
            strict = (
                times[challenger] < times[candidate] - 1e-10
                or energies[challenger] < energies[candidate] - 1e-10
            )
            if weak_time and weak_energy and strict:
                keep[candidate] = 0
                break

    frontier_times = np.zeros(capacity)
    frontier_energies = np.zeros(capacity)
    frontier_modes = np.zeros(capacity, dtype=np.int64)
    frontier_frequencies = np.zeros(capacity)
    frontier_count = 0
    for index in range(count):
        if keep[index] == 0:
            continue
        frontier_times[frontier_count] = times[index]
        frontier_energies[frontier_count] = energies[index]
        frontier_modes[frontier_count] = modes[index]
        frontier_frequencies[frontier_count] = frequencies[index]
        frontier_count += 1

    return (
        frontier_times,
        frontier_energies,
        frontier_modes,
        frontier_frequencies,
        frontier_count,
        count,
    )


@njit(cache=True)
def _local_capacity(dt, de, pars):
    """Maximum local workload from the finite analytic candidate set."""

    if dt <= 0.0 or de <= 0.0:
        return 0.0
    fmin = pars[2]
    fmax = pars[3]
    kappa = pars[4]
    hover = pars[5]
    candidates = np.zeros(4)
    count = 0
    candidates[count] = fmin
    count += 1
    candidates[count] = fmax
    count += 1
    if kappa > 0.0:
        candidates[count] = (hover / (2.0 * kappa)) ** (1.0 / 3.0)
        count += 1
        rhs = de / dt - hover
        if rhs > 0.0:
            candidates[count] = (rhs / kappa) ** (1.0 / 3.0)
            count += 1

    capacity = 0.0
    for index in range(count):
        frequency = min(fmax, max(fmin, candidates[index]))
        energy_per_workload = kappa * frequency * frequency + hover / frequency
        value = min(dt * frequency, de / max(energy_per_workload, 1e-15))
        if value > capacity:
            capacity = value
    return max(0.0, capacity)


@njit(cache=True)
def _maximum_capacity(task, dt, de, pars, aa, bb, dd, ee):
    """Maximum workload executable by Local or one available MEC server."""

    capacity = _local_capacity(dt, de, pars)
    for mec in range(aa.shape[1]):
        if aa[task, mec] >= 1e90:
            continue
        if bb[task, mec] > 0.0:
            time_capacity = (dt - aa[task, mec]) / bb[task, mec]
        else:
            time_capacity = 1e100
        if ee[task, mec] > 0.0:
            energy_capacity = (de - dd[task, mec]) / ee[task, mec]
        else:
            energy_capacity = 1e100
        mec_capacity = max(0.0, min(time_capacity, energy_capacity))
        if mec_capacity > capacity:
            capacity = mec_capacity
    return max(0.0, capacity)


@njit(cache=True)
def _lognormal_cdf(mu, sigma, threshold):
    if threshold <= 0.0:
        return 0.0
    if sigma <= 1e-12:
        return 1.0 if math.exp(mu) <= threshold else 0.0
    z = (math.log(threshold) - mu) / (sigma * math.sqrt(2.0))
    return min(1.0, max(0.0, 0.5 * (1.0 + math.erf(z))))


@njit(cache=True)
def _lognormal_ppf(mu, sigma, probability):
    """Deterministic inverse LogNormal CDF for conditional quadrature."""

    if sigma <= 1e-12:
        return math.exp(mu)
    target = min(1.0 - 1e-15, max(1e-15, probability))
    left = -12.0
    right = 12.0
    for _ in range(58):
        middle = 0.5 * (left + right)
        cdf = 0.5 * (1.0 + math.erf(middle / math.sqrt(2.0)))
        if cdf < target:
            left = middle
        else:
            right = middle
    return math.exp(mu + sigma * (0.5 * (left + right)))


@njit(cache=True)
def _conditional_feasible_mean(mu, sigma, threshold, mass):
    """Mean workload conditional on the analytically feasible interval."""

    if mass <= 1e-14 or threshold <= 0.0:
        return 0.0
    if sigma <= 1e-12:
        return min(math.exp(mu), threshold)
    shifted = (
        math.log(threshold) - mu - sigma * sigma
    ) / (sigma * math.sqrt(2.0))
    truncated_first_moment = math.exp(mu + 0.5 * sigma * sigma) * (
        0.5 * (1.0 + math.erf(shifted))
    )
    representative = truncated_first_moment / mass
    return min(max(representative, 1e-14), threshold * (1.0 - 1e-10))


@njit(cache=True)
def _truncated_lognormal_first_moment(mu, sigma, threshold):
    """Return E[C 1(C <= threshold)] for a LogNormal workload."""

    if threshold <= 0.0:
        return 0.0
    if sigma <= 1e-12:
        workload = math.exp(mu)
        return workload if workload <= threshold else 0.0
    z = (
        math.log(threshold) - mu - sigma * sigma
    ) / (sigma * math.sqrt(2.0))
    return math.exp(mu + 0.5 * sigma * sigma) * (
        0.5 * (1.0 + math.erf(z))
    )


@njit(cache=True)
def _mec_capacity(mec, dt, de, aa, bb, dd, ee):
    if aa[mec] >= 1e90:
        return 0.0
    time_capacity = 1e100
    energy_capacity = 1e100
    if bb[mec] > 0.0:
        time_capacity = (dt - aa[mec]) / bb[mec]
    if ee[mec] > 0.0:
        energy_capacity = (de - dd[mec]) / ee[mec]
    return max(0.0, min(time_capacity, energy_capacity))


@njit(cache=True)
def _insert_threshold(values, count, candidate):
    """Append one positive finite threshold to a fixed Numba array."""

    if candidate <= 1e-14 or candidate >= 1e99:
        return count
    values[count] = candidate
    return count + 1


@njit(cache=True)
def _threshold_stratified_support(
    mu,
    sigma,
    dt,
    de,
    pars,
    aa,
    bb,
    dd,
    ee,
    support_budget,
):
    """Build a small action threshold aware LogNormal support.

    The support preserves exact interval probability and first workload
    moments.  It includes Local and MEC feasibility capacities as well as the
    pairwise crossings of their affine normalized occupation models.  When
    more than ``support_budget - 1`` feasible strata exist, adjacent strata are
    merged.  The final feasible stratum is protected, equal preferred modes
    are merged first, and weighted squared moment distortion breaks ties.

    Returns fixed length workload, probability, and feasible arrays together
    with the number of populated entries.
    """

    budget = max(2, int(support_budget))
    feasible_limit = max(1, budget - 1)
    action_count = len(aa) + 1
    threshold_capacity = action_count + action_count * (action_count - 1) // 2
    raw_thresholds = np.zeros(max(1, threshold_capacity))
    action_capacity = np.zeros(action_count)
    action_intercept = np.zeros(action_count)
    action_slope = np.zeros(action_count)
    action_available = np.zeros(action_count, np.int64)

    local_capacity = _local_capacity(dt, de, pars)
    action_capacity[0] = local_capacity
    if local_capacity > 1e-14:
        action_available[0] = 1
        fmin = pars[2]
        fmax = pars[3]
        kappa = pars[4]
        hover = pars[5]
        if kappa > 1e-30:
            local_frequency = (
                (de + hover * dt) / (2.0 * kappa * dt)
            ) ** (1.0 / 3.0)
        else:
            local_frequency = fmax
        local_frequency = min(fmax, max(fmin, local_frequency))
        action_intercept[0] = 0.0
        action_slope[0] = (
            1.0 / (local_frequency * dt)
            + (kappa * local_frequency * local_frequency + hover / local_frequency)
            / de
        )

    for mec in range(len(aa)):
        action = mec + 1
        capacity = _mec_capacity(mec, dt, de, aa, bb, dd, ee)
        action_capacity[action] = capacity
        if capacity > 1e-14:
            action_available[action] = 1
            action_intercept[action] = aa[mec] / dt + dd[mec] / de
            action_slope[action] = bb[mec] / dt + ee[mec] / de

    threshold_count = 0
    for action in range(action_count):
        if action_available[action] != 0:
            threshold_count = _insert_threshold(
                raw_thresholds, threshold_count, action_capacity[action]
            )
    for first in range(action_count):
        if action_available[first] == 0:
            continue
        for second in range(first + 1, action_count):
            if action_available[second] == 0:
                continue
            denominator = action_slope[first] - action_slope[second]
            if abs(denominator) <= 1e-15:
                continue
            crossing = (
                action_intercept[second] - action_intercept[first]
            ) / denominator
            common_capacity = min(
                action_capacity[first], action_capacity[second]
            )
            if crossing > 1e-14 and crossing < common_capacity * (1.0 - 1e-10):
                threshold_count = _insert_threshold(
                    raw_thresholds, threshold_count, crossing
                )

    workloads = np.zeros(budget)
    probabilities = np.zeros(budget)
    feasible = np.zeros(budget, np.int64)
    if threshold_count == 0:
        workloads[0] = math.exp(mu)
        probabilities[0] = 1.0
        return workloads, probabilities, feasible, 1

    # Deterministic in place insertion sort.
    for index in range(1, threshold_count):
        value = raw_thresholds[index]
        cursor = index - 1
        while cursor >= 0 and raw_thresholds[cursor] > value:
            raw_thresholds[cursor + 1] = raw_thresholds[cursor]
            cursor -= 1
        raw_thresholds[cursor + 1] = value

    thresholds = np.zeros(threshold_count)
    unique_count = 0
    for index in range(threshold_count):
        value = raw_thresholds[index]
        if unique_count == 0:
            thresholds[unique_count] = value
            unique_count += 1
            continue
        previous = thresholds[unique_count - 1]
        tolerance = 1e-10 * max(1.0, abs(previous), abs(value))
        if abs(value - previous) > tolerance:
            thresholds[unique_count] = value
            unique_count += 1

    maximum_capacity = thresholds[unique_count - 1]
    if sigma <= 1e-12:
        deterministic = math.exp(mu)
        workloads[0] = deterministic
        probabilities[0] = 1.0
        feasible[0] = 1 if deterministic <= maximum_capacity + 1e-12 else 0
        return workloads, probabilities, feasible, 1

    lower_bounds = np.zeros(unique_count)
    upper_bounds = np.zeros(unique_count)
    masses = np.zeros(unique_count)
    moments = np.zeros(unique_count)
    representatives = np.zeros(unique_count)
    modes = np.full(unique_count, -1, np.int64)
    interval_count = 0
    lower = 0.0
    lower_mass = 0.0
    lower_moment = 0.0
    for index in range(unique_count):
        upper = thresholds[index]
        upper_mass = _lognormal_cdf(mu, sigma, upper)
        upper_moment = _truncated_lognormal_first_moment(mu, sigma, upper)
        mass = max(0.0, upper_mass - lower_mass)
        moment = max(0.0, upper_moment - lower_moment)
        if mass > 0.0:
            representative = moment / mass if moment > 0.0 else 0.5 * (lower + upper)
            representative = min(upper, max(max(lower, 1e-14), representative))
            ok, _time, _energy, mode, _frequency, _cost = (
                iteration_v2.execute_virtual_v2(
                    representative,
                    dt,
                    de,
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
            )
            lower_bounds[interval_count] = lower
            upper_bounds[interval_count] = upper
            masses[interval_count] = mass
            moments[interval_count] = moment
            representatives[interval_count] = representative
            modes[interval_count] = mode if ok else -1
            interval_count += 1
        lower = upper
        lower_mass = upper_mass
        lower_moment = upper_moment

    while interval_count > feasible_limit:
        best_pair = -1
        best_same_rank = 2
        best_distortion = 1e300
        # First pass protects the stratum next to the feasibility boundary.
        for protection_pass in range(2):
            for index in range(interval_count - 1):
                if protection_pass == 0 and index + 1 == interval_count - 1:
                    continue
                combined_mass = masses[index] + masses[index + 1]
                if combined_mass <= 0.0:
                    distortion = 0.0
                else:
                    delta = representatives[index] - representatives[index + 1]
                    distortion = (
                        masses[index]
                        * masses[index + 1]
                        / combined_mass
                        * delta
                        * delta
                    )
                same_rank = 0 if modes[index] == modes[index + 1] else 1
                if (
                    same_rank < best_same_rank
                    or (
                        same_rank == best_same_rank
                        and distortion < best_distortion - 1e-12
                    )
                    or (
                        same_rank == best_same_rank
                        and abs(distortion - best_distortion) <= 1e-12
                        and (best_pair < 0 or index < best_pair)
                    )
                ):
                    best_pair = index
                    best_same_rank = same_rank
                    best_distortion = distortion
            if best_pair >= 0:
                break

        first = best_pair
        second = first + 1
        masses[first] += masses[second]
        moments[first] += moments[second]
        upper_bounds[first] = upper_bounds[second]
        if masses[first] > 0.0:
            representatives[first] = moments[first] / masses[first]
        representatives[first] = min(
            upper_bounds[first],
            max(max(lower_bounds[first], 1e-14), representatives[first]),
        )
        ok, _time, _energy, mode, _frequency, _cost = (
            iteration_v2.execute_virtual_v2(
                representatives[first],
                dt,
                de,
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
        )
        modes[first] = mode if ok else -1
        for index in range(second, interval_count - 1):
            lower_bounds[index] = lower_bounds[index + 1]
            upper_bounds[index] = upper_bounds[index + 1]
            masses[index] = masses[index + 1]
            moments[index] = moments[index + 1]
            representatives[index] = representatives[index + 1]
            modes[index] = modes[index + 1]
        interval_count -= 1

    populated = 0
    for index in range(interval_count):
        workloads[populated] = representatives[index]
        probabilities[populated] = masses[index]
        feasible[populated] = 1
        populated += 1

    feasible_mass = _lognormal_cdf(mu, sigma, maximum_capacity)
    tail_mass = max(0.0, 1.0 - feasible_mass)
    if tail_mass > 0.0 and populated < budget:
        workloads[populated] = maximum_capacity * (1.0 + 1e-8)
        probabilities[populated] = tail_mass
        feasible[populated] = 0
        populated += 1

    return workloads, probabilities, feasible, populated


@njit(cache=True)
def _threshold_moment_task_stats(
    node,
    t,
    e,
    task,
    mu,
    sigma,
    ft,
    fe,
    weights,
    deadline,
    pars,
    aa,
    bb,
    dd,
    ee,
    support_budget,
):
    """Exact completion mass and threshold stratified resource moments."""

    arrival_t, arrival_e, dt, de = iteration_v2.resource_limits_v2(
        node, t, e, task, ft, fe, deadline, pars
    )
    if dt <= 0.0 or de <= 0.0:
        return 0.0, 0.0, 0.0
    workloads, probabilities, feasible, count = _threshold_stratified_support(
        mu[task],
        sigma[task],
        dt,
        de,
        pars,
        aa[task],
        bb[task],
        dd[task],
        ee[task],
        support_budget,
    )
    probability = 0.0
    expected_time = 0.0
    expected_energy = 0.0
    for branch in range(count):
        if feasible[branch] == 0:
            continue
        ok, service_time, service_energy, _mode, _frequency, _cost = (
            iteration_v2.execute_virtual_v2(
                workloads[branch],
                dt,
                de,
                pars,
                aa[task],
                bb[task],
                dd[task],
                ee[task],
                0,
                1.0,
                1.0,
                0.0,
            )
        )
        if ok:
            mass = probabilities[branch]
            probability += mass
            expected_time += mass * service_time
            expected_energy += mass * service_energy
    return probability, expected_time, expected_energy


@njit(cache=True)
def _threshold_remaining_potential(
    node,
    t,
    e,
    mask,
    mu,
    sigma,
    ft,
    fe,
    weights,
    deadline,
    pars,
    aa,
    bb,
    dd,
    ee,
    support_budget,
):
    """Two resource LP opportunity value using threshold moments."""

    n = len(weights) - 1
    values = np.zeros(n)
    time_cost = np.zeros(n)
    energy_cost = np.zeros(n)
    item_count = 0
    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (mask & bit) == 0:
            continue
        probability, service_time, service_energy = _threshold_moment_task_stats(
            node,
            t,
            e,
            task,
            mu,
            sigma,
            ft,
            fe,
            weights,
            deadline,
            pars,
            aa,
            bb,
            dd,
            ee,
            support_budget,
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
        values[item_count] = value
        time_cost[item_count] = incoming_time + service_time
        energy_cost[item_count] = incoming_energy + service_energy
        item_count += 1

    minimum_return_time = ft[node, 0]
    minimum_return_energy = fe[node, 0]
    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (mask & bit) != 0:
            minimum_return_time = min(minimum_return_time, ft[task, 0])
            minimum_return_energy = min(minimum_return_energy, fe[task, 0])
    capacity_time = pars[0] - t - minimum_return_time
    capacity_energy = pars[1] - e - minimum_return_energy
    return iteration_v2.lp_two_resource_exact(
        capacity_time,
        capacity_energy,
        values,
        time_cost,
        energy_cost,
        item_count,
    )


@njit(cache=True)
def _threshold_bellman_state_value(
    node,
    t,
    e,
    mask,
    mu,
    sigma,
    ft,
    fe,
    weights,
    deadline,
    pars,
    aa,
    bb,
    dd,
    ee,
    support_budget,
    lp_weight,
    depth,
):
    """Finite-depth nonanticipative Bellman value with an LP terminal value.

    ``depth`` counts future routing decisions represented explicitly.  At
    each decision, workload intervals retain their exact LogNormal mass and
    conditional first moment.  Skip, Local, and every feasible MEC action are
    then compared separately before the recursion continues.  Therefore a
    failed execution branch is never collapsed into the successful resource
    state.  The fractional two-resource LP is used only at depth zero.
    """

    if mask == 0:
        return 0.0, 0
    if depth <= 0:
        terminal = _threshold_remaining_potential(
            node,
            t,
            e,
            mask,
            mu,
            sigma,
            ft,
            fe,
            weights,
            deadline,
            pars,
            aa,
            bb,
            dd,
            ee,
            support_budget,
        )
        return lp_weight * terminal, 0

    n = len(weights) - 1
    best_value = 0.0
    best_task = 0
    expanded_total = 0
    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (mask & bit) == 0:
            continue
        if not iteration_v2.visit_ok_v2(
            node, t, e, task, ft, fe, deadline, pars
        ):
            continue
        arrival_t, arrival_e, dt, de = iteration_v2.resource_limits_v2(
            node, t, e, task, ft, fe, deadline, pars
        )
        if dt <= 0.0 or de <= 0.0:
            continue
        remaining = mask & (~bit)
        workloads, probabilities, feasible, branch_count = (
            _threshold_stratified_support(
                mu[task],
                sigma[task],
                dt,
                de,
                pars,
                aa[task],
                bb[task],
                dd[task],
                ee[task],
                support_budget,
            )
        )
        skip_future, expanded = _threshold_bellman_state_value(
            task,
            arrival_t,
            arrival_e,
            remaining,
            mu,
            sigma,
            ft,
            fe,
            weights,
            deadline,
            pars,
            aa,
            bb,
            dd,
            ee,
            support_budget,
            lp_weight,
            depth - 1,
        )
        expanded_total += expanded
        expected_value = 0.0
        for branch in range(branch_count):
            branch_best = skip_future
            if feasible[branch] != 0:
                workload = workloads[branch]
                local_ok, local_t, local_e, _mode, _frequency, _cost = (
                    iteration_v2.execute_virtual_v2(
                        workload,
                        dt,
                        de,
                        pars,
                        aa[task],
                        bb[task],
                        dd[task],
                        ee[task],
                        1,
                        1.0,
                        1.0,
                        0.0,
                    )
                )
                if local_ok:
                    future, expanded = _threshold_bellman_state_value(
                        task,
                        arrival_t + local_t,
                        arrival_e + local_e,
                        remaining,
                        mu,
                        sigma,
                        ft,
                        fe,
                        weights,
                        deadline,
                        pars,
                        aa,
                        bb,
                        dd,
                        ee,
                        support_budget,
                        lp_weight,
                        depth - 1,
                    )
                    expanded_total += expanded
                    value = weights[task] + future
                    if value > branch_best + 1e-12:
                        branch_best = value

                for mec in range(aa.shape[1]):
                    if aa[task, mec] >= 1e90:
                        continue
                    service_time = aa[task, mec] + bb[task, mec] * workload
                    service_energy = dd[task, mec] + ee[task, mec] * workload
                    if service_time > dt + 1e-9 or service_energy > de + 1e-9:
                        continue
                    future, expanded = _threshold_bellman_state_value(
                        task,
                        arrival_t + service_time,
                        arrival_e + service_energy,
                        remaining,
                        mu,
                        sigma,
                        ft,
                        fe,
                        weights,
                        deadline,
                        pars,
                        aa,
                        bb,
                        dd,
                        ee,
                        support_budget,
                        lp_weight,
                        depth - 1,
                    )
                    expanded_total += expanded
                    value = weights[task] + future
                    if value > branch_best + 1e-12:
                        branch_best = value
            expected_value += probabilities[branch] * branch_best
            expanded_total += 1

        if expected_value > best_value + 1e-12 or (
            abs(expected_value - best_value) <= 1e-12
            and expected_value > 0.0
            and (best_task == 0 or task < best_task)
        ):
            best_value = expected_value
            best_task = task
    return best_value, expanded_total


@njit(cache=True)
def top_choose_threshold_bellman_v2(
    node,
    t,
    e,
    visited_mask,
    mu,
    sigma,
    ft,
    fe,
    weights,
    deadline,
    pars,
    aa,
    bb,
    dd,
    ee,
    support_budget,
    lp_weight,
    continuation_depth,
):
    """Choose the root task using the same Bellman continuation as arrival."""

    n = len(weights) - 1
    all_mask = (np.uint64(1) << np.uint64(n)) - np.uint64(1)
    unvisited = all_mask & (~visited_mask)
    best_task = 0
    best_value = 0.0
    expanded_total = 0
    task_values = np.full(n + 1, -1e100)

    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (unvisited & bit) == 0:
            continue
        if not iteration_v2.visit_ok_v2(
            node, t, e, task, ft, fe, deadline, pars
        ):
            continue
        arrival_t, arrival_e, dt, de = iteration_v2.resource_limits_v2(
            node, t, e, task, ft, fe, deadline, pars
        )
        if dt <= 0.0 or de <= 0.0:
            continue
        remaining = unvisited & (~bit)
        workloads, probabilities, feasible, branch_count = (
            _threshold_stratified_support(
                mu[task],
                sigma[task],
                dt,
                de,
                pars,
                aa[task],
                bb[task],
                dd[task],
                ee[task],
                support_budget,
            )
        )
        skip_future, expanded = _threshold_bellman_state_value(
            task,
            arrival_t,
            arrival_e,
            remaining,
            mu,
            sigma,
            ft,
            fe,
            weights,
            deadline,
            pars,
            aa,
            bb,
            dd,
            ee,
            support_budget,
            lp_weight,
            continuation_depth,
        )
        expanded_total += expanded
        expected_value = 0.0
        for branch in range(branch_count):
            branch_best = skip_future
            if feasible[branch] != 0:
                workload = workloads[branch]
                local_ok, local_t, local_e, _mode, _frequency, _cost = (
                    iteration_v2.execute_virtual_v2(
                        workload,
                        dt,
                        de,
                        pars,
                        aa[task],
                        bb[task],
                        dd[task],
                        ee[task],
                        1,
                        1.0,
                        1.0,
                        0.0,
                    )
                )
                if local_ok:
                    future, expanded = _threshold_bellman_state_value(
                        task,
                        arrival_t + local_t,
                        arrival_e + local_e,
                        remaining,
                        mu,
                        sigma,
                        ft,
                        fe,
                        weights,
                        deadline,
                        pars,
                        aa,
                        bb,
                        dd,
                        ee,
                        support_budget,
                        lp_weight,
                        continuation_depth,
                    )
                    expanded_total += expanded
                    value = weights[task] + future
                    if value > branch_best + 1e-12:
                        branch_best = value
                for mec in range(aa.shape[1]):
                    if aa[task, mec] >= 1e90:
                        continue
                    service_time = aa[task, mec] + bb[task, mec] * workload
                    service_energy = dd[task, mec] + ee[task, mec] * workload
                    if service_time > dt + 1e-9 or service_energy > de + 1e-9:
                        continue
                    future, expanded = _threshold_bellman_state_value(
                        task,
                        arrival_t + service_time,
                        arrival_e + service_energy,
                        remaining,
                        mu,
                        sigma,
                        ft,
                        fe,
                        weights,
                        deadline,
                        pars,
                        aa,
                        bb,
                        dd,
                        ee,
                        support_budget,
                        lp_weight,
                        continuation_depth,
                    )
                    expanded_total += expanded
                    value = weights[task] + future
                    if value > branch_best + 1e-12:
                        branch_best = value
            expected_value += probabilities[branch] * branch_best
            expanded_total += 1

        task_values[task] = expected_value

        if expected_value > best_value + 1e-12 or (
            abs(expected_value - best_value) <= 1e-12
            and expected_value > 0.0
            and (best_task == 0 or task < best_task)
        ):
            best_task = task
            best_value = expected_value
    return best_task, best_value, expanded_total, task_values


@njit(cache=True)
def _threshold_beam_future(
    node0,
    time0,
    energy0,
    mask0,
    initial_value,
    mu,
    sigma,
    ft,
    fe,
    weights,
    deadline,
    pars,
    aa,
    bb,
    dd,
    ee,
    support_budget,
    beam_width,
    lp_weight,
    max_depth=0,
):
    """Expected state beam with dynamic threshold moment aggregation."""

    if mask0 == 0:
        return initial_value, 0
    n = len(weights) - 1
    width = max(1, int(beam_width))
    beam_node = np.zeros(width, np.int64)
    beam_time = np.zeros(width)
    beam_energy = np.zeros(width)
    beam_mask = np.zeros(width, np.uint64)
    beam_value = np.zeros(width)
    beam_node[0] = node0
    beam_time[0] = time0
    beam_energy[0] = energy0
    beam_mask[0] = mask0
    beam_value[0] = initial_value
    beam_count = 1
    best_terminal = initial_value
    expanded_total = 0

    depth_limit = n if max_depth <= 0 else min(n, int(max_depth))
    for _depth in range(depth_limit):
        capacity = width * n
        child_node = np.zeros(capacity, np.int64)
        child_time = np.zeros(capacity)
        child_energy = np.zeros(capacity)
        child_mask = np.zeros(capacity, np.uint64)
        child_value = np.zeros(capacity)
        child_score = np.full(capacity, -1e100)
        child_count = 0
        for state_index in range(beam_count):
            best_terminal = max(best_terminal, beam_value[state_index])
            for task in range(1, n + 1):
                bit = np.uint64(1) << np.uint64(task - 1)
                if (beam_mask[state_index] & bit) == 0:
                    continue
                if not iteration_v2.visit_ok_v2(
                    beam_node[state_index],
                    beam_time[state_index],
                    beam_energy[state_index],
                    task,
                    ft,
                    fe,
                    deadline,
                    pars,
                ):
                    continue
                probability, service_time, service_energy = (
                    _threshold_moment_task_stats(
                        beam_node[state_index],
                        beam_time[state_index],
                        beam_energy[state_index],
                        task,
                        mu,
                        sigma,
                        ft,
                        fe,
                        weights,
                        deadline,
                        pars,
                        aa,
                        bb,
                        dd,
                        ee,
                        support_budget,
                    )
                )
                if probability <= 1e-14:
                    continue
                next_time = (
                    beam_time[state_index]
                    + ft[beam_node[state_index], task]
                    + service_time
                )
                next_energy = (
                    beam_energy[state_index]
                    + fe[beam_node[state_index], task]
                    + service_energy
                )
                next_mask = beam_mask[state_index] & (~bit)
                next_value = beam_value[state_index] + weights[task] * probability
                potential = _threshold_remaining_potential(
                    task,
                    next_time,
                    next_energy,
                    next_mask,
                    mu,
                    sigma,
                    ft,
                    fe,
                    weights,
                    deadline,
                    pars,
                    aa,
                    bb,
                    dd,
                    ee,
                    support_budget,
                )
                child_node[child_count] = task
                child_time[child_count] = next_time
                child_energy[child_count] = next_energy
                child_mask[child_count] = next_mask
                child_value[child_count] = next_value
                child_score[child_count] = next_value + lp_weight * potential
                child_count += 1
                expanded_total += 1
        if child_count == 0:
            break

        keep_count = min(width, child_count)
        for keep in range(keep_count):
            best_index = -1
            best_score = -1e100
            best_value = -1e100
            best_resource = 1e100
            best_node = 10**9
            for child in range(child_count):
                if child_score[child] <= -1e90:
                    continue
                resource = (
                    child_time[child] / max(pars[0], 1e-12)
                    + child_energy[child] / max(pars[1], 1e-12)
                )
                better = False
                if child_score[child] > best_score + 1e-12:
                    better = True
                elif abs(child_score[child] - best_score) <= 1e-12:
                    if child_value[child] > best_value + 1e-12:
                        better = True
                    elif abs(child_value[child] - best_value) <= 1e-12:
                        if resource < best_resource - 1e-12:
                            better = True
                        elif (
                            abs(resource - best_resource) <= 1e-12
                            and child_node[child] < best_node
                        ):
                            better = True
                if better:
                    best_index = child
                    best_score = child_score[child]
                    best_value = child_value[child]
                    best_resource = resource
                    best_node = child_node[child]
            if best_index < 0:
                break
            beam_node[keep] = child_node[best_index]
            beam_time[keep] = child_time[best_index]
            beam_energy[keep] = child_energy[best_index]
            beam_mask[keep] = child_mask[best_index]
            beam_value[keep] = child_value[best_index]
            child_score[best_index] = -1e100
            best_terminal = max(best_terminal, beam_value[keep])
        beam_count = keep_count

    # A bounded lookahead keeps the same two-resource LP opportunity term as
    # the Beam ranking rule.  Full-horizon callers (max_depth <= 0) preserve
    # the original behavior exactly.
    if 0 < depth_limit < n:
        for state_index in range(beam_count):
            potential = _threshold_remaining_potential(
                beam_node[state_index],
                beam_time[state_index],
                beam_energy[state_index],
                beam_mask[state_index],
                mu,
                sigma,
                ft,
                fe,
                weights,
                deadline,
                pars,
                aa,
                bb,
                dd,
                ee,
                support_budget,
            )
            terminal = beam_value[state_index] + lp_weight * potential
            if terminal > best_terminal:
                best_terminal = terminal
    return best_terminal, expanded_total


@njit(cache=True)
def top_choose_root_calibrated_v2(
    node,
    t,
    e,
    visited_mask,
    points,
    probs,
    mu,
    sigma,
    ft,
    fe,
    weights,
    deadline,
    pars,
    aa,
    bb,
    dd,
    ee,
    beam_width,
    lp_weight,
    calibration_mode,
):
    """Choose the root task with one analytic probability correction.

    ``calibration_mode=1`` corrects only the immediate priority reward:

        Q_R = Q_quantile + w_i (p_i - p_hat_i).

    ``calibration_mode=2`` replaces the complete feasible/infeasible root mass:

        Q_M = p_i Q_feasible + (1-p_i) Q_skip.

    Both modes retain the same quantile continuation and LP guidance as AR.
    """

    n = len(weights) - 1
    all_mask = (np.uint64(1) << np.uint64(n)) - np.uint64(1)
    unvisited = all_mask & (~visited_mask)
    best_task = 0
    best_value = 0.0
    expanded_total = 0

    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (unvisited & bit) == 0:
            continue
        if not iteration_v2.visit_ok_v2(
            node, t, e, task, ft, fe, deadline, pars
        ):
            continue

        arrival_t, arrival_e, dt, de = iteration_v2.resource_limits_v2(
            node, t, e, task, ft, fe, deadline, pars
        )
        if dt <= 0.0 or de <= 0.0:
            continue
        threshold = _maximum_capacity(task, dt, de, pars, aa, bb, dd, ee)
        exact_mass = _lognormal_cdf(mu[task], sigma[task], threshold)
        remaining = unvisited & (~bit)

        quantile_value = 0.0
        discrete_mass = 0.0
        feasible_value_sum = 0.0
        for branch in range(len(probs)):
            ok, service_t, service_e, _mode, _frequency, _cost = (
                iteration_v2.execute_virtual_v2(
                    points[task, branch],
                    dt,
                    de,
                    pars,
                    aa[task],
                    bb[task],
                    dd[task],
                    ee[task],
                    0,
                    1.0,
                    1.0,
                    0.0,
                )
            )
            if ok:
                next_t = arrival_t + service_t
                next_e = arrival_e + service_e
                immediate = weights[task]
                discrete_mass += probs[branch]
            else:
                next_t = arrival_t
                next_e = arrival_e
                immediate = 0.0

            if remaining == 0:
                total = immediate
                expanded = 0
            else:
                total, expanded = iteration_v2.beam_future_v2(
                    task,
                    next_t,
                    next_e,
                    remaining,
                    immediate,
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
                    0,
                    1.0,
                    1.0,
                    0.0,
                    beam_width,
                    True,
                    lp_weight,
                    True,
                    0.0,
                )
            expanded_total += expanded
            quantile_value += probs[branch] * total
            if ok:
                feasible_value_sum += probs[branch] * total

        if calibration_mode == 1:
            value = quantile_value + weights[task] * (exact_mass - discrete_mass)
        else:
            if discrete_mass > 1e-14:
                feasible_value = feasible_value_sum / discrete_mass
            elif exact_mass > 1e-14:
                representative = _conditional_feasible_mean(
                    mu[task], sigma[task], threshold, exact_mass
                )
                ok, service_t, service_e, _mode, _frequency, _cost = (
                    iteration_v2.execute_virtual_v2(
                        representative,
                        dt,
                        de,
                        pars,
                        aa[task],
                        bb[task],
                        dd[task],
                        ee[task],
                        0,
                        1.0,
                        1.0,
                        0.0,
                    )
                )
                if ok and remaining != 0:
                    feasible_value, expanded = iteration_v2.beam_future_v2(
                        task,
                        arrival_t + service_t,
                        arrival_e + service_e,
                        remaining,
                        weights[task],
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
                        0,
                        1.0,
                        1.0,
                        0.0,
                        beam_width,
                        True,
                        lp_weight,
                        True,
                        0.0,
                    )
                    expanded_total += expanded
                elif ok:
                    feasible_value = weights[task]
                else:
                    feasible_value = 0.0
            else:
                feasible_value = 0.0

            if remaining == 0:
                skip_value = 0.0
            else:
                skip_value, expanded = iteration_v2.beam_future_v2(
                    task,
                    arrival_t,
                    arrival_e,
                    remaining,
                    0.0,
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
                    0,
                    1.0,
                    1.0,
                    0.0,
                    beam_width,
                    True,
                    lp_weight,
                    True,
                    0.0,
                )
                expanded_total += expanded
            value = exact_mass * feasible_value + (1.0 - exact_mass) * skip_value

        if value > best_value + 1e-12 or (
            abs(value - best_value) <= 1e-12
            and (best_task == 0 or task < best_task)
        ):
            best_task = task
            best_value = value

    return best_task, best_value, expanded_total


@njit(cache=True)
def _native_future_value(
    task,
    next_t,
    next_e,
    remaining,
    immediate,
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
    beam_width,
    lp_weight,
    reuse_expected_stats,
):
    if remaining == 0:
        return immediate, 0
    if reuse_expected_stats != 0:
        return _beam_future_expected_stats_cached(
            task,
            next_t,
            next_e,
            remaining,
            immediate,
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
            beam_width,
            lp_weight,
        )
    return iteration_v2.beam_future_v2(
        task,
        next_t,
        next_e,
        remaining,
        immediate,
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
        0,
        1.0,
        1.0,
        0.0,
        beam_width,
        True,
        lp_weight,
        True,
        0.0,
    )


@njit(cache=True)
def _fill_expected_task_stats(
    node,
    t,
    e,
    mask,
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
    probability,
    expected_time,
    expected_energy,
):
    """Fill the distribution statistics used by one Beam state.

    The native Beam first computes these statistics to rank a child through
    the two-resource LP and then computes the same statistics again if that
    child survives to the next Beam layer. Keeping the three arrays with the
    child removes only that duplicate work. No state is rounded and no Beam
    decision is changed.
    """

    n = len(deadline) - 1
    for task in range(1, n + 1):
        probability[task] = 0.0
        expected_time[task] = 0.0
        expected_energy[task] = 0.0
        bit = np.uint64(1) << np.uint64(task - 1)
        if (mask & bit) == 0:
            continue
        p, service_t, service_e = iteration_v2.expected_task_stats_v2(
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
        probability[task] = p
        expected_time[task] = service_t
        expected_energy[task] = service_e


@njit(cache=True)
def _remaining_potential_from_stats(
    node,
    t,
    e,
    mask,
    probability,
    expected_time,
    expected_energy,
    ft,
    fe,
    weights,
    pars,
):
    """Evaluate the unchanged native LP guidance from cached statistics."""

    n = len(weights) - 1
    values = np.zeros(n)
    time_cost = np.zeros(n)
    energy_cost = np.zeros(n)
    count = 0
    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (mask & bit) == 0:
            continue
        value = weights[task] * probability[task]
        if value <= 1e-14:
            continue

        incoming_time = ft[node, task]
        incoming_energy = fe[node, task]
        for predecessor in range(1, n + 1):
            if predecessor == task:
                continue
            predecessor_bit = np.uint64(1) << np.uint64(predecessor - 1)
            if (mask & predecessor_bit) != 0:
                if ft[predecessor, task] < incoming_time:
                    incoming_time = ft[predecessor, task]
                if fe[predecessor, task] < incoming_energy:
                    incoming_energy = fe[predecessor, task]

        values[count] = value
        time_cost[count] = incoming_time + expected_time[task]
        energy_cost[count] = incoming_energy + expected_energy[task]
        count += 1

    capacity_time = pars[0] - t
    capacity_energy = pars[1] - e
    minimum_return_time = ft[node, 0]
    minimum_return_energy = fe[node, 0]
    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (mask & bit) != 0:
            if ft[task, 0] < minimum_return_time:
                minimum_return_time = ft[task, 0]
            if fe[task, 0] < minimum_return_energy:
                minimum_return_energy = fe[task, 0]
    capacity_time -= minimum_return_time
    capacity_energy -= minimum_return_energy
    return iteration_v2.lp_two_resource_exact(
        capacity_time,
        capacity_energy,
        values,
        time_cost,
        energy_cost,
        count,
    )


@njit(cache=True)
def _beam_future_expected_stats_cached(
    node0,
    t0,
    e0,
    mask0,
    initial_value,
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
    beam_width,
    lp_weight,
):
    """Native Beam with exact reuse of retained-state resource statistics."""

    if mask0 == 0:
        return initial_value, 0

    n = len(weights) - 1
    width = max(1, int(beam_width))
    beam_node = np.zeros(width, np.int64)
    beam_time = np.zeros(width)
    beam_energy = np.zeros(width)
    beam_mask = np.zeros(width, np.uint64)
    beam_value = np.zeros(width)
    beam_probability = np.zeros((width, n + 1))
    beam_expected_time = np.zeros((width, n + 1))
    beam_expected_energy = np.zeros((width, n + 1))
    beam_node[0] = node0
    beam_time[0] = t0
    beam_energy[0] = e0
    beam_mask[0] = mask0
    beam_value[0] = initial_value
    _fill_expected_task_stats(
        node0,
        t0,
        e0,
        mask0,
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
        beam_probability[0],
        beam_expected_time[0],
        beam_expected_energy[0],
    )
    beam_count = 1
    best_terminal = initial_value
    expanded = 0

    for _depth in range(n):
        capacity = width * n
        child_node = np.zeros(capacity, np.int64)
        child_time = np.zeros(capacity)
        child_energy = np.zeros(capacity)
        child_mask = np.zeros(capacity, np.uint64)
        child_value = np.zeros(capacity)
        child_score = np.full(capacity, -1e100)
        child_probability = np.zeros((capacity, n + 1))
        child_expected_time = np.zeros((capacity, n + 1))
        child_expected_energy = np.zeros((capacity, n + 1))
        child_count = 0

        for state_index in range(beam_count):
            if beam_value[state_index] > best_terminal:
                best_terminal = beam_value[state_index]
            for task in range(1, n + 1):
                bit = np.uint64(1) << np.uint64(task - 1)
                if (beam_mask[state_index] & bit) == 0:
                    continue
                p = beam_probability[state_index, task]
                if p <= 1e-14:
                    continue
                next_time = (
                    beam_time[state_index]
                    + ft[beam_node[state_index], task]
                    + beam_expected_time[state_index, task]
                )
                next_energy = (
                    beam_energy[state_index]
                    + fe[beam_node[state_index], task]
                    + beam_expected_energy[state_index, task]
                )
                next_mask = beam_mask[state_index] & (~bit)
                next_value = beam_value[state_index] + weights[task] * p

                _fill_expected_task_stats(
                    task,
                    next_time,
                    next_energy,
                    next_mask,
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
                    child_probability[child_count],
                    child_expected_time[child_count],
                    child_expected_energy[child_count],
                )
                potential = _remaining_potential_from_stats(
                    task,
                    next_time,
                    next_energy,
                    next_mask,
                    child_probability[child_count],
                    child_expected_time[child_count],
                    child_expected_energy[child_count],
                    ft,
                    fe,
                    weights,
                    pars,
                )

                child_node[child_count] = task
                child_time[child_count] = next_time
                child_energy[child_count] = next_energy
                child_mask[child_count] = next_mask
                child_value[child_count] = next_value
                child_score[child_count] = next_value + lp_weight * potential
                child_count += 1
                expanded += 1

        if child_count == 0:
            break

        keep_count = min(width, child_count)
        for keep in range(keep_count):
            best_index = -1
            best_score = -1e100
            best_value = -1e100
            best_resource = 1e100
            best_node = 10**9
            for child in range(child_count):
                if child_score[child] <= -1e90:
                    continue
                resource = (
                    child_time[child] / max(pars[0], 1e-12)
                    + child_energy[child] / max(pars[1], 1e-12)
                )
                better = False
                if child_score[child] > best_score + 1e-12:
                    better = True
                elif abs(child_score[child] - best_score) <= 1e-12:
                    if child_value[child] > best_value + 1e-12:
                        better = True
                    elif abs(child_value[child] - best_value) <= 1e-12:
                        if resource < best_resource - 1e-12:
                            better = True
                        elif (
                            abs(resource - best_resource) <= 1e-12
                            and child_node[child] < best_node
                        ):
                            better = True
                if better:
                    best_index = child
                    best_score = child_score[child]
                    best_value = child_value[child]
                    best_resource = resource
                    best_node = child_node[child]

            if best_index < 0:
                break
            beam_node[keep] = child_node[best_index]
            beam_time[keep] = child_time[best_index]
            beam_energy[keep] = child_energy[best_index]
            beam_mask[keep] = child_mask[best_index]
            beam_value[keep] = child_value[best_index]
            for task in range(1, n + 1):
                beam_probability[keep, task] = child_probability[best_index, task]
                beam_expected_time[keep, task] = child_expected_time[best_index, task]
                beam_expected_energy[keep, task] = child_expected_energy[best_index, task]
            child_score[best_index] = -1e100
            if beam_value[keep] > best_terminal:
                best_terminal = beam_value[keep]
        beam_count = keep_count

    return best_terminal, expanded


@njit(cache=True)
def top_choose_action_consistent_v2(
    node,
    t,
    e,
    visited_mask,
    points,
    probs,
    mu,
    sigma,
    ft,
    fe,
    weights,
    deadline,
    pars,
    aa,
    bb,
    dd,
    ee,
    beam_width,
    lp_weight,
    dual_price_local,
    dual_action_mode,
    analytic_root_mass,
):
    """One step action consistent policy improvement of the AR route.

    For each root task and workload quantile, the kernel evaluates Skip, the
    feasible Local action, and every feasible MEC action.  All actions use the
    same deterministic continuation model.  This removes the mismatch in
    which the original route assumed the occupation minimizing action but the
    post reveal controller later selected a different action.
    """

    n = len(weights) - 1
    all_mask = (np.uint64(1) << np.uint64(n)) - np.uint64(1)
    unvisited = all_mask & (~visited_mask)
    best_task = 0
    best_value = 0.0
    best_price_t = 0.0
    best_price_e = 0.0
    expanded_total = 0
    reuse_expected_stats = 1 if dual_action_mode == 5 else 0

    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (unvisited & bit) == 0:
            continue
        if not iteration_v2.visit_ok_v2(
            node, t, e, task, ft, fe, deadline, pars
        ):
            continue
        arrival_t, arrival_e, dt, de = iteration_v2.resource_limits_v2(
            node, t, e, task, ft, fe, deadline, pars
        )
        if dt <= 0.0 or de <= 0.0:
            continue
        remaining = unvisited & (~bit)

        price_t = 0.0
        price_e = 0.0
        local_time_weight = 1.0
        local_energy_weight = 1.0
        if (dual_price_local != 0 or dual_action_mode != 0) and remaining != 0:
            price_t, price_e, _lp_value = state_lp_dual_prices_v2(
                task,
                arrival_t,
                arrival_e,
                remaining,
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
            )
            local_time_weight, local_energy_weight = _dual_execution_weights(
                price_t, price_e, dt, de
            )

        # Skip has the same successor state for every workload branch and is
        # evaluated once per candidate task.
        skip_value, expanded = _native_future_value(
            task,
            arrival_t,
            arrival_e,
            remaining,
            0.0,
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
            beam_width,
            lp_weight,
            reuse_expected_stats,
        )
        expanded_total += expanded
        feasible_mass = 1.0
        if analytic_root_mass != 0:
            maximum_capacity = _maximum_capacity(
                task, dt, de, pars, aa, bb, dd, ee
            )
            feasible_mass = _lognormal_cdf(
                mu[task], sigma[task], maximum_capacity
            )
        expected_value = (1.0 - feasible_mass) * skip_value

        for branch in range(len(probs)):
            if feasible_mass <= 1e-14:
                break
            workload = points[task, branch]
            branch_probability = probs[branch]
            if analytic_root_mass != 0:
                base_uniform = _lognormal_cdf(
                    mu[task], sigma[task], points[task, branch]
                )
                workload = min(
                    maximum_capacity,
                    _lognormal_ppf(
                        mu[task],
                        sigma[task],
                        feasible_mass * base_uniform,
                    ),
                )
                branch_probability *= feasible_mass

            if dual_action_mode == 1:
                (
                    complete,
                    service_t,
                    service_e,
                    _selected_mode,
                    _selected_frequency,
                    _gain,
                    _dual_cost,
                ) = _reduced_cost_action(
                    workload,
                    dt,
                    de,
                    weights[task],
                    price_t,
                    price_e,
                    pars,
                    aa[task],
                    bb[task],
                    dd[task],
                    ee[task],
                )
                branch_value = skip_value
                if complete:
                    branch_value, expanded = _native_future_value(
                        task,
                        arrival_t + service_t,
                        arrival_e + service_e,
                        remaining,
                        weights[task],
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
                        beam_width,
                        lp_weight,
                        reuse_expected_stats,
                    )
                    expanded_total += expanded
                expected_value += branch_probability * branch_value
                continue

            if dual_action_mode == 3 or dual_action_mode == 5:
                (
                    has_successful_action,
                    service_t,
                    service_e,
                    _selected_mode,
                    _selected_frequency,
                    _dual_cost,
                ) = _shadow_cost_action(
                    workload,
                    dt,
                    de,
                    price_t,
                    price_e,
                    pars,
                    aa[task],
                    bb[task],
                    dd[task],
                    ee[task],
                )
                branch_value = skip_value
                if has_successful_action:
                    complete_value, expanded = _native_future_value(
                        task,
                        arrival_t + service_t,
                        arrival_e + service_e,
                        remaining,
                        weights[task],
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
                        beam_width,
                        lp_weight,
                        reuse_expected_stats,
                    )
                    expanded_total += expanded
                    if complete_value >= branch_value - 1e-12:
                        branch_value = complete_value
                expected_value += branch_probability * branch_value
                continue

            if dual_action_mode == 4:
                (
                    frontier_times,
                    frontier_energies,
                    _frontier_modes,
                    _frontier_frequencies,
                    frontier_count,
                    _candidate_count,
                ) = _pareto_shadow_actions(
                    workload,
                    dt,
                    de,
                    price_t,
                    price_e,
                    pars,
                    aa[task],
                    bb[task],
                    dd[task],
                    ee[task],
                )
                branch_best = skip_value
                for action_index in range(frontier_count):
                    complete_value, expanded = _native_future_value(
                        task,
                        arrival_t + frontier_times[action_index],
                        arrival_e + frontier_energies[action_index],
                        remaining,
                        weights[task],
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
                        beam_width,
                        lp_weight,
                        reuse_expected_stats,
                    )
                    expanded_total += expanded
                    if complete_value > branch_best + 1e-12:
                        branch_best = complete_value
                expected_value += branch_probability * branch_best
                continue

            branch_best = skip_value

            local_ok, local_t, local_e, _mode, _frequency, _cost = (
                iteration_v2.execute_virtual_v2(
                    workload,
                    dt,
                    de,
                    pars,
                    aa[task],
                    bb[task],
                    dd[task],
                    ee[task],
                    1,
                    local_time_weight,
                    local_energy_weight,
                    0.0,
                )
            )
            if local_ok:
                local_value, expanded = _native_future_value(
                    task,
                    arrival_t + local_t,
                    arrival_e + local_e,
                    remaining,
                    weights[task],
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
                    beam_width,
                    lp_weight,
                    reuse_expected_stats,
                )
                expanded_total += expanded
                if local_value > branch_best + 1e-12:
                    branch_best = local_value

            for mec in range(aa.shape[1]):
                if aa[task, mec] >= 1e90:
                    continue
                service_t = aa[task, mec] + bb[task, mec] * workload
                service_e = dd[task, mec] + ee[task, mec] * workload
                if service_t > dt + 1e-9 or service_e > de + 1e-9:
                    continue
                mec_value, expanded = _native_future_value(
                    task,
                    arrival_t + service_t,
                    arrival_e + service_e,
                    remaining,
                    weights[task],
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
                    beam_width,
                    lp_weight,
                    reuse_expected_stats,
                )
                expanded_total += expanded
                if mec_value > branch_best + 1e-12:
                    branch_best = mec_value

            expected_value += branch_probability * branch_best

        if expected_value > best_value + 1e-12 or (
            abs(expected_value - best_value) <= 1e-12
            and (best_task == 0 or task < best_task)
        ):
            best_task = task
            best_value = expected_value
            best_price_t = price_t
            best_price_e = price_e

    return best_task, best_value, expanded_total, best_price_t, best_price_e


@njit(cache=True)
def top_choose_threshold_stratified_action_consistent_v2(
    node,
    t,
    e,
    visited_mask,
    mu,
    sigma,
    ft,
    fe,
    weights,
    deadline,
    pars,
    aa,
    bb,
    dd,
    ee,
    support_budget,
    beam_width,
    lp_weight,
    max_depth=0,
):
    """Action consistent root recourse with analytical moment aggregation.

    Local and every MEC action are compared for each retained root stratum.
    Future tasks use one threshold moment matched successor inside a width
    bounded route beam.  No unvisited realized workload enters this kernel.
    """

    n = len(weights) - 1
    all_mask = (np.uint64(1) << np.uint64(n)) - np.uint64(1)
    unvisited = all_mask & (~visited_mask)
    best_task = 0
    best_value = 0.0
    expanded_total = 0

    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (unvisited & bit) == 0:
            continue
        if not iteration_v2.visit_ok_v2(
            node, t, e, task, ft, fe, deadline, pars
        ):
            continue
        arrival_t, arrival_e, dt, de = iteration_v2.resource_limits_v2(
            node, t, e, task, ft, fe, deadline, pars
        )
        if dt <= 0.0 or de <= 0.0:
            continue
        remaining = unvisited & (~bit)
        workloads, probabilities, feasible, branch_count = (
            _threshold_stratified_support(
                mu[task],
                sigma[task],
                dt,
                de,
                pars,
                aa[task],
                bb[task],
                dd[task],
                ee[task],
                support_budget,
            )
        )

        skip_value, expanded = _threshold_beam_future(
            task,
            arrival_t,
            arrival_e,
            remaining,
            0.0,
            mu,
            sigma,
            ft,
            fe,
            weights,
            deadline,
            pars,
            aa,
            bb,
            dd,
            ee,
            support_budget,
            beam_width,
            lp_weight,
            max_depth,
        )
        expanded_total += expanded
        expected_value = 0.0

        for branch in range(branch_count):
            branch_best = skip_value
            if feasible[branch] != 0:
                workload = workloads[branch]
                local_ok, local_t, local_e, _mode, _frequency, _cost = (
                    iteration_v2.execute_virtual_v2(
                        workload,
                        dt,
                        de,
                        pars,
                        aa[task],
                        bb[task],
                        dd[task],
                        ee[task],
                        1,
                        1.0,
                        1.0,
                        0.0,
                    )
                )
                if local_ok:
                    local_value, expanded = _threshold_beam_future(
                        task,
                        arrival_t + local_t,
                        arrival_e + local_e,
                        remaining,
                        weights[task],
                        mu,
                        sigma,
                        ft,
                        fe,
                        weights,
                        deadline,
                        pars,
                        aa,
                        bb,
                        dd,
                        ee,
                        support_budget,
                        beam_width,
                        lp_weight,
                        max_depth,
                    )
                    expanded_total += expanded
                    if local_value > branch_best + 1e-12:
                        branch_best = local_value

                for mec in range(aa.shape[1]):
                    if aa[task, mec] >= 1e90:
                        continue
                    service_time = aa[task, mec] + bb[task, mec] * workload
                    service_energy = dd[task, mec] + ee[task, mec] * workload
                    if service_time > dt + 1e-9 or service_energy > de + 1e-9:
                        continue
                    mec_value, expanded = _threshold_beam_future(
                        task,
                        arrival_t + service_time,
                        arrival_e + service_energy,
                        remaining,
                        weights[task],
                        mu,
                        sigma,
                        ft,
                        fe,
                        weights,
                        deadline,
                        pars,
                        aa,
                        bb,
                        dd,
                        ee,
                        support_budget,
                        beam_width,
                        lp_weight,
                        max_depth,
                    )
                    expanded_total += expanded
                    if mec_value > branch_best + 1e-12:
                        branch_best = mec_value

            expected_value += probabilities[branch] * branch_best

        if expected_value > best_value + 1e-12 or (
            abs(expected_value - best_value) <= 1e-12
            and (best_task == 0 or task < best_task)
        ):
            best_task = task
            best_value = expected_value

    return best_task, best_value, expanded_total
