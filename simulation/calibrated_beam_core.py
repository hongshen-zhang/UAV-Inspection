"""Numba kernels for analytic probability calibrated GH continuation beams."""

from __future__ import annotations

import math
import numpy as np
from numba import njit

from batch_adapter import iteration_v2, robust_core


@njit(cache=True)
def _remaining_potential(
    node,
    t,
    e,
    mask,
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
):
    """Two resource LP guidance with exact lognormal completion masses."""

    n = len(weights) - 1
    values = np.zeros(n)
    time_cost = np.zeros(n)
    energy_cost = np.zeros(n)
    count = 0
    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (mask & bit) == 0:
            continue
        probability, expected_time, expected_energy = robust_core.calibrated_task_stats(
            node,
            t,
            e,
            task,
            points,
            probs,
            mu,
            sigma,
            ft,
            fe,
            deadline,
            pars,
            aa,
            bb,
            dd,
            ee,
            0,
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
                if ft[predecessor, task] < incoming_time:
                    incoming_time = ft[predecessor, task]
                if fe[predecessor, task] < incoming_energy:
                    incoming_energy = fe[predecessor, task]
        values[count] = value
        time_cost[count] = incoming_time + expected_time
        energy_cost[count] = incoming_energy + expected_energy
        count += 1

    minimum_return_time = ft[node, 0]
    minimum_return_energy = fe[node, 0]
    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (mask & bit) != 0:
            if ft[task, 0] < minimum_return_time:
                minimum_return_time = ft[task, 0]
            if fe[task, 0] < minimum_return_energy:
                minimum_return_energy = fe[task, 0]
    capacity_time = pars[0] - t - minimum_return_time
    capacity_energy = pars[1] - e - minimum_return_energy
    return robust_core.lp_two_resource_exact(
        capacity_time,
        capacity_energy,
        values,
        time_cost,
        energy_cost,
        count,
    )


@njit(cache=True)
def beam_future_calibrated(
    node0,
    time0,
    energy0,
    mask0,
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
    max_depth,
):
    """Expected state beam with analytic completion probability at every task."""

    if mask0 == 0:
        return 0.0, 0
    n = len(weights) - 1
    width = max(1, int(beam_width))
    beam_node = np.zeros(width, np.int64)
    beam_time = np.zeros(width)
    beam_energy = np.zeros(width)
    beam_mask = np.zeros(width, np.uint64)
    beam_value = np.zeros(width)
    beam_terminal_score = np.zeros(width)
    beam_node[0] = node0
    beam_time[0] = time0
    beam_energy[0] = energy0
    beam_mask[0] = mask0
    beam_count = 1
    best_terminal = 0.0
    expanded = 0

    depth_limit = n if max_depth <= 0 else min(n, int(max_depth))
    for _ in range(depth_limit):
        capacity = width * n
        child_node = np.zeros(capacity, np.int64)
        child_time = np.zeros(capacity)
        child_energy = np.zeros(capacity)
        child_mask = np.zeros(capacity, np.uint64)
        child_value = np.zeros(capacity)
        child_score = np.full(capacity, -1e100)
        child_count = 0
        for state_index in range(beam_count):
            if beam_value[state_index] > best_terminal:
                best_terminal = beam_value[state_index]
            for task in range(1, n + 1):
                bit = np.uint64(1) << np.uint64(task - 1)
                if (beam_mask[state_index] & bit) == 0:
                    continue
                if not robust_core.visit_ok(
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
                probability, expected_time, expected_energy = robust_core.calibrated_task_stats(
                    beam_node[state_index],
                    beam_time[state_index],
                    beam_energy[state_index],
                    task,
                    points,
                    probs,
                    mu,
                    sigma,
                    ft,
                    fe,
                    deadline,
                    pars,
                    aa,
                    bb,
                    dd,
                    ee,
                    0,
                )
                if probability <= 1e-14:
                    continue
                next_time = (
                    beam_time[state_index]
                    + ft[beam_node[state_index], task]
                    + expected_time
                )
                next_energy = (
                    beam_energy[state_index]
                    + fe[beam_node[state_index], task]
                    + expected_energy
                )
                next_mask = beam_mask[state_index] & (~bit)
                next_value = beam_value[state_index] + weights[task] * probability
                potential = _remaining_potential(
                    task,
                    next_time,
                    next_energy,
                    next_mask,
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
        selected_nodes = np.full(width, -1, np.int64)
        actual_keep = 0
        for keep in range(keep_count):
            best_index = -1
            best_score = -1e100
            best_value = -1e100
            best_resource = 1e100
            best_node = 10**9
            for pass_index in range(2):
                for child in range(child_count):
                    if child_score[child] <= -1e90:
                        continue
                    if pass_index == 0:
                        duplicate = False
                        for prior in range(keep):
                            if selected_nodes[prior] == child_node[child]:
                                duplicate = True
                                break
                        if duplicate:
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
                if best_index >= 0:
                    break
            if best_index < 0:
                break
            beam_node[keep] = child_node[best_index]
            beam_time[keep] = child_time[best_index]
            beam_energy[keep] = child_energy[best_index]
            beam_mask[keep] = child_mask[best_index]
            beam_value[keep] = child_value[best_index]
            beam_terminal_score[keep] = child_score[best_index]
            selected_nodes[keep] = child_node[best_index]
            child_score[best_index] = -1e100
            actual_keep += 1
            if beam_value[keep] > best_terminal:
                best_terminal = beam_value[keep]
        beam_count = actual_keep
        if beam_count == 0:
            break
    if 0 < depth_limit < n and beam_count > 0:
        cutoff_value = beam_terminal_score[0]
        for state_index in range(1, beam_count):
            if beam_terminal_score[state_index] > cutoff_value:
                cutoff_value = beam_terminal_score[state_index]
        if cutoff_value > best_terminal:
            best_terminal = cutoff_value
    return best_terminal, expanded


@njit(cache=True)
def _fill_calibrated_stats(
    node,
    time_s,
    energy_kj,
    mask,
    points,
    probs,
    mu,
    sigma,
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
    """Fill exact-mass resource statistics for one retained Beam state."""

    n = len(deadline) - 1
    for task in range(1, n + 1):
        probability[task] = 0.0
        expected_time[task] = 0.0
        expected_energy[task] = 0.0
        bit = np.uint64(1) << np.uint64(task - 1)
        if (mask & bit) == 0:
            continue
        p, service_t, service_e = robust_core.calibrated_task_stats(
            node,
            time_s,
            energy_kj,
            task,
            points,
            probs,
            mu,
            sigma,
            ft,
            fe,
            deadline,
            pars,
            aa,
            bb,
            dd,
            ee,
            0,
        )
        probability[task] = p
        expected_time[task] = service_t
        expected_energy[task] = service_e


@njit(cache=True)
def _calibrated_potential_from_stats(
    node,
    time_s,
    energy_kj,
    mask,
    probability,
    expected_time,
    expected_energy,
    ft,
    fe,
    weights,
    pars,
):
    """Evaluate the unchanged calibrated LP guide from retained statistics."""

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

    minimum_return_time = ft[node, 0]
    minimum_return_energy = fe[node, 0]
    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (mask & bit) != 0:
            if ft[task, 0] < minimum_return_time:
                minimum_return_time = ft[task, 0]
            if fe[task, 0] < minimum_return_energy:
                minimum_return_energy = fe[task, 0]
    capacity_time = pars[0] - time_s - minimum_return_time
    capacity_energy = pars[1] - energy_kj - minimum_return_energy
    return robust_core.lp_two_resource_exact(
        capacity_time,
        capacity_energy,
        values,
        time_cost,
        energy_cost,
        count,
    )


@njit(cache=True)
def _beam_future_calibrated_cached_impl(
    node0,
    time0,
    energy0,
    mask0,
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
    max_depth,
    branch_aware_guidance,
):
    """Cached calibrated Beam with optional Bernoulli successor guidance."""

    if mask0 == 0:
        return 0.0, 0
    n = len(weights) - 1
    width = max(1, int(beam_width))
    beam_node = np.zeros(width, np.int64)
    beam_time = np.zeros(width)
    beam_energy = np.zeros(width)
    beam_mask = np.zeros(width, np.uint64)
    beam_value = np.zeros(width)
    beam_terminal_score = np.zeros(width)
    beam_probability = np.zeros((width, n + 1))
    beam_expected_time = np.zeros((width, n + 1))
    beam_expected_energy = np.zeros((width, n + 1))
    beam_node[0] = node0
    beam_time[0] = time0
    beam_energy[0] = energy0
    beam_mask[0] = mask0
    _fill_calibrated_stats(
        node0,
        time0,
        energy0,
        mask0,
        points,
        probs,
        mu,
        sigma,
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
    best_terminal = 0.0
    expanded = 0

    depth_limit = n if max_depth <= 0 else min(n, int(max_depth))
    for _depth in range(depth_limit):
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
                probability = beam_probability[state_index, task]
                if probability <= 1e-14:
                    continue
                arrival_time = (
                    beam_time[state_index]
                    + ft[beam_node[state_index], task]
                )
                arrival_energy = (
                    beam_energy[state_index]
                    + fe[beam_node[state_index], task]
                )
                next_time = (
                    arrival_time + beam_expected_time[state_index, task]
                )
                next_energy = (
                    arrival_energy + beam_expected_energy[state_index, task]
                )
                next_mask = beam_mask[state_index] & (~bit)
                next_value = beam_value[state_index] + weights[task] * probability
                _fill_calibrated_stats(
                    task,
                    next_time,
                    next_energy,
                    next_mask,
                    points,
                    probs,
                    mu,
                    sigma,
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
                potential = _calibrated_potential_from_stats(
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
                if branch_aware_guidance and next_mask != 0:
                    # The ordinary expected-state Beam evaluates the concave
                    # fractional LP opportunity at E[S'].  Here we retain its
                    # one-state propagation cost but score the child with the
                    # exact Bernoulli success/failure mixture induced by the
                    # analytically calibrated completion mass.  Downstream
                    # task statistics remain those of the common mean state,
                    # so this is a targeted Jensen correction rather than a
                    # full scenario tree.
                    success_time = arrival_time + (
                        beam_expected_time[state_index, task]
                        / max(probability, 1e-14)
                    )
                    success_energy = arrival_energy + (
                        beam_expected_energy[state_index, task]
                        / max(probability, 1e-14)
                    )
                    failure_potential = _calibrated_potential_from_stats(
                        task,
                        arrival_time,
                        arrival_energy,
                        next_mask,
                        child_probability[child_count],
                        child_expected_time[child_count],
                        child_expected_energy[child_count],
                        ft,
                        fe,
                        weights,
                        pars,
                    )
                    success_potential = _calibrated_potential_from_stats(
                        task,
                        success_time,
                        success_energy,
                        next_mask,
                        child_probability[child_count],
                        child_expected_time[child_count],
                        child_expected_energy[child_count],
                        ft,
                        fe,
                        weights,
                        pars,
                    )
                    potential = (
                        probability * success_potential
                        + (1.0 - probability) * failure_potential
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
        selected_nodes = np.full(width, -1, np.int64)
        actual_keep = 0
        for keep in range(keep_count):
            best_index = -1
            best_score = -1e100
            best_value = -1e100
            best_resource = 1e100
            best_node = 10**9
            for pass_index in range(2):
                for child in range(child_count):
                    if child_score[child] <= -1e90:
                        continue
                    if pass_index == 0:
                        duplicate = False
                        for prior in range(keep):
                            if selected_nodes[prior] == child_node[child]:
                                duplicate = True
                                break
                        if duplicate:
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
                if best_index >= 0:
                    break
            if best_index < 0:
                break
            beam_node[keep] = child_node[best_index]
            beam_time[keep] = child_time[best_index]
            beam_energy[keep] = child_energy[best_index]
            beam_mask[keep] = child_mask[best_index]
            beam_value[keep] = child_value[best_index]
            beam_terminal_score[keep] = child_score[best_index]
            for task in range(1, n + 1):
                beam_probability[keep, task] = child_probability[best_index, task]
                beam_expected_time[keep, task] = child_expected_time[best_index, task]
                beam_expected_energy[keep, task] = child_expected_energy[best_index, task]
            selected_nodes[keep] = child_node[best_index]
            child_score[best_index] = -1e100
            actual_keep += 1
            if beam_value[keep] > best_terminal:
                best_terminal = beam_value[keep]
        beam_count = actual_keep
        if beam_count == 0:
            break
    if 0 < depth_limit < n and beam_count > 0:
        cutoff_value = beam_terminal_score[0]
        for state_index in range(1, beam_count):
            if beam_terminal_score[state_index] > cutoff_value:
                cutoff_value = beam_terminal_score[state_index]
        if cutoff_value > best_terminal:
            best_terminal = cutoff_value
    return best_terminal, expanded


@njit(cache=True)
def beam_future_calibrated_cached(
    node0,
    time0,
    energy0,
    mask0,
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
    max_depth,
):
    """Calibrated Beam with exact retained-state statistic reuse."""

    return _beam_future_calibrated_cached_impl(
        node0,
        time0,
        energy0,
        mask0,
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
        max_depth,
        False,
    )


@njit(cache=True)
def beam_future_calibrated_branch_cached(
    node0,
    time0,
    energy0,
    mask0,
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
    max_depth,
):
    """Cached Beam with success/failure expected LP guidance."""

    return _beam_future_calibrated_cached_impl(
        node0,
        time0,
        energy0,
        mask0,
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
        max_depth,
        True,
    )


@njit(cache=True)
def conditioned_top_calibrated(
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
    max_depth,
):
    """Single policy root recursion with exact completion mass.

    Feasible deterministic workload nodes preserve their individual successor
    states. Their relative quadrature weights are normalized only within the
    feasible region, while the total feasible weight is replaced by the exact
    LogNormal CDF mass. All infeasible workloads share the arrival without
    service transition and therefore need only one continuation evaluation.
    """

    n = len(weights) - 1
    all_mask = (np.uint64(1) << np.uint64(n)) - np.uint64(1)
    remaining_all = all_mask & (~visited_mask)
    best_task = 0
    best_value = 0.0
    expanded_total = 0

    for task in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(task - 1)
        if (remaining_all & bit) == 0:
            continue
        if not robust_core.visit_ok(
            node, t, e, task, ft, fe, deadline, pars
        ):
            continue
        arrival_t, arrival_e, dt, de = robust_core.resource_limits(
            node, t, e, task, ft, fe, deadline, pars
        )
        if dt <= 0.0 or de <= 0.0:
            continue
        capacity = robust_core.max_feasible_workload(
            dt, de, pars, aa[task], bb[task], dd[task], ee[task], 0
        )
        exact_mass = robust_core.lognormal_completion_probability(
            mu[task], sigma[task], capacity
        )
        if exact_mass <= 1e-14:
            continue

        remaining = remaining_all & (~bit)
        feasible_discrete_mass = 0.0
        feasible_value_sum = 0.0
        for branch in range(len(probs)):
            ok, service_t, service_e, _mode, _frequency, _cost = (
                robust_core.execute_virtual(
                    points[task, branch],
                    dt,
                    de,
                    pars,
                    aa[task],
                    bb[task],
                    dd[task],
                    ee[task],
                    0,
                )
            )
            if not ok:
                continue
            if remaining == 0:
                future = 0.0
                expanded = 0
            else:
                future, expanded = beam_future_calibrated(
                    task,
                    arrival_t + service_t,
                    arrival_e + service_e,
                    remaining,
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
                    max_depth,
                )
            expanded_total += expanded
            feasible_discrete_mass += probs[branch]
            feasible_value_sum += probs[branch] * (weights[task] + future)

        if feasible_discrete_mass > 1e-14:
            feasible_value = feasible_value_sum / feasible_discrete_mass
        else:
            if sigma[task] > 1e-12:
                z = (
                    math.log(max(capacity, 1e-300))
                    - mu[task]
                    - sigma[task] * sigma[task]
                ) / (sigma[task] * math.sqrt(2.0))
                truncated_first_moment = math.exp(
                    mu[task] + 0.5 * sigma[task] * sigma[task]
                ) * 0.5 * (1.0 + math.erf(z))
                representative = truncated_first_moment / max(exact_mass, 1e-14)
            else:
                representative = math.exp(mu[task])
            representative = min(
                max(representative, 1e-14), max(capacity * (1.0 - 1e-10), 1e-14)
            )
            ok, service_t, service_e, _mode, _frequency, _cost = (
                robust_core.execute_virtual(
                    representative,
                    dt,
                    de,
                    pars,
                    aa[task],
                    bb[task],
                    dd[task],
                    ee[task],
                    0,
                )
            )
            if not ok:
                representative = max(1e-14, 0.7 * capacity)
                ok, service_t, service_e, _mode, _frequency, _cost = (
                    robust_core.execute_virtual(
                        representative,
                        dt,
                        de,
                        pars,
                        aa[task],
                        bb[task],
                        dd[task],
                        ee[task],
                        0,
                    )
                )
            if not ok:
                continue
            if remaining == 0:
                future = 0.0
                expanded = 0
            else:
                future, expanded = beam_future_calibrated(
                    task,
                    arrival_t + service_t,
                    arrival_e + service_e,
                    remaining,
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
                    max_depth,
                )
            expanded_total += expanded
            feasible_value = weights[task] + future

        infeasible_mass = max(0.0, 1.0 - exact_mass)
        infeasible_value = 0.0
        if infeasible_mass > 1e-14 and remaining != 0:
            infeasible_value, expanded = beam_future_calibrated(
                task,
                arrival_t,
                arrival_e,
                remaining,
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
                max_depth,
            )
            expanded_total += expanded

        value = exact_mass * feasible_value + infeasible_mass * infeasible_value
        if value > best_value + 1e-12 or (
            abs(value - best_value) <= 1e-12
            and (best_task == 0 or task < best_task)
        ):
            best_task = task
            best_value = value

    return best_task, best_value, expanded_total


@njit(cache=True)
def bellman_action_consistent_top(
    node,
    t,
    e,
    visited_mask,
    root_points,
    root_probs,
    continuation_points,
    continuation_probs,
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
    max_depth,
):
    """Action consistent root recursion with one common continuation value.

    For every candidate task and deterministic root workload point, Skip, the
    normalized occupation minimizing Local action, and each feasible MEC are
    compared. Every action is evaluated by the same analytic probability
    calibrated finite depth Beam used by realized post reveal recourse.
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
        if not robust_core.visit_ok(node, t, e, task, ft, fe, deadline, pars):
            continue
        arrival_t, arrival_e, dt, de = robust_core.resource_limits(
            node, t, e, task, ft, fe, deadline, pars
        )
        if dt <= 0.0 or de <= 0.0:
            continue
        remaining = unvisited & (~bit)

        skip_future = 0.0
        if remaining != 0:
            skip_future, expanded = beam_future_calibrated(
                task,
                arrival_t,
                arrival_e,
                remaining,
                continuation_points,
                continuation_probs,
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
                max_depth,
            )
            expanded_total += expanded

        expected_value = 0.0
        for branch in range(len(root_probs)):
            workload = root_points[task, branch]
            branch_best = skip_future

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
                local_future = 0.0
                if remaining != 0:
                    local_future, expanded = beam_future_calibrated(
                        task,
                        arrival_t + local_t,
                        arrival_e + local_e,
                        remaining,
                        continuation_points,
                        continuation_probs,
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
                        max_depth,
                    )
                    expanded_total += expanded
                branch_best = max(branch_best, weights[task] + local_future)

            for mec in range(aa.shape[1]):
                if aa[task, mec] >= 1e90:
                    continue
                service_t = aa[task, mec] + bb[task, mec] * workload
                service_e = dd[task, mec] + ee[task, mec] * workload
                if service_t > dt + 1e-9 or service_e > de + 1e-9:
                    continue
                mec_future = 0.0
                if remaining != 0:
                    mec_future, expanded = beam_future_calibrated(
                        task,
                        arrival_t + service_t,
                        arrival_e + service_e,
                        remaining,
                        continuation_points,
                        continuation_probs,
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
                        max_depth,
                    )
                    expanded_total += expanded
                branch_best = max(branch_best, weights[task] + mec_future)

            expected_value += root_probs[branch] * branch_best

        if expected_value > best_value + 1e-12 or (
            abs(expected_value - best_value) <= 1e-12
            and (best_task == 0 or task < best_task)
        ):
            best_task = task
            best_value = expected_value

    return best_task, best_value, expanded_total
