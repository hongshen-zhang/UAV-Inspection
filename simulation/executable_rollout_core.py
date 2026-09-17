"""Executable continuation kernels for rollout policy improvement.

The continuation policy is fixed by four transparent route score parameters.
It chooses a route from distributions only and reads a scenario workload only
after reaching that task.  Every returned value is therefore the reward of a
feasible nonanticipative policy, rather than an LP or perfect information upper
bound.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from batch_adapter import robust_core


@njit(cache=True)
def _nearby_priority(task, mask, ft, weights):
    """Return a bounded three neighbour opportunity score."""

    first = 0.0
    second = 0.0
    third = 0.0
    for other in range(1, len(weights)):
        bit = np.uint64(1) << np.uint64(other - 1)
        if other == task or (mask & bit) == 0:
            continue
        value = weights[other] / (1.0 + ft[task, other] / 240.0)
        if value > first:
            third = second
            second = first
            first = value
        elif value > second:
            third = second
            second = value
        elif value > third:
            third = value
    # Priorities in the generated experiments lie on the same scale as the
    # original rollout controller.  The divisor keeps this a bounded local
    # correction instead of a second reward objective.
    return (first + second + third) / 45.0


@njit(cache=True)
def base_route_score(
    node,
    time_s,
    energy_kj,
    task,
    mask,
    points,
    probabilities,
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
    cost_floor,
    cost_power,
    cluster_weight,
    deadline_weight,
):
    """Score one executable base policy route decision.

    The numerator is the exact LogNormal expected completion reward.  The
    denominator is the normalized detour and service occupation.  A deadline
    multiplier and a small nearby task term can be enabled explicitly.
    """

    bit = np.uint64(1) << np.uint64(task - 1)
    if (mask & bit) == 0:
        return -1e100
    if not robust_core.visit_ok(
        node, time_s, energy_kj, task, ft, fe, deadline, pars
    ):
        return -1e100
    probability, expected_time, expected_energy = (
        robust_core.calibrated_task_stats(
            node,
            time_s,
            energy_kj,
            task,
            points,
            probabilities,
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
    )
    if probability <= 1e-14:
        return -1e100

    remaining_time = pars[0] - time_s - ft[node, 0]
    remaining_energy = pars[1] - energy_kj - fe[node, 0]
    if remaining_time <= 1e-12 or remaining_energy <= 1e-12:
        return -1e100
    detour_time = ft[node, task] + ft[task, 0] - ft[node, 0]
    detour_energy = fe[node, task] + fe[task, 0] - fe[node, 0]
    if detour_time < 0.0:
        detour_time = 0.0
    if detour_energy < 0.0:
        detour_energy = 0.0
    occupation = (
        (detour_time + expected_time) / remaining_time
        + (detour_energy + expected_energy) / remaining_energy
    )
    denominator = (max(cost_floor, 1e-12) + max(occupation, 0.0)) ** max(
        cost_power, 0.0
    )
    value = weights[task] * probability / max(denominator, 1e-12)

    arrival_time = time_s + ft[node, task]
    available_time = min(
        deadline[task] - arrival_time,
        pars[0] - arrival_time - ft[task, 0],
    )
    urgency = 1.0 - max(0.0, available_time) / max(remaining_time, 1e-12)
    if urgency < 0.0:
        urgency = 0.0
    if urgency > 1.0:
        urgency = 1.0
    value *= 1.0 + max(deadline_weight, 0.0) * urgency
    if cluster_weight > 0.0:
        value += (
            cluster_weight
            * weights[task]
            * probability
            * _nearby_priority(task, mask, ft, weights)
        )
    return value


@njit(cache=True)
def base_route_choice(
    node,
    time_s,
    energy_kj,
    mask,
    points,
    probabilities,
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
    cost_floor,
    cost_power,
    cluster_weight,
    deadline_weight,
):
    """Return the fixed base policy action and its score."""

    best_task = 0
    best_score = -1e100
    for task in range(1, len(weights)):
        score = base_route_score(
            node,
            time_s,
            energy_kj,
            task,
            mask,
            points,
            probabilities,
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
            cost_floor,
            cost_power,
            cluster_weight,
            deadline_weight,
        )
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and score > -1e90
            and (best_task == 0 or task < best_task)
        ):
            best_task = task
            best_score = score
    return best_task, best_score


@njit(cache=True)
def candidate_shortlist(
    node,
    time_s,
    energy_kj,
    mask,
    points,
    probabilities,
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
    width,
    cost_floor,
    cost_power,
    cluster_weight,
    deadline_weight,
):
    """Return the top fixed base policy candidates in deterministic order."""

    limit = max(1, min(int(width), len(weights) - 1))
    tasks = np.zeros(limit, dtype=np.int64)
    scores = np.full(limit, -1e100)
    used = np.zeros(len(weights), dtype=np.uint8)
    count = 0
    for slot in range(limit):
        best_task = 0
        best_score = -1e100
        for task in range(1, len(weights)):
            if used[task] != 0:
                continue
            score = base_route_score(
                node,
                time_s,
                energy_kj,
                task,
                mask,
                points,
                probabilities,
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
                cost_floor,
                cost_power,
                cluster_weight,
                deadline_weight,
            )
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12
                and score > -1e90
                and (best_task == 0 or task < best_task)
            ):
                best_task = task
                best_score = score
        if best_task == 0 or best_score <= -1e90:
            break
        tasks[count] = best_task
        scores[count] = best_score
        used[best_task] = 1
        count += 1
    return tasks, scores, count


@njit(cache=True)
def rollout_value(
    node,
    time_s,
    energy_kj,
    mask,
    scenarios,
    scenario_probabilities,
    points,
    probabilities,
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
    cost_floor,
    cost_power,
    cluster_weight,
    deadline_weight,
    max_steps,
):
    """Expected reward of the fixed feasible nonanticipative base policy."""

    if mask == 0:
        return 0.0, 0
    expected_reward = 0.0
    expanded = 0
    for scenario_index in range(scenarios.shape[0]):
        scenario_node = node
        scenario_time = time_s
        scenario_energy = energy_kj
        scenario_mask = mask
        reward = 0.0
        for _step in range(max(0, int(max_steps))):
            if scenario_mask == 0:
                break
            task, _score = base_route_choice(
                scenario_node,
                scenario_time,
                scenario_energy,
                scenario_mask,
                points,
                probabilities,
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
                cost_floor,
                cost_power,
                cluster_weight,
                deadline_weight,
            )
            if task == 0:
                break
            arrival_time, arrival_energy, available_time, available_energy = (
                robust_core.resource_limits(
                    scenario_node,
                    scenario_time,
                    scenario_energy,
                    task,
                    ft,
                    fe,
                    deadline,
                    pars,
                )
            )
            bit = np.uint64(1) << np.uint64(task - 1)
            scenario_mask &= ~bit
            scenario_node = task
            scenario_time = arrival_time
            scenario_energy = arrival_energy
            if available_time > 0.0 and available_energy > 0.0:
                ok, service_time, service_energy, _mode, _frequency, _cost = (
                    robust_core.execute_virtual(
                        scenarios[scenario_index, task],
                        available_time,
                        available_energy,
                        pars,
                        aa[task],
                        bb[task],
                        dd[task],
                        ee[task],
                        0,
                    )
                )
                if ok:
                    scenario_time += service_time
                    scenario_energy += service_energy
                    reward += weights[task]
            expanded += 1
        expected_reward += scenario_probabilities[scenario_index] * reward
    return expected_reward, expanded
