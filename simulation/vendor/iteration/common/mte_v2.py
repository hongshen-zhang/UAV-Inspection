"""Second-generation MTE-Top and MTE-Sub planner.

The original implementation in ``mte_core.py`` remains unchanged so that all
paper baselines can still be reproduced.  This module adds three algorithmic
refinements without using any realized workload from an unvisited task:

1. Equal-probability quantile discretization of each lognormal workload prior.
   This is more stable for the discontinuous completion indicator than a very
   low-order Gauss-Hermite rule.
2. Scenario-conditioned recourse in MTE-Top.  Each possible workload revealed
   at the candidate first task creates its own post-task state, and the future
   route is estimated from that state.  The old code forced one common second
   task and then planned from an averaged state.
3. Bottleneck-aware MTE-Sub.  Feasible local/MEC actions are compared by a
   weighted normalized resource cost plus an optional max-utilization term,
   which discourages exhausting one resource while leaving the other idle.

All routines are deterministic for a fixed mission state and prior.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit
from scipy.special import ndtri

from .mte_core import lp_two_resource_exact


def quantile_workload_points(mu: np.ndarray, sigma: np.ndarray, order: int):
    """Return midpoint quantiles and equal weights for lognormal priors."""
    order = int(order)
    if order < 1:
        raise ValueError("order must be >= 1")
    q = (np.arange(order, dtype=np.float64) + 0.5) / float(order)
    z = ndtri(q)
    pts = np.zeros((len(mu), order), dtype=np.float64)
    for i in range(1, len(mu)):
        pts[i] = np.exp(mu[i] + sigma[i] * z)
    probs = np.full(order, 1.0 / order, dtype=np.float64)
    return pts, probs


@njit(cache=True)
def execute_virtual_v2(c, dt, de, pars, aa, bb, dd, ee, force_mode,
                       time_weight, energy_weight, balance_weight):
    """Solve the refined MTE-Sub action for one workload value.

    ``aa + bb*c`` and ``dd + ee*c`` are MEC time and onboard-energy models.
    ``force_mode`` is 0 (auto), 1 (local only), or 2 (MEC only).
    """
    if c <= 0.0 or dt <= 0.0 or de <= 0.0:
        return False, 0.0, 0.0, -1, 0.0, 1e100

    wt = max(time_weight, 1e-12)
    we = max(energy_weight, 1e-12)
    bw = max(balance_weight, 0.0)
    fmin = pars[2]
    fmax = pars[3]
    kappa = pars[4]
    ph = pars[5]

    best = 1e100
    best_t = 0.0
    best_e = 0.0
    best_mode = -1
    best_f = 0.0

    if force_mode != 2:
        lo = max(fmin, c / dt)
        hi = fmax
        if lo <= hi + 1e-12:
            if lo > hi:
                lo = hi

            # Local CPU energy is U-shaped; first intersect [lo, hi] with its
            # energy-feasible subinterval.
            f_energy = (ph / (2.0 * kappa)) ** (1.0 / 3.0)
            f_energy = min(hi, max(lo, f_energy))
            e_min = kappa * c * f_energy * f_energy + ph * c / f_energy
            if e_min <= de + 1e-9:
                e_lo = kappa * c * lo * lo + ph * c / lo
                if e_lo > de:
                    a = lo
                    b = f_energy
                    for _ in range(55):
                        x = 0.5 * (a + b)
                        ex = kappa * c * x * x + ph * c / x
                        if ex > de:
                            a = x
                        else:
                            b = x
                    lo = b

                e_hi = kappa * c * hi * hi + ph * c / hi
                if e_hi > de:
                    a = f_energy
                    b = hi
                    for _ in range(55):
                        x = 0.5 * (a + b)
                        ex = kappa * c * x * x + ph * c / x
                        if ex > de:
                            b = x
                        else:
                            a = x
                    hi = a

                # The objective is convex and piecewise smooth.  Its optimum
                # is at a boundary, a smooth-region stationary point, or the
                # time/energy-utilization equality kink.
                f_sum = ((wt * de + we * ph * dt) /
                         (2.0 * we * kappa * dt)) ** (1.0 / 3.0)
                f_time_region = (((wt + bw) * de + we * ph * dt) /
                                 (2.0 * we * kappa * dt)) ** (1.0 / 3.0)
                f_energy_region = ((wt * de + (we + bw) * ph * dt) /
                                   (2.0 * (we + bw) * kappa * dt)) ** (1.0 / 3.0)
                rhs = de / dt - ph
                f_equal = f_energy
                if rhs > 0.0:
                    f_equal = (rhs / kappa) ** (1.0 / 3.0)

                for f0 in (lo, hi, f_energy, f_sum, f_time_region,
                           f_energy_region, f_equal):
                    f = min(hi, max(lo, f0))
                    xt = c / f
                    xe = kappa * c * f * f + ph * xt
                    if xt <= dt + 1e-8 and xe <= de + 1e-8:
                        ut = xt / dt
                        ue = xe / de
                        cost = wt * ut + we * ue + bw * max(ut, ue)
                        if cost < best - 1e-12:
                            best = cost
                            best_t = xt
                            best_e = xe
                            best_mode = 0
                            best_f = f

    if force_mode != 1:
        for m in range(len(aa)):
            if aa[m] >= 1e90:
                continue
            xt = aa[m] + bb[m] * c
            xe = dd[m] + ee[m] * c
            if xt <= dt + 1e-9 and xe <= de + 1e-9:
                ut = xt / dt
                ue = xe / de
                cost = wt * ut + we * ue + bw * max(ut, ue)
                if cost < best - 1e-12:
                    best = cost
                    best_t = xt
                    best_e = xe
                    best_mode = m + 1
                    best_f = 0.0

    return (best_mode >= 0, best_t, best_e, best_mode, best_f, best)


@njit(cache=True)
def visit_ok_v2(node, t, e, i, ft, fe, deadline, pars):
    at = t + ft[node, i]
    ae = e + fe[node, i]
    return (at + ft[i, 0] <= pars[0] + 1e-9 and
            ae + fe[i, 0] <= pars[1] + 1e-9 and
            deadline[i] - at > 0.0)


@njit(cache=True)
def resource_limits_v2(node, t, e, i, ft, fe, deadline, pars):
    at = t + ft[node, i]
    ae = e + fe[node, i]
    dt = min(deadline[i] - at, pars[0] - at - ft[i, 0])
    de = pars[1] - ae - fe[i, 0]
    return at, ae, dt, de


@njit(cache=True)
def expected_task_stats_v2(node, t, e, i, points, probs, ft, fe, deadline,
                           pars, aa, bb, dd, ee, force_mode,
                           time_weight, energy_weight, balance_weight):
    """Completion probability and unconditional expected execution resources."""
    at, ae, dt, de = resource_limits_v2(node, t, e, i, ft, fe, deadline, pars)
    if dt <= 0.0 or de <= 0.0:
        return 0.0, 0.0, 0.0
    p = 0.0
    et = 0.0
    en = 0.0
    for g in range(len(probs)):
        ok, xt, xe, mode, f, cost = execute_virtual_v2(
            points[i, g], dt, de, pars, aa[i], bb[i], dd[i], ee[i],
            force_mode, time_weight, energy_weight, balance_weight,
        )
        if ok:
            pr = probs[g]
            p += pr
            et += pr * xt
            en += pr * xe
    return p, et, en


@njit(cache=True)
def remaining_potential_v2(node, t, e, mask, points, probs, ft, fe, weights,
                           deadline, pars, aa, bb, dd, ee, force_mode,
                           time_weight, energy_weight, balance_weight,
                           use_lp, tighten_return):
    """LP upper bound used only to rank beam states."""
    if not use_lp:
        return 0.0
    n = len(weights) - 1
    vals = np.zeros(n)
    time_cost = np.zeros(n)
    energy_cost = np.zeros(n)
    count = 0

    for k in range(1, n + 1):
        bit = np.uint64(1) << np.uint64(k - 1)
        if (mask & bit) == 0:
            continue
        p, et, en = expected_task_stats_v2(
            node, t, e, k, points, probs, ft, fe, deadline, pars,
            aa, bb, dd, ee, force_mode, time_weight, energy_weight,
            balance_weight,
        )
        value = weights[k] * p
        if value <= 1e-14:
            continue

        min_in_t = ft[node, k]
        min_in_e = fe[node, k]
        for u in range(1, n + 1):
            if u == k:
                continue
            bu = np.uint64(1) << np.uint64(u - 1)
            if (mask & bu) != 0:
                if ft[u, k] < min_in_t:
                    min_in_t = ft[u, k]
                if fe[u, k] < min_in_e:
                    min_in_e = fe[u, k]

        vals[count] = value
        time_cost[count] = min_in_t + et
        energy_cost[count] = min_in_e + en
        count += 1

    cap_t = pars[0] - t
    cap_e = pars[1] - e
    if tighten_return:
        # Every continuation eventually returns.  The smallest return cost
        # among possible terminal nodes preserves the upper-bound character.
        min_return_t = ft[node, 0]
        min_return_e = fe[node, 0]
        for k in range(1, n + 1):
            bit = np.uint64(1) << np.uint64(k - 1)
            if (mask & bit) != 0:
                if ft[k, 0] < min_return_t:
                    min_return_t = ft[k, 0]
                if fe[k, 0] < min_return_e:
                    min_return_e = fe[k, 0]
        cap_t -= min_return_t
        cap_e -= min_return_e

    return lp_two_resource_exact(cap_t, cap_e, vals, time_cost,
                                 energy_cost, count)


@njit(cache=True)
def beam_future_v2(node0, t0, e0, mask0, initial_value, points, probs, ft, fe,
                   weights, deadline, pars, aa, bb, dd, ee, force_mode,
                   time_weight, energy_weight, balance_weight, beam_width,
                   use_lp, lp_weight, tighten_return, min_completion_prob):
    """Expected-state beam rollout from a scenario-conditioned state."""
    if mask0 == 0:
        return initial_value, 0

    n = len(weights) - 1
    B = max(1, beam_width)
    bnode = np.zeros(B, np.int64)
    bt = np.zeros(B)
    be = np.zeros(B)
    bmask = np.zeros(B, np.uint64)
    bj = np.zeros(B)
    bnode[0] = node0
    bt[0] = t0
    be[0] = e0
    bmask[0] = mask0
    bj[0] = initial_value
    bcount = 1
    best_terminal = initial_value
    expanded = 0

    for depth in range(n):
        capacity = B * n
        cnode = np.zeros(capacity, np.int64)
        ct = np.zeros(capacity)
        ce = np.zeros(capacity)
        cmask = np.zeros(capacity, np.uint64)
        cj = np.zeros(capacity)
        cscore = np.full(capacity, -1e100)
        ccount = 0

        for h in range(bcount):
            if bj[h] > best_terminal:
                best_terminal = bj[h]
            for k in range(1, n + 1):
                bit = np.uint64(1) << np.uint64(k - 1)
                if (bmask[h] & bit) == 0:
                    continue
                if not visit_ok_v2(bnode[h], bt[h], be[h], k, ft, fe,
                                   deadline, pars):
                    continue

                p, et, en = expected_task_stats_v2(
                    bnode[h], bt[h], be[h], k, points, probs, ft, fe,
                    deadline, pars, aa, bb, dd, ee, force_mode,
                    time_weight, energy_weight, balance_weight,
                )
                if p <= max(1e-14, min_completion_prob):
                    continue

                nt = bt[h] + ft[bnode[h], k] + et
                ne = be[h] + fe[bnode[h], k] + en
                nm = bmask[h] & (~bit)
                nj = bj[h] + weights[k] * p
                potential = remaining_potential_v2(
                    k, nt, ne, nm, points, probs, ft, fe, weights, deadline,
                    pars, aa, bb, dd, ee, force_mode, time_weight,
                    energy_weight, balance_weight, use_lp, tighten_return,
                )

                cnode[ccount] = k
                ct[ccount] = nt
                ce[ccount] = ne
                cmask[ccount] = nm
                cj[ccount] = nj
                cscore[ccount] = nj + lp_weight * potential
                ccount += 1
                expanded += 1

        if ccount == 0:
            break

        nkeep = min(B, ccount)
        for keep in range(nkeep):
            best_idx = -1
            best_score = -1e100
            best_j = -1e100
            best_resource = 1e100
            best_node = 10 ** 9
            for q in range(ccount):
                if cscore[q] <= -1e90:
                    continue
                resource = (ct[q] / max(pars[0], 1e-12) +
                            ce[q] / max(pars[1], 1e-12))
                better = False
                if cscore[q] > best_score + 1e-12:
                    better = True
                elif abs(cscore[q] - best_score) <= 1e-12:
                    if cj[q] > best_j + 1e-12:
                        better = True
                    elif abs(cj[q] - best_j) <= 1e-12:
                        if resource < best_resource - 1e-12:
                            better = True
                        elif (abs(resource - best_resource) <= 1e-12 and
                              cnode[q] < best_node):
                            better = True
                if better:
                    best_idx = q
                    best_score = cscore[q]
                    best_j = cj[q]
                    best_resource = resource
                    best_node = cnode[q]

            if best_idx < 0:
                break
            bnode[keep] = cnode[best_idx]
            bt[keep] = ct[best_idx]
            be[keep] = ce[best_idx]
            bmask[keep] = cmask[best_idx]
            bj[keep] = cj[best_idx]
            cscore[best_idx] = -1e100
            if bj[keep] > best_terminal:
                best_terminal = bj[keep]
        bcount = nkeep

    return best_terminal, expanded


@njit(cache=True)
def top_choose_adaptive_v2(node, t, e, visited_mask, points, probs, ft, fe,
                           weights, deadline, pars, aa, bb, dd, ee, force_mode,
                           time_weight, energy_weight, balance_weight,
                           beam_width, use_lp, lp_weight, tighten_return,
                           min_completion_prob, risk_lambda):
    """Select the next task using scenario-conditioned first-stage recourse."""
    n = len(weights) - 1
    allmask = (np.uint64(1) << np.uint64(n)) - np.uint64(1)
    unvisited = allmask & (~visited_mask)

    best_i = 0
    best_score = 0.0
    best_expected = 0.0
    best_short = 0.0
    best_future = 0.0
    expanded_total = 0

    for i in range(1, n + 1):
        biti = np.uint64(1) << np.uint64(i - 1)
        if (unvisited & biti) == 0:
            continue
        if not visit_ok_v2(node, t, e, i, ft, fe, deadline, pars):
            continue

        at_i, ae_i, dt_i, de_i = resource_limits_v2(
            node, t, e, i, ft, fe, deadline, pars,
        )
        remaining = unvisited & (~biti)
        expected_total = 0.0
        expected_square = 0.0
        expected_short = 0.0

        for g in range(len(probs)):
            ok, xt, xe, mode, f, cost = execute_virtual_v2(
                points[i, g], dt_i, de_i, pars, aa[i], bb[i], dd[i], ee[i],
                force_mode, time_weight, energy_weight, balance_weight,
            )
            if ok:
                next_t = at_i + xt
                next_e = ae_i + xe
                immediate = weights[i]
            else:
                # Arrival has already occurred; an infeasible execution is
                # skipped, after which the online route is replanned.
                next_t = at_i
                next_e = ae_i
                immediate = 0.0

            if remaining == 0:
                total = immediate
                expanded = 0
            else:
                total, expanded = beam_future_v2(
                    i, next_t, next_e, remaining, immediate, points, probs,
                    ft, fe, weights, deadline, pars, aa, bb, dd, ee,
                    force_mode, time_weight, energy_weight, balance_weight,
                    beam_width, use_lp, lp_weight, tighten_return,
                    min_completion_prob,
                )
            expanded_total += expanded
            pr = probs[g]
            expected_total += pr * total
            expected_square += pr * total * total
            expected_short += pr * immediate

        variance = max(0.0, expected_square - expected_total * expected_total)
        score = expected_total - max(risk_lambda, 0.0) * math.sqrt(variance)
        expected_future = expected_total - expected_short

        if (score > best_score + 1e-12 or
            (abs(score - best_score) <= 1e-12 and
             (expected_total > best_expected + 1e-12 or
              (abs(expected_total - best_expected) <= 1e-12 and
               (best_i == 0 or i < best_i))))):
            best_i = i
            best_score = score
            best_expected = expected_total
            best_short = expected_short
            best_future = expected_future

    return best_i, best_score, best_short, best_future, expanded_total
