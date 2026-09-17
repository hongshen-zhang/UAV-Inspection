#!/usr/bin/env python3
"""Evaluate the matched mean-workload control of the frozen Proposed DP."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESEARCH = HERE

import planner
from batch_adapter import BatchCase
from config import load_unified_policies
from enhanced_dp import EnhancedPlanner, EnhancedSpec

BASE_POLICY = load_unified_policies(RESEARCH / 'policies.json')['CMSACR']
MEAN_POLICY = replace(BASE_POLICY, continuation=replace(
    BASE_POLICY.continuation, support_rule='mean', support_order=1))
SPEC = EnhancedSpec(shortlist=9, time_step=25., energy_step=7.5,
    quadrature=3, tail_weight=0., tail_depth=6, cost_power=.5, rounding=1,
    mode='both', arrival_rule='original', integration=0)
METHOD = 'Mean-workload DP'


class MeanWorkloadDP(EnhancedPlanner):
    enhanced_spec = SPEC

    def __init__(self, bundle, policy):
        super().__init__(bundle, policy)
        original_mu, original_sigma = bundle.robust_arrays[2:4]
        expected = np.exp(original_mu + .5 * original_sigma**2)
        assert policy.continuation.support_rule == 'mean'
        assert self.continuation.points.shape == (len(expected), 1)
        assert np.all(self.sigma == 0.)
        assert np.allclose(np.exp(self.mu), expected, rtol=1e-12)
        assert np.allclose(self.price_points[:, 0], expected, rtol=1e-12)
        assert np.array_equal(self.price_points, self.continuation.points)
        assert np.array_equal(self.price_probabilities, np.ones(1))
        assert np.array_equal(self.continuation.base.mu, self.mu)
        assert np.array_equal(self.continuation.base.sigma, self.sigma)
        # Include the support in audit identifiers; the DP settings are shared.
        identity = {'enhanced_spec': asdict(SPEC),
                    'continuation': policy.continuation.fingerprint}
        self.continuation.signature = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]

    def select_task(self, state):
        task, detail = super().select_task(state)
        detail['selection_mode'] = 'mean_workload_dp'
        return task, detail


def audit_trace(bundle, row):
    """Check the actual sampled workloads, completions and physical constraints."""
    mission, cfg = bundle.iteration, bundle.iteration.cfg
    remaining = set(range(1, cfg.n_tasks + 1))
    node, time_s, energy_kj = 0, 0., 0.
    weight, completed, skipped, local, mec = 0., 0, 0, 0, 0
    events = json.loads(row['trace'])
    for event in events:
        task = int(event['task'])
        assert task in remaining
        item = mission.tasks[task - 1]
        assert event['route_continuation_signature'] == event['arrival_continuation_signature']
        assert event['revealed_workload_gcy'] == item.workload_gcy
        assert event['time_s'] >= time_s + mission.flight_time[node, task] - 1e-7
        assert event['energy_kj'] >= energy_kj + mission.flight_energy_kj[node, task] - 1e-7
        if event['mode'] == 'skip':
            skipped += 1
            assert np.isclose(event['time_s'], time_s + mission.flight_time[node, task])
            assert np.isclose(event['energy_kj'], energy_kj + mission.flight_energy_kj[node, task])
        else:
            completed += 1
            weight += item.weight
            assert event['time_s'] <= item.deadline_s + 1e-7
            if event['mode'] == 'local':
                local += 1
                assert cfg.local_f_min_ghz - 1e-9 <= event['frequency_ghz'] <= cfg.local_f_max_ghz + 1e-9
            elif event['mode'] == 'mec':
                mec += 1
                server = event['mec']
                assert item.ul_rates[server] >= cfg.ul_min_mbps
                assert item.dl_rates[server] >= cfg.dl_min_mbps
            else:
                raise AssertionError(event['mode'])
        assert event['time_s'] + mission.flight_time[task, 0] <= cfg.t_max + 1e-7
        assert event['energy_kj'] + mission.flight_energy_kj[task, 0] <= cfg.e_max_kj + 1e-7
        remaining.remove(task)
        node, time_s, energy_kj = task, event['time_s'], event['energy_kj']
    assert row['return_success'] == 1 and row['max_constraint_residual'] <= 1e-7
    assert row['completed_tasks'] == completed and row['skipped_tasks'] == skipped
    assert row['visited_tasks'] == completed + skipped == len(events)
    assert row['local_actions'] == local and row['mec_actions'] == mec
    assert np.isclose(row['weighted_completed'], weight)
    assert np.isclose(row['safe_mcr'], weight / row['total_weight'])
    assert np.isclose(row['final_time_s'], time_s + mission.flight_time[node, 0])
    assert np.isclose(row['final_energy_kj'], energy_kj + mission.flight_energy_kj[node, 0])
    assert row['route_arrival_signature_match'] == 1
    return len(events)


