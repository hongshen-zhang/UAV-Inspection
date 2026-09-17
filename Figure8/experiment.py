#!/usr/bin/env python3
"""Run candidate-cap performance and serial per-decision timing experiments.

Quick check: python experiment.py --seeds 1 --workers 1 --caps 9
Plot new results: python plot.py --results results.csv --timing timing.csv
Dependencies: numpy, pandas, scipy, numba (Python 3.12 recommended).
Timing measures Top + Sub calls, excludes construction and JIT warmup,
and is expected to vary with the computer and its load.
"""
from __future__ import annotations
import os
import sys
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "simulation"))
import planner
from experiment_support import NOMINAL_CASE, get_case
from run_mean_workload_dp import (
    BASE_POLICY, MEAN_POLICY, SPEC, EnhancedPlanner, MeanWorkloadDP, audit_trace,
)

METHODS = ("Proposed", "Mean-workload DP")


class CapProposed(EnhancedPlanner):
    enhanced_spec = SPEC


class CapMean(MeanWorkloadDP):
    enhanced_spec = SPEC

    def __init__(self, bundle, policy):
        super().__init__(bundle, policy)
        # The original mean-control class identifies L=9; record the active L.
        self.continuation.signature = hashlib.sha256(json.dumps({
            "enhanced_spec": asdict(self.enhanced_spec),
            "continuation": policy.continuation.fingerprint,
        }, sort_keys=True).encode()).hexdigest()[:16]


class TimingMixin:
    def __init__(self, bundle, policy):
        self.timing_rows = []
        super().__init__(bundle, policy)

    def select_task(self, state):
        origin = int(state.node)
        used_time, used_energy = float(state.time_s), float(state.energy_kj)
        epoch = len(self.timing_rows)
        started = time.perf_counter_ns()
        task, detail = super().select_task(state)
        elapsed = time.perf_counter_ns() - started
        self.timing_rows.append(dict(
            epoch=epoch, origin_node=origin, mission_time_before_s=used_time,
            mission_energy_before_kj=used_energy, selected_task=int(task),
            decision_kind="task" if int(task) != 0 else "return",
            top_ns=elapsed, sub_ns=0, top_ms=elapsed / 1e6,
            sub_ms=0.0, decision_ms=elapsed / 1e6,
            execution_mode="return" if int(task) == 0 else "pending"))
        return task, detail

    def choose_actual_action(self, state, task, workload):
        started = time.perf_counter_ns()
        action = super().choose_actual_action(state, task, workload)
        elapsed = time.perf_counter_ns() - started
        row = self.timing_rows[-1]
        assert row["decision_kind"] == "task" and row["selected_task"] == int(task)
        row.update(sub_ns=elapsed, sub_ms=elapsed / 1e6,
                   decision_ms=(row["top_ns"] + elapsed) / 1e6,
                   execution_mode=action.mode, observed_workload_gcy=float(workload))
        return action


class TimedProposed(TimingMixin, CapProposed):
    pass


class TimedMean(TimingMixin, CapMean):
    pass


CLASSES = {"Proposed": CapProposed, "Mean-workload DP": CapMean}
TIMED_CLASSES = {"Proposed": TimedProposed, "Mean-workload DP": TimedMean}


def execute(method, cap, seed, timed=False):
    case = get_case(NOMINAL_CASE)
    bundle = case.build_bundle(seed)
    cls = (TIMED_CLASSES if timed else CLASSES)[method]
    cls.enhanced_spec = replace(SPEC, shortlist=cap)
    policy = MEAN_POLICY if method == "Mean-workload DP" else BASE_POLICY
    holder = {}

    def construct(given_bundle, given_policy):
        holder["instance"] = cls(given_bundle, given_policy)
        return holder["instance"]

    previous = planner.ConsistentACARPlanner
    planner.ConsistentACARPlanner = construct
    try:
        result = planner.run_unified_mission(bundle, policy, record_trace=True)
    finally:
        planner.ConsistentACARPlanner = previous
    instance = holder["instance"]
    audit_trace(bundle, result)
    assert instance.enhanced_spec == replace(SPEC, shortlist=cap)
    assert instance.continuation.spec == instance.enhanced_spec
    row = case.metadata(seed)
    row.update(result)
    row.update(seed=seed, method=method, L=cap, wcr=float(result["safe_mcr"]),
               dp_state_evaluations=int(instance.continuation.expansions))
    events = []
    if timed:
        events = instance.timing_rows
        for event in events:
            event.update(seed=seed, method=method, L=cap)
        task_events = [event for event in events if event["decision_kind"] == "task"]
        assert len(task_events) == int(result["visited_tasks"])
        assert all(event["top_ns"] > 0 and event["sub_ns"] > 0 for event in task_events)
        row.update(task_decision_count=len(task_events),
                   task_decision_total_ms=sum(event["decision_ms"] for event in task_events),
                   terminal_return_calls=len(events) - len(task_events))
    return row, events


def performance_worker(job):
    return execute(*job)[0]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=300)
    parser.add_argument("--seed-start", type=int, default=2026092000)
    parser.add_argument("--workers", type=int, default=3, help="Performance workers; timing is always serial.")
    parser.add_argument("--caps", type=int, nargs="+", choices=range(1, 12), default=list(range(1, 12)))
    parser.add_argument("--phase", choices=("performance", "timing", "both"), default="both")
    parser.add_argument("--timing-seeds", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", type=Path, default=HERE / "results.csv")
    parser.add_argument("--timing-output", type=Path, default=HERE / "timing.csv")
    parser.add_argument("--decisions-output", type=Path, default=HERE / "decisions.csv")
    args = parser.parse_args()
    if not 1 <= args.seeds <= 300 or args.workers < 1 or args.timing_seeds < 1 or args.warmups < 0:
        parser.error("Invalid seed, worker or warmup count.")
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    if min(seeds) < 2026092000 or max(seeds) > 2026092299:
        parser.error("Archived seeds range from 2026092000 through 2026092299.")
    caps = sorted(set(args.caps))
    performance = []
    if args.phase in ("performance", "both"):
        jobs = [(method, cap, seed) for cap in caps for seed in seeds for method in METHODS]
        if args.workers == 1:
            performance = [performance_worker(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                performance = list(pool.map(performance_worker, jobs))
        write_csv(args.output, performance)
    if args.phase in ("timing", "both"):
        timing_seeds = seeds[:min(args.timing_seeds, len(seeds))]
        reference = {(row["method"], int(row["L"]), int(row["seed"])): row for row in performance}
        # Compile shared JIT signatures once before the registered per-cell warmups.
        execute("Proposed", 3, timing_seeds[0], timed=True)
        missions, decisions = [], []
        cells = [(method, cap) for cap in caps for method in (METHODS if cap % 2 else METHODS[::-1])]
        for method, cap in cells:
            for index in range(args.warmups):
                execute(method, cap, timing_seeds[index % len(timing_seeds)], timed=True)
            for seed in timing_seeds:
                row, events = execute(method, cap, seed, timed=True)
                prior = reference.get((method, cap, seed))
                if prior:
                    assert row["route"] == prior["route"]
                    for key in ("wcr", "weighted_completed", "completed_tasks", "final_time_s", "final_energy_kj", "dp_state_evaluations"):
                        assert np.isclose(row[key], prior[key], rtol=0, atol=1e-8)
                missions.append(row)
                decisions.extend(events)
            print(f"Timed {method}, L={cap}: {len(timing_seeds)} missions", flush=True)
        write_csv(args.timing_output, missions)
        write_csv(args.decisions_output, decisions)


if __name__ == "__main__":
    main()
