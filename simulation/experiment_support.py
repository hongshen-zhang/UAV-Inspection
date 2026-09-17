"""Shared experiment inputs and the six planners shown in the archived figures.

The numerical kernels and planner classes are preserved from the original
experiments. This module only provides portable paths and experiment plumbing.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import planner
from batch_adapter import BatchCase, scenario_digest
from run_mean_workload_dp import (
    BASE_POLICY, MEAN_POLICY, SPEC, MeanWorkloadDP, EnhancedPlanner, audit_trace,
)
from run_baselines_formal import (
    PriorityPlanner, MeanWorkloadPlanner, MyopicPlanner,
)
from run_additional_baselines import NearestNeighborPlanner

HERE = Path(__file__).resolve().parent
NOMINAL_CASE = HERE / "nominal_case"
POLICY_PATH = HERE / "policies.json"
METHODS = {
    "Proposed": (EnhancedPlanner, BASE_POLICY),
    "Priority Greedy": (PriorityPlanner, BASE_POLICY),
    "Mean-workload Greedy": (MeanWorkloadPlanner, BASE_POLICY),
    "Mean-workload DP": (MeanWorkloadDP, MEAN_POLICY),
    "Distribution-aware Myopic": (MyopicPlanner, BASE_POLICY),
    "Nearest Neighbor": (NearestNeighborPlanner, BASE_POLICY),
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty experiment input table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_case(destination, case_id=None, physics_updates=None, seeds=None):
    """Copy the frozen paired inputs; optionally change budgets or keep seeds.

    Workloads, task deadlines, public distributions, positions, and link rates
    remain fixed. Sensitivity experiments explicitly transform the relevant
    inputs in their own source code after this helper returns.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    payload = json.loads((NOMINAL_CASE / "config.json").read_text())
    case_id = str(case_id or payload["case_id"])
    payload["case_id"] = case_id
    payload["physics"].update(copy.deepcopy(physics_updates or {}))
    payload["geometry_source"] = "nodes.csv"
    tasks = read_csv(NOMINAL_CASE / "tasks.csv")
    seed_set = None if seeds is None else {int(seed) for seed in seeds}
    if seed_set is not None:
        available = {int(row["seed"]) for row in tasks}
        if not seed_set or not seed_set <= available:
            raise ValueError("Seeds must be a nonempty subset of the 300 paired seeds")
        tasks = [row for row in tasks if int(row["seed"]) in seed_set]
    for row in tasks:
        row["case_id"] = case_id
    nodes = read_csv(NOMINAL_CASE / "nodes.csv")
    links = read_csv(NOMINAL_CASE / "links.csv")
    grouped = {}
    for row in tasks:
        grouped.setdefault(int(row["seed"]), []).append(row)
    summaries = []
    for seed, rows in sorted(grouped.items()):
        content = dict(physics=payload["physics"], nodes=nodes, links=links, tasks=rows)
        digest = hashlib.sha256(json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()).hexdigest()
        summaries.append(dict(seed=seed, mission_fingerprint=digest))
    (destination / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
    for filename, rows in [("tasks.csv", tasks), ("nodes.csv", nodes),
                           ("links.csv", links), ("mission_summary.csv", summaries)]:
        write_csv(destination / filename, rows)
    get_case.cache_clear()
    return destination


@lru_cache(maxsize=32)
def get_case(path):
    return BatchCase(Path(path))


def run_mission(case, seed, method, record_trace=True):
    """Run one real mission with the original policy and physical executor.

    Planners use a module-level dispatch hook in the frozen implementation;
    parallel experiments therefore use processes rather than threads.
    """
    if not isinstance(case, BatchCase):
        case = get_case(case)
    cls, policy = METHODS[method]
    previous = planner.ConsistentACARPlanner
    previous_spec = EnhancedPlanner.enhanced_spec
    try:
        EnhancedPlanner.enhanced_spec = SPEC
        planner.ConsistentACARPlanner = cls
        bundle = case.build_bundle(int(seed))
        row = case.metadata(int(seed))
        row.update(planner.run_unified_mission(bundle, policy, record_trace=record_trace))
        row.update(seed=int(seed), method=method, wcr=row["safe_mcr"],
                   config_id="U9" if method == "Proposed" else method,
                   scenario_digest=scenario_digest(bundle.iteration))
        if record_trace:
            row["audited_arrival_actions"] = audit_trace(bundle, row)
        return row
    finally:
        planner.ConsistentACARPlanner = previous
        EnhancedPlanner.enhanced_spec = previous_spec
