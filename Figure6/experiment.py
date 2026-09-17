#!/usr/bin/env python3
"""Run the Figure 6 experiment; then: python plot.py --results results.csv.

Quick check: python experiment.py --seeds 1 --workers 1 --budgets 2200
Dependencies: numpy, pandas, scipy, numba (Python 3.12 recommended).
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "simulation"))
from experiment_support import NOMINAL_CASE, METHODS, get_case, run_mission, copy_case

BUDGETS = (800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200)
PARAMETER = 't_max'


def worker(job):
    case_path, seed, budget, methods = job
    case = get_case(case_path)
    rows = []
    for method in methods:
        row = run_mission(case, seed, method, record_trace=True)
        if budget is not None:
            row.update(budget=float(budget), axis='time')
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=300, help="Number of paired seeds (1 to 300).")
    parser.add_argument("--seed-start", type=int, default=2026092000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--budgets", type=float, nargs="+", choices=BUDGETS, default=BUDGETS)
    parser.add_argument("--output", type=Path, default=HERE / "results.csv")
    args = parser.parse_args()
    if not 1 <= args.seeds <= 300 or args.workers < 1:
        parser.error("Use 1-300 seeds and at least one worker.")
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    if min(seeds) < 2026092000 or max(seeds) > 2026092299:
        parser.error("The archived workload seeds are 2026092000 through 2026092299.")
    budgets = args.budgets
    with tempfile.TemporaryDirectory(prefix="uav-figure6-") as directory:
        jobs = []
        for budget in budgets:
            case_path = Path(directory) / ("nominal" if budget is None else "budget_" + str(budget))
            updates = {} if budget is None else {PARAMETER: float(budget)}
            copy_case(case_path, case_id="nominal" if budget is None else "t" + f"{budget:g}",
                      physics_updates=updates, seeds=seeds)
            jobs.extend((str(case_path), seed, budget, args.methods) for seed in seeds)
        rows = []
        if args.workers == 1:
            for job in jobs:
                rows.extend(worker(job))
                print(f"{len(rows)} / {len(jobs) * len(args.methods)} outcomes", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(worker, jobs):
                    rows.extend(result)
                    print(f"{len(rows)} / {len(jobs) * len(args.methods)} outcomes", flush=True)
    rows.sort(key=lambda row: (float(row.get("budget", 0)), int(row["seed"]), row["method"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} outcomes to {args.output}")


if __name__ == "__main__":
    main()
