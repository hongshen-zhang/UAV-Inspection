#!/usr/bin/env python3
"""Run the five Figure 12 interventions under the three original budgets.

Default: 5 variants x 3 settings x all 300 paired seeds = 4,500 real missions.
Smoke: python experiment.py --seeds 1 --workers 1 --settings default
Each raw result includes the route, action trace, and physical feasibility audit.
TopMyopic changes only task selection; no-proactive-skip changes both forecasting
and actual execution. The readable intervention source is in adjacent modules.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "simulation"))
from sensitivity_cases import SEEDS, execute_jobs

SETTINGS = {"default": (2200, 620), "larger_time": (2600, 620), "lower_energy": (2200, 380)}
VARIANTS = ("proposed", "mean_workload_dp", "no_priority", "top_myopic", "no_proactive_skip")
LABELS = {"proposed": "Proposed", "mean_workload_dp": "Mean-workload DP",
          "no_priority": "No priority factor", "top_myopic": "Myopic Top",
          "no_proactive_skip": "No proactive skip"}


def worker(job):
    from experiment_support import get_case, run_mission
    from run_mean_workload_dp import BASE_POLICY
    from batch_adapter import scenario_digest
    import planner
    import ablation_base
    import top_variants
    import sub_variants
    folder, setting, variant, seed = job
    case = get_case(folder)
    if variant in ("proposed", "mean_workload_dp"):
        row = run_mission(case, seed, LABELS[variant], record_trace=True)
    else:
        cls = (sub_variants.NoProactiveSkipPlanner if variant == "no_proactive_skip"
               else top_variants.CLASSES[variant])
        bundle = case.build_bundle(seed)
        holder = {}
        def construct(b, p):
            holder["planner"] = cls(b, p)
            return holder["planner"]
        previous = planner.ConsistentACARPlanner
        try:
            planner.ConsistentACARPlanner = construct
            result = planner.run_unified_mission(bundle, BASE_POLICY, record_trace=True)
        finally:
            planner.ConsistentACARPlanner = previous
        instance = holder["planner"]
        row = case.metadata(seed)
        row.update(result)
        row.update(wcr=float(row["safe_mcr"]), scenario_digest=scenario_digest(bundle.iteration))
        row.update(ablation_base.validate_trace(bundle, row))
        if variant == "no_proactive_skip":
            row.update(sub_variants.audit_actions(instance, row))
            row["intervention"] = sub_variants.DEFINITIONS[variant]
        else:
            row["intervention"] = json.dumps(top_variants.HOOKS[variant], sort_keys=True)
            row["first_screen"] = json.dumps(getattr(instance.continuation, "first_screen", None))
            row["sub_full_continuation_calls"] = instance.continuation.calls
    row.update(setting=setting, variant_id=variant, method=LABELS[variant], seed=seed,
               t_s=SETTINGS[setting][0], e_kj=SETTINGS[setting][1])
    return row


def main():
    from experiment_support import copy_case
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=300, choices=range(1, 301), metavar="1..300")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--output", type=Path, default=HERE / "results.csv")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if len(set(args.settings)) != len(args.settings) or len(set(args.variants)) != len(args.variants):
        parser.error("Repeated settings or variants are not allowed")
    output = args.output.resolve()
    seeds = SEEDS[:args.seeds]
    input_root = output.parent / (output.stem + "_inputs")
    jobs = []
    for setting in args.settings:
        t, e = SETTINGS[setting]
        folder = copy_case(input_root / setting, case_id=setting,
                           physics_updates={"t_max": t, "e_max_kj": e}, seeds=seeds)
        jobs.extend((str(folder), setting, variant, seed)
                    for seed in seeds for variant in args.variants)
    started = time.perf_counter()
    rows = execute_jobs(jobs, worker, args.workers, output)
    meta = dict(status="complete", seeds=list(seeds), settings=args.settings,
                variants=args.variants, rows=len(rows), paper_seed_count=300,
                uses_all_paper_seeds=len(seeds) == 300, simulations_performed=True,
                elapsed_seconds=time.perf_counter()-started,
                inputs_directory=input_root.name, result_file=output.name)
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved {len(rows)} real ablation outcomes to {output}")


if __name__ == "__main__":
    main()
