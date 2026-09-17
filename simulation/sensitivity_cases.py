"""Exact paired input transformations and result summaries for Figures 9-12.

Input formulas are preserved from the original uncertainty, workload-scale,
and MEC experiments. Plotting imports only NumPy and the Python standard library.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
METHOD_NAMES = ("Proposed", "Weight Greedy", "Mean-workload Greedy", "Mean-workload DP",
                "Distribution-aware Myopic", "Nearest Neighbor")
SEEDS = tuple(range(2026092000, 2026092300))
BOOTSTRAP_SEED = 2026091510
BOOTSTRAPS = 20000


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("No result rows to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_case(destination, kind, value, seeds):
    """Copy the frozen sample bank and transform just the selected variable."""
    from experiment_support import copy_case, get_case
    updates = {}
    if kind == "mec":
        updates["mec_cpu_ghz"] = [float(value)] * 3
    elif kind == "workload":
        # Task CSV values are authoritative; this field is generator metadata.
        updates["workload_scale"] = 2.5 * float(value)
    elif kind != "uncertainty":
        raise ValueError(f"Unknown sensitivity kind: {kind}")
    folder = copy_case(destination, case_id=f"{kind}_{value:g}",
                       physics_updates=updates, seeds=seeds)
    rows = read_rows(folder / "tasks.csv")
    if kind == "uncertainty" and value != 1.0:
        base_z = {(int(r["seed"]), int(r["task_id"])):
                  (math.log(float(r['workload_gcy'])) - float(r['mu'])) / float(r['sigma'])
                  for r in rows}
        for row in rows:
            sigma = float(row["sigma"]) * float(value)
            mean = float(row["mean_workload_gcy"])
            mu = math.log(mean) - sigma * sigma / 2
            if value == 0:
                workload = row["mean_workload_gcy"]
            else:
                z = base_z[int(row["seed"]), int(row["task_id"])]
                workload = f"{math.exp(mu + sigma * z):.17g}"
            row.update(mu=f"{mu:.17g}", sigma=f"{sigma:.17g}", workload_gcy=workload)
    elif kind == "workload" and value != 1.0:
        for row in rows:
            row.update(mu=f"{float(row['mu']) + math.log(value):.17g}",
                       mean_workload_gcy=f"{float(row['mean_workload_gcy']) * value:.17g}",
                       workload_gcy=f"{float(row['workload_gcy']) * value:.17g}")
    write_rows(folder / "tasks.csv", rows)
    payload = json.loads((folder / "config.json").read_text())
    payload["axis"] = kind
    payload["axis_value"] = float(value)
    (folder / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
    nodes, links = read_rows(folder / "nodes.csv"), read_rows(folder / "links.csv")
    summaries = []
    for seed in seeds:
        content = dict(physics=payload["physics"], nodes=nodes, links=links,
                       tasks=[r for r in rows if int(r["seed"]) == seed])
        digest = hashlib.sha256(json.dumps(content, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        summaries.append(dict(seed=seed, mission_fingerprint=digest))
    write_rows(folder / "mission_summary.csv", summaries)
    get_case.cache_clear()
    return folder


def sensitivity_job(job):
    from experiment_support import run_mission
    folder, point, method, seed = job
    engine_method = "Priority Greedy" if method == "Weight Greedy" else method
    row = run_mission(folder, seed, engine_method, record_trace=True)
    row.update(point=point, method=method)
    return row


def execute_jobs(jobs, worker, workers, output):
    """Use processes because the original simulator dispatch hook is global."""
    rows = []
    try:
        if workers == 1:
            iterator = map(worker, jobs)
            for row in iterator:
                rows.append(row)
                if len(rows) % 100 == 0:
                    print(f"Completed {len(rows)}/{len(jobs)} missions", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for row in pool.map(worker, jobs, chunksize=1):
                    rows.append(row)
                    if len(rows) % 100 == 0:
                        print(f"Completed {len(rows)}/{len(jobs)} missions", flush=True)
    except BaseException:
        if rows:
            write_rows(Path(output).with_suffix(".partial.csv"), rows)
        raise
    write_rows(output, rows)
    return rows


def run_sensitivity(kind, points, seeds, workers, output):
    output = Path(output).resolve()
    input_root = output.parent / (output.stem + "_inputs")
    cases = [(point, make_case(input_root / f"{point:g}", kind, point, seeds))
             for point in points]
    jobs = [(str(folder), point, method, seed) for point, folder in cases
            for seed in seeds for method in METHOD_NAMES]
    started = time.perf_counter()
    rows = execute_jobs(jobs, sensitivity_job, workers, output)
    meta = dict(experiment=kind, status="complete", points=points,
                seeds=list(seeds), methods=list(METHOD_NAMES), rows=len(rows),
                paper_seed_count=300, uses_all_paper_seeds=len(seeds) == 300,
                workers=workers, elapsed_seconds=time.perf_counter()-started,
                inputs_directory=input_root.name, result_file=output.name,
                recorded_traces=True, simulations_performed=True)
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved {len(rows)} real mission outcomes to {output}")


def result_matrices(path, keys, series_names, key_field="point", series_field="method"):
    """Validate a complete paired grid before calculating sampling intervals."""
    rows = read_rows(path)
    convert = str if key_field == "setting" else float
    available = {convert(r[key_field]) for r in rows}
    keys = tuple(keys) if keys is not None else tuple(sorted(available))
    if available != set(keys):
        raise ValueError("Result settings differ from the requested plot")
    lookup = {}
    for row in rows:
        key = (convert(row[key_field]), row[series_field], int(row["seed"]))
        if key in lookup:
            raise ValueError(f"Duplicate mission: {key}")
        lookup[key] = row
    seeds = tuple(sorted({int(row["seed"]) for row in rows}))
    expected = {(k, name, seed) for k in keys for name in series_names for seed in seeds}
    if set(lookup) != expected:
        raise ValueError("Results must contain all plotted methods with the same seed set at every setting")
    values = np.array([[[float(lookup[k, name, seed]["wcr"]) for name in series_names]
                         for k in keys] for seed in seeds])
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("WCR must be a finite fraction in [0, 1]")
    mean = 100 * values.mean(axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((BOOTSTRAPS, len(keys), len(series_names)))
    for first in range(0, BOOTSTRAPS, 100):
        last = min(first + 100, BOOTSTRAPS)
        indices = rng.integers(0, len(seeds), size=(last-first, len(seeds)))
        draws[first:last] = 100 * values[indices].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975], axis=0)
    # Repeated deterministic values have exactly zero sampling variation.
    deterministic = np.all(values == values[:1], axis=0)
    mean[deterministic] = (100 * values[0])[deterministic]
    low[deterministic] = high[deterministic] = mean[deterministic]
    low, high = np.minimum(low, mean), np.maximum(high, mean)
    return keys, (mean, low, high), lookup, seeds


def share_matrices(keys, lookup, seeds):
    """Mode shares use total completed tasks across missions as denominator."""
    means = []
    for key in keys:
        rows = [lookup[key, "Proposed", seed] for seed in seeds]
        local = sum(float(r["local_actions"]) for r in rows)
        mec = sum(float(r["mec_actions"]) for r in rows)
        completed = sum(float(r["completed_tasks"]) for r in rows)
        if not math.isclose(local + mec, completed) or completed <= 0:
            raise ValueError("Mode shares require consistent nonzero completed task counts")
        means.append([100*local/completed, 100*mec/completed])
    means = np.asarray(means)
    # No confidence intervals are displayed in the original mode-share panel.
    return means, means.copy(), means.copy()
