"""Run the current experiments from model parameters and random seeds.

No recorded observations or input archives are required. CSV results and run
metadata are generated locally. The numerical kernels are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np

from .inputs import (ABLATION_VARIANTS, build_case, generate_workload,
                     study_config, study_points, case_seeds)

METHODS = ('Proposed', 'Weight Greedy', 'Mean-workload Greedy',
           'Mean-workload DP', 'Distribution-aware Myopic', 'Nearest Neighbor',
           'IO', 'Rollout', 'ADAPT')
ABLATIONS = {method: variant for variant, method in ABLATION_VARIANTS.items()}
NUMERICAL = {'Default': ('default', {}),
             'DP bins G=1': ('dp_g1', {'quadrature': 1}),
             'DP bins G=5': ('dp_g5', {'quadrature': 5}),
             'Support H=3': ('support_h3', {'support_order': 3}),
             'Support H=15': ('support_h15', {'support_order': 15}),
             'Time grid 50 s': ('time_grid50', {'time_step': 50.0}),
             'Energy grid 15 kJ': ('energy_grid15', {'energy_step': 15.0})}
FIELDS = ['study', 'method', 'point', 'seed', 'setting', 'variant_id', 'L',
          'group_id', 'priority_config', 'wcr', 'weighted_completed', 'total_weight',
          'completed_tasks', 'visited_tasks', 'skipped_tasks', 'local_actions',
          'mec_actions', 'final_time_s', 'final_energy_kj', 'return_success',
          'max_constraint_residual', 't_max_s', 'e_max_kj', 'route', 'runtime_ms',
          'planner_runtime_ms', 'selection_times_ms', 'execution_times_ms', 'trace']
TIMING_FIELDS = ['method', 'L', 'seed', 'task_decision_total_ms',
                 'task_decision_count', 'total_selection_ms', 'total_execution_ms',
                 'warmups', 'time_budget_s', 'energy_budget_kj']


def methods_for(study):
    if study == 'Figure7':
        return tuple(ABLATIONS)
    if study == 'Figure8':
        return ('Proposed', 'Mean-workload DP')
    if study == 'ExecutionAblation':
        return ('Proposed', 'Normalized resource cost')
    if study == 'NumericalAccuracy':
        return tuple(NUMERICAL)
    return METHODS


def execute(job):
    study, point, method, seed = job
    public = build_case(study, point)
    hidden = generate_workload(public, seed)
    algorithm = method
    options = {}
    variant = ABLATIONS.get(method, '') if study == 'Figure7' else ''
    if study == 'NumericalAccuracy':
        variant, options = NUMERICAL[method]
        algorithm = 'Proposed'
    if study == 'Figure8':
        options['L'] = int(point)
    started = perf_counter()
    if algorithm in ('IO', 'Rollout', 'ADAPT'):
        from .literature_runner import literature_run
        row = literature_run(public, hidden, algorithm, seed)
        row['runtime_ms'] = (perf_counter() - started) * 1000
    else:
        from .proposed import run
        row = run(public, hidden, method=algorithm, seed=seed, **options)
    assert 0 <= row['wcr'] <= 1
    assert row['return_success'] == 1
    assert row['final_time_s'] <= public['pars'][0] + 1e-7
    assert row['final_energy_kj'] <= public['pars'][1] + 1e-7
    row.update(study=study, method=method, point=float(point), seed=int(seed),
               variant_id=variant, setting='', L=int(point) if study == 'Figure8' else 9,
               group_id=int(point) if study == 'Table4' else '',
               priority_config=int(point) if study == 'Table4' else '',
               t_max_s=float(public['pars'][0]), e_max_kj=float(public['pars'][1]))
    if study in ('Figure7', 'ExecutionAblation'):
        row['setting'] = study_config(study)['settings'][int(point)][0]
    if study == 'ExecutionAblation':
        row['variant_id'] = 'proposed' if method == 'Proposed' else 'normalized'
    return row


def serialized(row):
    return {name: json.dumps(row[name], separators=(',', ':'))
            if isinstance(row.get(name), (list, dict)) else row.get(name, '')
            for name in FIELDS}


def source_identity():
    here = Path(__file__).resolve().parent
    files = sorted(here.rglob('*.py')) + [here.parent / 'generate_nominal_case.py']
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(here.parent)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def result_key(row):
    return (row['study'], float(row['point']), row['method'], int(row['seed']))


def measure_timing(args, output, methods, points):
    """Serial warmed Top+Sub timing, including the final return decision."""
    target = output / 'timing.csv'
    if target.exists() and not args.overwrite:
        raise ValueError(f'{target} exists; use --overwrite to replace it.')
    if args.warmups < 3 or args.timing_count < 2:
        raise ValueError('Timing requires at least 3 warmups and 2 measured missions.')
    if args.timing_count > 300 or args.warmups > 300:
        raise ValueError('At most 300 paired seeds are available.')
    with target.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        for point in points:
            seeds = list(case_seeds('Figure8', point))
            for method in methods:
                for seed in seeds[:args.warmups]:
                    execute(('Figure8', point, method, seed))
            for seed in seeds[:args.timing_count]:
                for method in methods:
                    row = execute(('Figure8', point, method, seed))
                    selection, execution = row['selection_times_ms'], row['execution_times_ms']
                    assert len(selection) in (len(execution), len(execution) + 1)
                    writer.writerow(dict(method=method, L=int(point), seed=seed,
                        task_decision_total_ms=sum(selection) + sum(execution),
                        task_decision_count=len(selection),
                        total_selection_ms=sum(selection), total_execution_ms=sum(execution),
                        warmups=args.warmups, time_budget_s=2200, energy_budget_kj=460))
                    handle.flush()
            print(f'Measured L={point:g}: {args.timing_count} paired seeds', flush=True)
    print(f'Saved {target}')


def main(study, folder):
    folder = Path(folder)
    settings = study_config(study).get('settings', ())
    setting_help = '; '.join(f'--point {i}: {name} (T={t:g} s, E={e:g} kJ)'
                             for i, (name, t, e) in enumerate(settings))
    parser = argparse.ArgumentParser(description=__doc__, epilog=setting_help or None)
    parser.add_argument('--method', action='append', choices=methods_for(study),
                        help='Repeat to select methods; default: all methods for this study.')
    parser.add_argument('--point', type=float, action='append',
                        help='Repeat to select sweep points; default: all study points.')
    parser.add_argument('--count', type=int, default=None,
                        help='Paired missions per method/point (default: 300; Table IV: 10).')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=folder / 'data')
    parser.add_argument('--resume', action='store_true', help='Continue an existing run with matching code.')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing local result CSV.')
    parser.add_argument('--timing', action='store_true', help='Figure8 only: measure warmed decision times serially.')
    parser.add_argument('--timing-count', type=int, default=30)
    parser.add_argument('--warmups', type=int, default=3)
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error('--resume and --overwrite are mutually exclusive')
    if args.workers < 1 or (args.count is not None and args.count < 1):
        parser.error('--workers and --count must be positive')
    methods = list(dict.fromkeys(args.method or methods_for(study)))
    available = list(study_points(study))
    points = list(dict.fromkeys(args.point or available))
    if any(not any(np.isclose(p, x, rtol=0, atol=1e-10) for x in available) for p in points):
        parser.error(f'Choose points from {available}')
    points = [next(x for x in available if np.isclose(p, x, rtol=0, atol=1e-10)) for p in points]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.timing:
        if study != 'Figure8':
            parser.error('--timing is only available for Figure8')
        if args.workers != 1:
            parser.error('Use --workers 1 for serial decision timing')
        try:
            measure_timing(args, output, methods, points)
        except ValueError as exc:
            parser.error(str(exc))
        return
    jobs = []
    for point in points:
        seeds = list(case_seeds(study, point))
        count = len(seeds) if args.count is None else args.count
        if count > len(seeds):
            parser.error(f'{study}/{point} provides {len(seeds)} paired seeds; reduce --count')
        jobs.extend((study, point, method, seed) for method in methods for seed in seeds[:count])
    target, metadata = output / 'results.csv', output / 'run.json'
    identity = source_identity()
    completed = set()
    if target.exists() and not args.overwrite:
        if not args.resume:
            parser.error(f'{target} exists; use --resume or --overwrite')
        if not metadata.exists() or json.loads(metadata.read_text()).get('source_sha256') != identity:
            parser.error('Existing results use different or unknown source code; choose a new output directory')
        with target.open(newline='') as handle:
            completed = {result_key(row) for row in csv.DictReader(handle)}
    remaining = [job for job in jobs if job not in completed]
    metadata.write_text(json.dumps(dict(study=study, source_sha256=identity,
        methods=methods, points=points, requested_missions=len(jobs),
        settings=[dict(point=i, setting=name, t_max_s=t, e_max_kj=e)
                  for i, (name, t, e) in enumerate(settings)],
        default_time_s=2200, default_energy_kj=460), indent=2) + '\n')
    if not remaining:
        print(f'All {len(jobs)} requested missions already exist in {target}')
        return
    print(f'Running {len(remaining)} missions; output: {target}', flush=True)
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        rows = pool.map(execute, remaining, chunksize=1) if pool else map(execute, remaining)
        append = bool(completed) and args.resume
        with target.open('a' if append else 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if not append:
                writer.writeheader()
            for index, row in enumerate(rows, 1):
                writer.writerow(serialized(row))
                handle.flush()
                if index == 1 or index % 100 == 0 or index == len(remaining):
                    print(f'{index}/{len(remaining)} completed', flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    print(f'Saved {target}')
