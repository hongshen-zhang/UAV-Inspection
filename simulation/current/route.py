"""Generate the nominal route from the same controller used by the studies."""
from __future__ import annotations

import json
import math

from ..generate_nominal_case import NODES
from .inputs import SEED_START, build_case, generate_workload
from .proposed import run as run_mission


def mission_geometry():
    """Return the map coordinates from the public scenario parameters."""
    base = next(node for node in NODES if node['node_type'] == 'base')
    return {
        'base': {'name': 'Base', 'lon': float(base['lon']),
                 'lat': float(base['lat'])},
        'mec': [
            {'name': f"MEC {int(node['mec_id'])}",
             'lon': float(node['lon']), 'lat': float(node['lat'])}
            for node in NODES if node['node_type'] == 'mec'
        ],
        'tasks': [
            {'idx': int(node['task_id']),
             'lon': float(node['lon']), 'lat': float(node['lat'])}
            for node in NODES if node['node_type'] == 'task'
        ],
    }


def run(seed=SEED_START):
    """Run and validate one nominal mission without any recorded input data."""
    public = build_case('Figure4')
    hidden = generate_workload(public, seed)
    seed = int(seed)
    row = run_mission(public, hidden, method='Proposed', seed=seed)
    trace = json.loads(row.pop('trace'))
    row.update(mcr=row['wcr'], t_max_s=float(public['pars'][0]),
               e_max_kj=float(public['pars'][1]))
    visited = [int(event['task']) for event in trace]
    completed = [event for event in trace if event['mode'] != 'skip']
    checks = {
        'return_within_budgets': bool(row['return_success'])
            and 0.0 <= row['final_time_s'] <= row['t_max_s'] + 1e-7
            and 0.0 <= row['final_energy_kj'] <= row['e_max_kj'] + 1e-7,
        'unique_visits': len(visited) == len(set(visited)) == row['visited_tasks'],
        'completion_counts': len(completed) == row['completed_tasks']
            == row['local_actions'] + row['mec_actions']
            and len(trace) - len(completed) == row['skipped_tasks'],
        'completed_weight': math.isclose(
            sum(float(public['weights'][int(event['task'])])
                for event in completed),
            row['weighted_completed'], abs_tol=1e-8, rel_tol=0.0),
        'completed_deadlines': all(
            event['time_s'] <= public['deadline'][int(event['task'])] + 1e-7
            for event in completed),
        'route_matches_trace': row['route'] == '-'.join(map(str, [0, *visited, 0])),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError('Mission validation failed: ' + ', '.join(failed))
    return {
        'description': 'Fresh execution of the nominal mission.',
        'seed': seed,
        'validation': checks,
        'mission_result': row,
        'mission_geometry': mission_geometry(),
        'trace': trace,
    }
