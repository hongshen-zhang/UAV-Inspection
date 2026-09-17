"""Generate the paper's paired inputs from model constants and random seeds.

No stored observations, sample banks, CSV files or NPZ archives are read.
Study names use the paper numbering: Figure4 is the main comparison,
Figure5 the time sweep, Figure6 the energy sweep and Figure7 the ablation.
The legacy top-level folders use the mapping in LEGACY_STUDIES.

``public = build_case(study, point)`` returns only the 15 public model arrays
as dictionary keys. Its attributes retain the generation settings, so pass
that object directly to ``generate_workload(public, seed)``. The returned
workload is private to the mission driver and must never enter a planner.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math

import numpy as np
from scipy.special import ndtri

try:
    from ..generate_nominal_case import (
        CONFIG, NODES, TASK_ATTRIBUTES, channel_rows, project_lonlat,
    )
except ImportError:  # Also support ``current.inputs`` with simulation on sys.path.
    from generate_nominal_case import (
        CONFIG, NODES, TASK_ATTRIBUTES, channel_rows, project_lonlat,
    )

SEED_START = 2026092000
SEEDS = tuple(range(SEED_START, SEED_START + 300))
BOOTSTRAP_SEED = 2026091510
BOOTSTRAP_REPLICATES = 20000
# These original formal conditions were drawn directly from their scaled
# priors. Later intermediate points transformed the nominal sample bank.
FORMAL_SCALES = (0.5, 0.75, 1.25, 1.5)
DEFAULT_OPTIONS = dict(L=9, support_order=9, quadrature=3,
                       time_step=25.0, energy_step=7.5)
METHODS = ('Proposed', 'Weight Greedy', 'Mean-workload Greedy',
           'Mean-workload DP', 'Distribution-aware Myopic', 'Nearest Neighbor',
           'IO', 'Rollout', 'ADAPT')
PUBLIC_NAMES = ('weights', 'deadline', 'mu', 'sigma', 'pars', 'aa', 'bb',
                'dd', 'ee', 'flight_time', 'flight_energy', 'points',
                'probabilities', 'mec_frequencies', 'mean_workloads')
LEGACY_STUDIES = {'Figure5': 'Figure4', 'Figure6': 'Figure5',
                  'Figure7': 'Figure6', 'Figure12': 'Figure7',
                  'Figure8': 'Figure8', 'Figure9': 'Figure9',
                  'Figure10': 'Figure10', 'Figure11': 'Figure11',
                  'PriorityRobustness': 'Table4'}
ABLATION_VARIANTS = {
    'proposed': 'Proposed', 'mean_workload_dp': 'Mean-workload DP',
    'no_priority': 'No priority factor', 'top_myopic': 'Myopic Top',
    'no_proactive_skip': 'No proactive skip',
}
ABLATION_SETTINGS = (
    ('default', 2200.0, 460.0),
    ('larger_time', 2600.0, 460.0),
    ('lower_energy', 2200.0, 380.0),
)
# The execution study's original case order differs from Figure7's.
EXECUTION_SETTINGS = (ABLATION_SETTINGS[0], ABLATION_SETTINGS[2],
                      ABLATION_SETTINGS[1])
NUMERICAL_VARIANTS = {
    'default': ('Default', {}),
    'dp_g1': ('DP bins G=1', {'quadrature': 1}),
    'dp_g5': ('DP bins G=5', {'quadrature': 5}),
    'support_h3': ('Support H=3', {'support_order': 3}),
    'support_h15': ('Support H=15', {'support_order': 15}),
    'time_grid50': ('Time grid 50 s', {'time_step': 50.0}),
    'energy_grid15': ('Energy grid 15 kJ', {'energy_step': 15.0}),
}
STUDIES = {
    'Figure4': dict(axis='nominal', points=(1.0,), methods=METHODS),
    'Figure5': dict(axis='time', points=tuple(range(800, 3201, 200)), methods=METHODS),
    'Figure6': dict(axis='energy', points=(140, 200, 260, 300, 340, 380, 420,
                                         460, 500, 560, 620, 680, 740, 800),
                    methods=METHODS),
    'Figure7': dict(axis='ablation', points=(0, 1, 2),
                    methods=tuple(ABLATION_VARIANTS.values()),
                    variants=ABLATION_VARIANTS, settings=ABLATION_SETTINGS),
    'Figure8': dict(axis='candidate_cap', points=tuple(range(1, 12)),
                    methods=('Proposed', 'Mean-workload DP'),
                    timing_seed_count=30, timing_warmups=3),
    'Figure9': dict(axis='uncertainty', points=tuple(i / 8 for i in range(17)),
                    methods=METHODS),
    'Figure10': dict(axis='workload', points=tuple(i / 8 for i in range(1, 17)),
                     methods=METHODS),
    'Figure11': dict(axis='mec', points=tuple(round(.35 * i, 2) for i in range(2, 17)),
                     methods=METHODS),
    'Table4': dict(axis='priority', points=tuple(range(1, 31)), methods=METHODS,
                   seeds_per_point=10),
    'ExecutionAblation': dict(axis='execution_ablation', points=(0, 1, 2),
                              methods=('Proposed', 'Normalized resource cost'),
                              settings=EXECUTION_SETTINGS),
    'NumericalAccuracy': dict(axis='numerical', points=(1.0,),
                              methods=tuple(x[0] for x in NUMERICAL_VARIANTS.values()),
                              variants=NUMERICAL_VARIANTS),
}


class PublicArrays(dict):
    """Public arrays with separate, nonrandom generation metadata."""

    def __init__(self, arrays, *, study, point, uncertainty_scale=1.0,
                 workload_scale=1.0):
        super().__init__(arrays)
        self.study = study
        self.point = point
        self.uncertainty_scale = float(uncertainty_scale)
        self.workload_scale = float(workload_scale)

    def copy(self):
        return PublicArrays({key: value.copy() for key, value in self.items()},
                            study=self.study, point=self.point,
                            uncertainty_scale=self.uncertainty_scale,
                            workload_scale=self.workload_scale)


def study_config(study):
    """Return a fresh study description, including points, methods and options."""
    if study not in STUDIES:
        raise ValueError(f'Unknown study {study!r}; choose {tuple(STUDIES)}')
    return dict(name=study, seeds=SEEDS, options=dict(DEFAULT_OPTIONS),
                **deepcopy(STUDIES[study]))


def study_points(study):
    return list(study_config(study)['points'])


def _point(study, point):
    cfg = study_config(study)
    if point is None:
        return 9 if study == 'Figure8' else cfg['points'][0]
    if isinstance(point, str) and 'settings' in cfg:
        for index, (name, _, _) in enumerate(cfg['settings']):
            if point == name:
                return index
    value = float(point)
    for candidate in cfg['points']:
        if math.isclose(value, float(candidate), rel_tol=0.0, abs_tol=1e-12):
            return candidate
    raise ValueError(f'Invalid {study} point {point!r}; choose {cfg["points"]}')


def study_seeds(study, point=None):
    """The paired seed set; Table4 uses ten disjoint seeds per priority vector."""
    point = _point(study, point)
    if study != 'Table4':
        return SEEDS
    group = int(point)
    return SEEDS[2 * (group - 1):2 * group] + SEEDS[60 + 8 * (group - 1):60 + 8 * group]


def case_seeds(study, point=None):
    """List form of the case's canonical seed set, for experiment runners."""
    return list(study_seeds(study, point))


def priority_weights(group):
    """Exactly the original 30 independent priority assignments."""
    if int(group) != group or not 1 <= int(group) <= 30:
        raise ValueError('Priority group must be an integer from 1 through 30')
    return np.random.default_rng(2026091500 + int(group)).permutation(
        np.asarray([1, 3, 6, 10, 15] * 4, dtype=int))


def geometry():
    """Return model coordinates; rounded task xy matches the original CSV format."""
    tasks = sorted(TASK_ATTRIBUTES, key=lambda row: int(row['task_id']))
    mecs = sorted((row for row in NODES if row['node_type'] == 'mec'),
                  key=lambda row: int(row['mec_id']))
    base = next(row for row in NODES if row['node_type'] == 'base')
    task_xy = np.asarray([[float(f'{v:.12f}') for v in project_lonlat(
        float(row['lon']), float(row['lat']))] for row in tasks])
    return dict(base_xy=np.asarray([float(base['x_m']), float(base['y_m'])]),
                task_xy=task_xy,
                mec_xy=np.asarray([[float(row['x_m']), float(row['y_m'])] for row in mecs]),
                task_lonlat=np.asarray([[float(row['lon']), float(row['lat'])] for row in tasks]),
                mec_lonlat=np.asarray([[float(row['lon']), float(row['lat'])] for row in mecs]))


@lru_cache(maxsize=1)
def _nominal_arrays():
    phys = CONFIG['physics']
    n, m = int(phys['n_tasks']) + 1, int(phys['n_mec'])
    arrays = {name: np.zeros(n, dtype=np.float64) for name in
              ('weights', 'deadline', 'mu', 'sigma', 'mean_workloads')}
    for name in ('aa', 'bb', 'dd', 'ee'):
        arrays[name] = np.full((n, m), 1e100 if name in ('aa', 'dd') else 0.0)
    arrays['mec_frequencies'] = np.asarray(phys['mec_cpu_ghz'], dtype=np.float64)
    # Link generation uses unrounded projected positions, as the original generator.
    geo = geometry()
    projected_mecs = np.asarray([project_lonlat(*xy) for xy in geo['mec_lonlat']])
    for task in TASK_ATTRIBUTES:
        i = int(task['task_id'])
        mean, sigma = float(task['mean_workload_gcy']), float(task['sigma'])
        mu = math.log(mean) - 0.5 * sigma * sigma
        arrays['weights'][i] = float(task['weight'])
        arrays['deadline'][i] = float(task['deadline_s'])
        arrays['mu'][i], arrays['sigma'][i] = mu, sigma
        arrays['mean_workloads'][i] = mean
        xy = project_lonlat(float(task['lon']), float(task['lat']))
        for j, link in enumerate(channel_rows(xy, projected_mecs, phys, i)):
            ru, rd = link['ul_rate_mbps'], link['dl_rate_mbps']
            if ru < phys['ul_min_mbps'] or rd < phys['dl_min_mbps']:
                continue
            tu = float(task['data_mbit']) / ru
            td = float(task['output_ratio']) * float(task['data_mbit']) / rd
            fm, ph = arrays['mec_frequencies'][j], float(phys['hover_power_w']) / 1000.0
            arrays['aa'][i, j], arrays['bb'][i, j] = tu + td, 1.0 / fm
            arrays['dd'][i, j] = phys['tx_power_w'] * tu / 1000.0 + phys['rx_power_w'] * td / 1000.0 + ph * (tu + td)
            arrays['ee'][i, j] = ph / fm
    coordinates = np.vstack((geo['base_xy'], geo['task_xy']))
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    arrays['flight_time'] = distance / float(phys['speed_mps'])
    arrays['flight_energy'] = arrays['flight_time'] * float(phys['flight_power_w']) / 1000.0
    arrays['pars'] = np.asarray([2200.0, 460.0, phys['local_f_min_ghz'],
                                phys['local_f_max_ghz'], phys['kappa_kj_per_gcycle_ghz2'],
                                phys['hover_power_w'] / 1000.0], dtype=np.float64)
    return arrays


def build_case(study='Figure4', point=None):
    """Construct public model arrays for one nominal, sweep or ablation case."""
    point = _point(study, point)
    arrays = {key: value.copy() for key, value in _nominal_arrays().items()}
    uncertainty, workload = 1.0, 1.0
    if study == 'Figure5':
        arrays['pars'][0] = float(point)
    elif study == 'Figure6':
        arrays['pars'][1] = float(point)
    elif study in ('Figure7', 'ExecutionAblation'):
        _, t, e = STUDIES[study]['settings'][int(point)]
        arrays['pars'][:2] = (t, e)
    elif study == 'Figure9' and point != 1.0:
        uncertainty = float(point)
        for i in range(1, len(arrays['mu'])):
            sigma = float(arrays['sigma'][i]) * uncertainty
            arrays['sigma'][i] = sigma
            arrays['mu'][i] = math.log(float(arrays['mean_workloads'][i])) - sigma * sigma / 2
    elif study == 'Figure10' and point != 1.0:
        workload = float(point)
        for i in range(1, len(arrays['mu'])):
            arrays['mean_workloads'][i] = float(arrays['mean_workloads'][i]) * workload
            if point in FORMAL_SCALES:
                sigma = float(arrays['sigma'][i])
                arrays['mu'][i] = math.log(float(arrays['mean_workloads'][i])) - .5 * sigma * sigma
            else:
                arrays['mu'][i] = float(arrays['mu'][i]) + math.log(workload)
    elif study == 'Figure11':
        original_scales = {1.4: .5, 2.1: .75, 2.8: 1.0, 4.2: 1.5, 5.6: 2.0}
        frequency = 2.8 * original_scales[point] if point in original_scales else float(point)
        arrays['mec_frequencies'][:] = frequency
        valid = arrays['aa'] < 1e90
        arrays['bb'][valid] = 1.0 / frequency
        arrays['ee'][valid] = arrays['pars'][5] / frequency
    elif study == 'Table4':
        arrays['weights'][1:] = priority_weights(int(point))
    # Literature baselines use nine midpoint quantiles; Proposed rebuilds its
    # original GH support inside its controller, so these are not substituted.
    z = ndtri((np.arange(9, dtype=np.float64) + 0.5) / 9.0)
    arrays['points'] = np.zeros((len(arrays['mu']), 9), dtype=np.float64)
    for i in range(1, len(arrays['mu'])):
        arrays['points'][i] = np.exp(arrays['mu'][i] + arrays['sigma'][i] * z)
    arrays['probabilities'] = np.full(9, 1.0 / 9.0, dtype=np.float64)
    # The public kernel adapter derives means from the serialized priors;
    # transformations above use the task model's original decimal means.
    arrays['mean_workloads'] = np.exp(arrays['mu'] + .5 * arrays['sigma'] ** 2)
    arrays['mean_workloads'][0] = 0.0
    return PublicArrays({key: arrays[key] for key in PUBLIC_NAMES}, study=study,
                        point=point, uncertainty_scale=uncertainty,
                        workload_scale=workload)


def generate_workload(public, seed):
    """Draw one hidden, paired realization without reading a stored sample bank."""
    if not isinstance(public, PublicArrays):
        raise TypeError('Pass the PublicArrays object returned by build_case')
    if int(seed) != seed or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nominal = _nominal_arrays()
    hidden = np.zeros(len(public['mu']), dtype=np.float64)
    for i in range(1, len(hidden)):
        mu, sigma = float(nominal['mu'][i]), float(nominal['sigma'][i])
        rng = np.random.default_rng(np.random.SeedSequence([seed, i, 20260911]))
        original_z = float(rng.standard_normal())
        if public.study in ('Figure9', 'Figure10') and public.point in FORMAL_SCALES:
            hidden[i] = math.exp(float(public['mu'][i]) + float(public['sigma'][i]) * original_z)
            continue
        value = math.exp(mu + sigma * original_z)
        if public.uncertainty_scale == 0.0:
            value = float(nominal['mean_workloads'][i])
        elif public.uncertainty_scale != 1.0:
            # Preserve the original base-normal bank's exact round-trip rule.
            z = (math.log(value) - mu) / sigma
            value = math.exp(float(public['mu'][i]) + float(public['sigma'][i]) * z)
        if public.workload_scale != 1.0:
            value *= public.workload_scale
        hidden[i] = value
    return hidden
