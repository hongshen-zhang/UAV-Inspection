"""Single-component MTE-Top ablations; all frozen simulator files are read-only."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simulation'))
import ablation_base as original
from ablation_base import (EnhancedPlanner, EnhancedContinuation, SPEC, POLICY,
    ShortlistContinuation, MyopicContinuation, ReferencePlanner,
    MeanInformationPlanner, screen_rows, iteration_v2, robust_core)

NAMES = {
    'no_feasibility': 'Without feasibility score',
    'no_cost': 'Without resource cost score',
    'no_priority': 'Without priority score',
    'no_time_cost': 'Without time cost score',
    'no_energy_cost': 'Without energy cost score',
    'top_myopic': 'Without Top future value',
}
HOOKS = {
    'no_feasibility': {'component': 'shortlist score', 'formula': 'ell / (time_cost + energy_cost)**0.5'},
    'no_cost': {'component': 'shortlist score', 'formula': 'ell * feasible_probability'},
    'no_priority': {'component': 'shortlist score', 'formula': 'feasible_probability / (time_cost + energy_cost)**0.5'},
    'no_time_cost': {'component': 'shortlist score', 'formula': 'ell * feasible_probability / energy_cost**0.5'},
    'no_energy_cost': {'component': 'shortlist score', 'formula': 'ell * feasible_probability / time_cost**0.5'},
    'top_myopic': {'component': 'next-task selection only',
        'formula': 'expected immediate reward with the original G=3 root integration',
        'preserved': 'original score shortlist; actual Sub LP, frequency optimization, completion/skip comparison and EnhancedContinuation DP'},
}


def score_components(base, state):
    """Identical frozen public-information statistics, exposing score factors."""
    ft, fe = base.bundle.iteration.flight_time, base.bundle.iteration.flight_energy_kj
    rows = {}
    for task in range(1, len(base.weights)):
        if not state.remaining_mask & (1 << (task - 1)):
            continue
        if not iteration_v2.visit_ok_v2(state.node, state.time_s, state.energy_kj,
                task, ft, fe, base.deadline, base.pars):
            continue
        p, st, se = robust_core.calibrated_task_stats(state.node, state.time_s,
            state.energy_kj, task, base.points, base.probabilities, base.mu,
            base.sigma, ft, fe, base.deadline, base.pars, base.aa, base.bb,
            base.dd, base.ee, 0)
        time_cost = (ft[state.node, task] + st) / max(base.pars[0] - state.time_s, 1e-12)
        energy_cost = (fe[state.node, task] + se) / max(base.pars[1] - state.energy_kj, 1e-12)
        rows[task] = (float(base.weights[task]), float(p), float(time_cost), float(energy_cost))
    return rows


def modified_score(factors, mode):
    priority, probability, time_cost, energy_cost = factors
    if mode == 'no_feasibility': probability = 1.
    elif mode == 'no_priority': priority = 1.
    elif mode == 'no_time_cost': time_cost = 0.
    elif mode == 'no_energy_cost': energy_cost = 0.
    elif mode == 'no_cost': return priority * probability
    else: raise ValueError(mode)
    return priority * probability / max(time_cost + energy_cost, 1e-12) ** SPEC.cost_power


class ScoreContinuation(ShortlistContinuation):
    def __init__(self, base, spec, mode):
        super().__init__(base, spec)
        self.mode = mode
        self.first_screen = None

    def selected_tasks(self, state, scores):
        reference = [-item[1] for item in sorted(scores, reverse=True)[:self.spec.shortlist]]
        components = score_components(self.base, state)
        changed = [(modified_score(factors, self.mode), -task) for task, factors in components.items()]
        assert set(components) == {-item[1] for item in scores}
        chosen = [-item[1] for item in sorted(changed, reverse=True)[:self.spec.shortlist]]
        self.shortlist_calls += 1
        self.shortlist_changed_calls += int(chosen != reference)
        self.shortlist_set_changed_calls += int(set(chosen) != set(reference))
        self.last_tasks = chosen
        if self.first_screen is None:
            self.first_screen = {'reference_tasks': reference, 'selected_tasks': chosen,
                'components': components, 'modified_scores': {str(-i): s for s, i in changed}}
        return np.asarray(chosen, dtype=np.int64)


class ScorePlanner(EnhancedPlanner):
    enhanced_spec = SPEC
    score_mode = None
    def __init__(self, bundle, policy):
        super().__init__(bundle, policy)
        self.continuation = ScoreContinuation(self.continuation.base, self.enhanced_spec, self.score_mode)


class NoFeasibilityPlanner(ScorePlanner): score_mode = 'no_feasibility'
class NoCostPlanner(ScorePlanner): score_mode = 'no_cost'
class NoPriorityPlanner(ScorePlanner): score_mode = 'no_priority'
class NoTimeCostPlanner(ScorePlanner): score_mode = 'no_time_cost'
class NoEnergyCostPlanner(ScorePlanner): score_mode = 'no_energy_cost'


class TopMyopicPlanner(EnhancedPlanner):
    """One-step selection, but the actual Sub retains the full continuation DP."""
    enhanced_spec = SPEC
    def __init__(self, bundle, policy):
        super().__init__(bundle, policy)
        self.top_selector = MyopicContinuation(self.continuation.base, self.enhanced_spec)

    def select_task(self, state):
        self.stats['route_calls'] += 1
        task, value = self.top_selector.plan(self._plan_state(state))
        return task, {'selection_mode': 'top_only_immediate_reward', 'selected_value': value,
            'continuation_signature': self.continuation.signature}


CLASSES = {
    'no_feasibility': NoFeasibilityPlanner, 'no_cost': NoCostPlanner,
    'no_priority': NoPriorityPlanner, 'no_time_cost': NoTimeCostPlanner,
    'no_energy_cost': NoEnergyCostPlanner, 'top_myopic': TopMyopicPlanner,
}
