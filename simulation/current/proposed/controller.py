"""Public-array adapters around the original route and arrival algorithms.

No planner stores realized workloads. The mission driver supplies a scalar
workload to choose_actual_action only after task selection has finished.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import math
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr
from . import physics as iteration_v2
from . import physics as robust_core
from . import prices as root_calibrated_core
from .dp import EnhancedSpec, EnhancedContinuation
EPS=1e-12

@dataclass(frozen=True)
class PlanState:
    node: int
    time_s: float
    energy_kj: float
    remaining_mask: int


@dataclass(frozen=True)
class ExecutionAction:
    mode: str
    mec: int
    frequency_ghz: float
    service_time_s: float
    service_energy_kj: float
    occupation: float


def immediate_reward(priority: float, action: ExecutionAction) -> float:
    """Return the original MTE reward without a configurable coefficient."""

    return float(priority) if action.mode != "skip" else 0.0


def deterministic_mean_support(
    mu: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse each LogNormal workload to its arithmetic mean.

    The returned ``mu`` and ``sigma`` describe a degenerate LogNormal at
    ``E[C]``.  Passing those arrays to the analytic completion-mass code is
    essential: changing only the quadrature points would still let the planner
    use the original distribution through its exact CDF.
    """

    mean = np.exp(
        np.asarray(mu, dtype=np.float64)
        + np.square(np.asarray(sigma, dtype=np.float64)) / 2.0
    )
    points = mean[:, np.newaxis]
    probabilities = np.ones(1, dtype=np.float64)
    deterministic_mu = np.log(np.maximum(mean, np.finfo(np.float64).tiny))
    deterministic_sigma = np.zeros_like(deterministic_mu)
    return points, probabilities, deterministic_mu, deterministic_sigma


class BasePlanner:
    """Initialization adapter; decision methods below are original source."""
    def __init__(self, public_arrays, *, mean=False, L=9, time_step=25.,
                 energy_step=7.5, quadrature=3, support_order=9):
        for name in ('weights','deadline','mu','sigma','pars','aa','bb','dd','ee'):
            value=np.array(public_arrays[name], dtype=np.float64, copy=True)
            setattr(self,name,value)
        # Real execution/LP take physical fields only. Virtual prediction uses
        # the original pressure defaults q=1, beta=0.5 from BatchCase.
        self.pars=self.pars[:6].copy()
        self.n_tasks=len(self.weights)-1
        ft=np.array(public_arrays['flight_time'],dtype=np.float64,copy=True)
        fe=np.array(public_arrays['flight_energy'],dtype=np.float64,copy=True)
        self.mission=SimpleNamespace(flight_time=ft,flight_energy_kj=fe,
            cfg=SimpleNamespace(mec_cpu_ghz=np.array(public_arrays['mec_frequencies'],copy=True)))
        if mean:
            points, probabilities, self.mu, self.sigma=deterministic_mean_support(self.mu,self.sigma)
        else:
            # Literature-baseline archives also carry a nine-point quantile
            # support. The original six methods instead rebuild GH support.
            nodes,weights=hermgauss(int(support_order))
            probabilities=(weights/math.sqrt(math.pi)).astype(np.float64)
            points=np.zeros((len(self.mu),int(support_order)),dtype=np.float64)
            for i in range(1,len(self.mu)):
                points[i]=np.exp(self.mu[i]+math.sqrt(2.)*self.sigma[i]*nodes)
        self.price_points,self.price_probabilities=points,probabilities
        self.policy=SimpleNamespace(action_rule='mode_separated',
            continuation=SimpleNamespace(analytic_mass_calibration=False))
        self.stats={name:0 for name in ('route_calls','arrival_calls','dual_price_calls',
            'dual_price_cache_hits','action_value_calls')}
        self._price_cache={}
        self.enhanced_spec=EnhancedSpec(shortlist=int(L),time_step=float(time_step),
            energy_step=float(energy_step),quadrature=int(quadrature),tail_weight=0.,
            tail_depth=6,cost_power=.5,rounding=1,mode='both',arrival_rule='original',integration=0)
        base=SimpleNamespace(points=points,probabilities=probabilities,mu=self.mu,
            sigma=self.sigma,weights=self.weights,deadline=self.deadline,
            pars=np.r_[self.pars,1.,.5],aa=self.aa,bb=self.bb,dd=self.dd,ee=self.ee,
            conditional_uniform=np.ones(1),bundle=SimpleNamespace(iteration=self.mission))
        self.continuation=EnhancedContinuation(base,self.enhanced_spec)
        for value in (self.weights,self.deadline,self.mu,self.sigma,self.pars,self.aa,
                      self.bb,self.dd,self.ee,ft,fe,points,probabilities):
            value.flags.writeable=False

    def _plan_state(self, state):
        return state

    def select_task(self, state):
        self.stats['route_calls']+=1
        task,value=self.continuation.plan(self._plan_state(state))
        return task, {'selection_mode':'bounded_stochastic_dp','selected_value':value}

    def _arrival(
        self, plan: PlanState, task: int
    ) -> tuple[float, float, float, float] | None:
        if not iteration_v2.visit_ok_v2(
            int(plan.node),
            float(plan.time_s),
            float(plan.energy_kj),
            int(task),
            self.mission.flight_time,
            self.mission.flight_energy_kj,
            self.deadline,
            self.pars,
        ):
            return None
        arrival_t, arrival_e, available_t, available_e = (
            iteration_v2.resource_limits_v2(
                int(plan.node),
                float(plan.time_s),
                float(plan.energy_kj),
                int(task),
                self.mission.flight_time,
                self.mission.flight_energy_kj,
                self.deadline,
                self.pars,
            )
        )
        if available_t <= 0.0 or available_e <= 0.0:
            return None
        return (
            float(arrival_t),
            float(arrival_e),
            float(available_t),
            float(available_e),
        )


    def _prices(
        self, task: int, arrival_t: float, arrival_e: float, remaining: int
    ) -> tuple[float, float]:
        if self.policy.action_rule != "mode_separated" or int(remaining) == 0:
            return 0.0, 0.0
        cache_key = (
            int(task),
            float(arrival_t),
            float(arrival_e),
            int(remaining),
        )
        cached = self._price_cache.get(cache_key)
        if cached is not None:
            self.stats["dual_price_cache_hits"] += 1
            return cached
        if self.policy.continuation.analytic_mass_calibration:
            price_t, price_e, _ = self._analytic_prices(
                int(task),
                float(arrival_t),
                float(arrival_e),
                int(remaining),
            )
        else:
            price_t, price_e, _ = root_calibrated_core.state_lp_dual_prices_v2(
                int(task),
                float(arrival_t),
                float(arrival_e),
                np.uint64(remaining),
                np.asarray(self.price_points, dtype=np.float64),
                np.asarray(self.price_probabilities, dtype=np.float64),
                self.mission.flight_time,
                self.mission.flight_energy_kj,
                self.weights,
                self.deadline,
                self.pars,
                self.aa,
                self.bb,
                self.dd,
                self.ee,
            )
        self.stats["dual_price_calls"] += 1
        result = max(0.0, float(price_t)), max(0.0, float(price_e))
        self._price_cache[cache_key] = result
        return result


    @staticmethod
    def _local_weights(
        price_t: float, price_e: float, available_t: float, available_e: float
    ) -> tuple[float, float]:
        scaled_t = max(0.0, float(price_t)) * max(float(available_t), 0.0)
        scaled_e = max(0.0, float(price_e)) * max(float(available_e), 0.0)
        total = scaled_t + scaled_e
        if total <= 1e-14:
            return 1.0, 1.0
        return (
            max(EPS, 2.0 * scaled_t / total),
            max(EPS, 2.0 * scaled_e / total),
        )


    def _completion_actions(
        self,
        task: int,
        workload: float,
        available_t: float,
        available_e: float,
        price_t: float,
        price_e: float,
    ) -> tuple[ExecutionAction, ...]:
        if self.policy.action_rule == "mode_separated":
            time_weight, energy_weight = self._local_weights(
                price_t, price_e, available_t, available_e
            )
        else:
            time_weight, energy_weight = 1.0, 1.0

        actions: list[ExecutionAction] = []
        local_ok, local_t, local_e, _mode, local_f, _cost = (
            iteration_v2.execute_virtual_v2(
                float(workload),
                float(available_t),
                float(available_e),
                self.pars,
                self.aa[int(task)],
                self.bb[int(task)],
                self.dd[int(task)],
                self.ee[int(task)],
                1,
                float(time_weight),
                float(energy_weight),
                0.0,
            )
        )
        if local_ok:
            actions.append(
                ExecutionAction(
                    "local",
                    -1,
                    float(local_f),
                    float(local_t),
                    float(local_e),
                    float(local_t) / max(float(available_t), EPS)
                    + float(local_e) / max(float(available_e), EPS),
                )
            )

        for mec in range(self.aa.shape[1]):
            if float(self.aa[int(task), mec]) >= 1e90:
                continue
            service_t = float(
                self.aa[int(task), mec]
                + self.bb[int(task), mec] * float(workload)
            )
            service_e = float(
                self.dd[int(task), mec]
                + self.ee[int(task), mec] * float(workload)
            )
            if service_t > float(available_t) + 1e-9:
                continue
            if service_e > float(available_e) + 1e-9:
                continue
            actions.append(
                ExecutionAction(
                    "mec",
                    int(mec),
                    float(self.mission.cfg.mec_cpu_ghz[mec]),
                    service_t,
                    service_e,
                    service_t / max(float(available_t), EPS)
                    + service_e / max(float(available_e), EPS),
                )
            )

        if self.policy.action_rule == "mode_separated" and actions:
            selected = min(
                actions,
                key=lambda action: (
                    float(price_t) * action.service_time_s
                    + float(price_e) * action.service_energy_kj,
                    action.occupation,
                    action.service_time_s,
                    action.service_energy_kj,
                    action.mec,
                ),
            )
            return (selected,)
        return tuple(actions)


    @staticmethod
    def _skip_action() -> ExecutionAction:
        return ExecutionAction("skip", -1, 0.0, 0.0, 0.0, 0.0)


    def _successor(
        self,
        task: int,
        arrival_t: float,
        arrival_e: float,
        remaining: int,
        action: ExecutionAction,
    ) -> PlanState:
        return PlanState(
            int(task),
            float(arrival_t) + float(action.service_time_s),
            float(arrival_e) + float(action.service_energy_kj),
            int(remaining),
        )


    def _action_value(
        self,
        task: int,
        arrival_t: float,
        arrival_e: float,
        remaining: int,
        action: ExecutionAction,
    ) -> float:
        self.stats["action_value_calls"] += 1
        successor = self._successor(
            task, arrival_t, arrival_e, remaining, action
        )
        reward = immediate_reward(float(self.weights[int(task)]), action)
        return self.continuation.value(successor, reward)


    def _best_action(
        self,
        task: int,
        workload: float,
        arrival: tuple[float, float, float, float],
        remaining: int,
        price_t: float,
        price_e: float,
    ) -> tuple[ExecutionAction, float]:
        arrival_t, arrival_e, available_t, available_e = arrival
        completions = self._completion_actions(
            task,
            workload,
            available_t,
            available_e,
            price_t,
            price_e,
        )
        actions = (self._skip_action(), *completions)
        scored = [
            (
                self._action_value(
                    task, arrival_t, arrival_e, remaining, action
                ),
                action,
            )
            for action in actions
        ]
        value, selected = max(
            scored,
            key=lambda item: (
                item[0],
                1 if item[1].mode != "skip" else 0,
                -item[1].occupation,
                -item[1].service_time_s,
                -item[1].service_energy_kj,
                -item[1].mec,
            ),
        )
        return selected, float(value)


    def choose_actual_action(
        self, state: Any, task: int, workload: float
    ) -> ExecutionAction:
        """Choose recourse using the same evaluator used by ``select_task``."""

        self.stats["arrival_calls"] += 1
        plan = self._plan_state(state)
        arrival = self._arrival(plan, int(task))
        if arrival is None:
            return self._skip_action()
        remaining = plan.remaining_mask & (~(1 << (int(task) - 1)))
        price_t, price_e = self._prices(
            int(task), arrival[0], arrival[1], remaining
        )
        selected, _value = self._best_action(
            int(task),
            float(workload),
            arrival,
            remaining,
            price_t,
            price_e,
        )
        return selected



class HeuristicPlanner(BasePlanner):
    rule='density'
    def select_task(self,state):
        self.stats['route_calls']+=1
        s=self._plan_state(state);best=0;value=0
        for i in range(1,self.n_tasks+1):
            if not s.remaining_mask & (1<<(i-1)):continue
            arrival=self._arrival(s,i)
            if arrival is None:continue
            p,t,e=robust_core.calibrated_task_stats(s.node,s.time_s,s.energy_kj,i,
                self.price_points,self.price_probabilities,self.mu,self.sigma,
                self.mission.flight_time,self.mission.flight_energy_kj,self.deadline,self.pars,
                self.aa,self.bb,self.dd,self.ee,0)
            if p<=1e-12:continue
            cost=(self.mission.flight_time[s.node,i]+t)/max(self.pars[0]-s.time_s,1e-9)
            cost+=(self.mission.flight_energy_kj[s.node,i]+e)/max(self.pars[1]-s.energy_kj,1e-9)
            v=self.weights[i]*p/max(cost,1e-9)
            if self.rule=='nearest':v=1/max(self.mission.flight_time[s.node,i],1e-9)
            if v>value+1e-12:best,value=i,v
        return best,{'selection_mode':self.rule,'selected_value':float(value)}
    def _prices(self,*args):return 0.0,0.0
    def _best_action(self,task,workload,arrival,remaining,price_t,price_e):
        actions=self._completion_actions(task,workload,arrival[2],arrival[3],0.0,0.0)
        return (actions[0],float(self.weights[task])) if actions else (self._skip_action(),0.0)


class PriorityPlanner(HeuristicPlanner):
    def select_task(self, state):
        self.stats["route_calls"] += 1
        plan = self._plan_state(state)
        best = 0
        best_key = None
        for task in range(1, self.n_tasks + 1):
            if not plan.remaining_mask & (1 << (task - 1)):
                continue
            if self._arrival(plan, task) is None:
                continue
            key = (-float(self.weights[task]), float(self.mission.flight_time[plan.node, task]), task)
            if best_key is None or key < best_key:
                best, best_key = task, key
        return int(best), {"selection_mode": "priority_greedy"}


def _mean_completion(plan, task, planner_obj):
    """Return the deterministic arithmetic-mean workload service tuple.

    Cost and Mean-workload Greedy use exactly one workload value (the
    arithmetic mean) for route ranking.  The physical feasibility check and
    normalized service cost are still evaluated with the same execution model
    as the online planner.
    """
    arrival = planner_obj._arrival(plan, task)
    if arrival is None:
        return None
    _at, _ae, dt, de = arrival
    mean_c = float(np.exp(planner_obj.mu[task] + 0.5 * planner_obj.sigma[task] ** 2))
    ok, st, se, _mode, _freq, _cost = iteration_v2.execute_virtual_v2(
        mean_c, dt, de, planner_obj.pars, planner_obj.aa[task],
        planner_obj.bb[task], planner_obj.dd[task], planner_obj.ee[task],
        0, 1.0, 1.0, 0.0,
    )
    if not ok:
        return None
    normalized = (
        float(st) / max(float(dt), 1e-12)
        + float(se) / max(float(de), 1e-12)
    )
    return float(st), float(se), float(normalized)


class MeanWorkloadPlanner(HeuristicPlanner):
    """Greedy route ranked by priority divided by mean-workload cost."""

    def select_task(self, state):
        self.stats["route_calls"] += 1
        plan = self._plan_state(state)
        best = 0
        best_key = None
        for task in range(1, self.n_tasks + 1):
            if not plan.remaining_mask & (1 << (task - 1)):
                continue
            arrival = self._arrival(plan, task)
            if arrival is None:
                continue
            mean_service = _mean_completion(plan, task, self)
            if mean_service is None:
                continue
            service_t, service_e, _ = mean_service
            cost = (
                (self.mission.flight_time[plan.node, task] + service_t)
                / max(float(self.pars[0]), 1e-12)
                + (self.mission.flight_energy_kj[plan.node, task] + service_e)
                / max(float(self.pars[1]), 1e-12)
            )
            score = float(self.weights[task]) / max(float(cost), 1e-12)
            key = (-score, float(self.mission.flight_time[plan.node, task]), task)
            if best_key is None or key < best_key:
                best, best_key = task, key
        return int(best), {"selection_mode": "mean_workload_greedy"}


class MyopicPlanner(HeuristicPlanner):
    def select_task(self, state):
        self.stats["route_calls"] += 1
        plan = self._plan_state(state)
        best = 0
        best_key = None
        for task in range(1, self.n_tasks + 1):
            if not plan.remaining_mask & (1 << (task - 1)):
                continue
            arrival = self._arrival(plan, task)
            if arrival is None:
                continue
            p, _service_t, _service_e = robust_core.calibrated_task_stats(
                plan.node, plan.time_s, plan.energy_kj, task,
                self.price_points, self.price_probabilities, self.mu, self.sigma,
                self.mission.flight_time, self.mission.flight_energy_kj,
                self.deadline, self.pars, self.aa, self.bb, self.dd, self.ee, 0,
            )
            key = (-float(self.weights[task] * p),
                   float(self.mission.flight_time[plan.node, task]), task)
            if best_key is None or key < best_key:
                best, best_key = task, key
        return int(best), {"selection_mode": "distribution_aware_myopic"}


class NearestNeighborPlanner(HeuristicPlanner):
    def select_task(self, state):
        self.stats["route_calls"] += 1
        plan = self._plan_state(state)
        candidates = [
            task for task in range(1, self.n_tasks + 1)
            if plan.remaining_mask & (1 << (task - 1))
            and self._arrival(plan, task) is not None
        ]
        best = min(
            candidates,
            key=lambda task: (float(self.mission.flight_time[plan.node, task]), task),
            default=0,
        )
        return int(best), {"selection_mode": "nearest_neighbor"}
