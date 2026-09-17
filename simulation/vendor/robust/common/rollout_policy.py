"""Nonanticipative scenario-rollout policy for robust online MTE planning.

The controller evaluates promising next tasks with common, mission-independent
workload scenarios.  The first task is fixed within each rollout, while all
later choices use only the current simulated state and the prior distributions.
Consequently, future decisions react to resources consumed by revealed
workloads but never inspect the workload of an unvisited task.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from statistics import NormalDist
import numpy as np

from .simulator import State, solve_execution, finish_return, visit_feasible
from .mte_core import prepare_arrays, quantile_workload_points, calibrated_task_stats
from .rollout_core import rollout_choose
from .rollout_mean_core import rollout_choose_mean


@dataclass(frozen=True)
class RolloutConfig:
    scenarios: int = 16
    shortlist: int = 6
    base_quadrature: int = 3
    cost_floor: float = 0.08
    cost_power: float = 1.0
    cluster_weight: float = 0.10
    max_future_steps: int = 20
    scenario_seed: int = 20260831
    risk_weight: float = 0.0
    lower_tail_fraction: float = 0.25
    time_reserve_s: float = 0.0
    energy_reserve_kj: float = 0.0
    base_modes: int = 2
    profile_count: int = 2


def _copy_state(s: State) -> State:
    return State(s.node, s.time_s, s.energy_kj, set(s.visited), set(s.completed),
                 list(s.route), s.local_actions, s.mec_actions, s.skipped)


def _apply_workload(mission, state: State, i: int, workload: float):
    old=_copy_state(state)
    state.time_s += mission.flight_time[old.node,i]
    state.energy_kj += mission.flight_energy_kj[old.node,i]
    state.node=i; state.visited.add(i); state.route.append(i)
    ex=solve_execution(mission,old,i,float(workload),'auto')
    if ex.feasible:
        state.time_s += ex.time_s; state.energy_kj += ex.energy_kj
        state.completed.add(i)
        if ex.mode=='local': state.local_actions += 1
        elif ex.mode=='mec': state.mec_actions += 1
    else:
        state.skipped += 1
    return ex


def _scenario_z(n_scenarios: int, n_tasks: int, seed: int) -> np.ndarray:
    """Latin-hypercube normal scenarios with exact marginal strata."""
    n=max(2,int(n_scenarios))
    rng=np.random.default_rng(seed)
    nd=NormalDist()
    z=np.empty((n,n_tasks),dtype=float)
    mid=(np.arange(n,dtype=float)+0.5)/n
    base=np.asarray([nd.inv_cdf(float(u)) for u in mid],dtype=float)
    for j in range(n_tasks):
        z[:,j]=base[rng.permutation(n)]
    return z


def _scenario_workloads(mu: np.ndarray, sigma: np.ndarray, rcfg: RolloutConfig, epoch: int) -> np.ndarray:
    # This bank is independent of the evaluated mission seed.  Epoch-dependent
    # deterministic shifts avoid repeatedly using the same joint pairing.
    z=_scenario_z(rcfg.scenarios,len(mu)-1,rcfg.scenario_seed+104729*int(epoch))
    return np.exp(mu[1:][None,:]+sigma[1:][None,:]*z)


def _static_cluster_bonus(mission, i: int, unvisited: list[int]) -> float:
    vals=[]
    for j in unvisited:
        if j==i: continue
        w=mission.tasks[j-1].weight
        vals.append(w/(1.0+mission.flight_time[i,j]/240.0))
    if not vals: return 0.0
    vals.sort(reverse=True)
    return sum(vals[:3])/45.0


def _fast_scores(mission,state:State,arrays,points,probs,rcfg:RolloutConfig):
    """Readable Python version of the distribution-aware candidate score."""
    weights,deadline,mu,sigma,pars,aa,bb,dd,ee=arrays
    rem_t=max(1e-9,mission.cfg.t_max-state.time_s-mission.flight_time[state.node,0])
    rem_e=max(1e-9,mission.cfg.e_max_kj-state.energy_kj-mission.flight_energy_kj[state.node,0])
    unvisited=[t.idx for t in mission.tasks if t.idx not in state.visited]
    out=[]
    for i in unvisited:
        if not visit_feasible(mission,state,i): continue
        p,et,en=calibrated_task_stats(state.node,state.time_s,state.energy_kj,i,
                points,probs,mu,sigma,mission.flight_time,mission.flight_energy_kj,
                deadline,pars,aa,bb,dd,ee,0)
        if p<=1e-12: continue
        inc_ft=mission.flight_time[state.node,i]+mission.flight_time[i,0]-mission.flight_time[state.node,0]
        inc_fe=mission.flight_energy_kj[state.node,i]+mission.flight_energy_kj[i,0]-mission.flight_energy_kj[state.node,0]
        cost=max(0.0,inc_ft)/rem_t + max(0.0,inc_fe)/rem_e + et/rem_t + en/rem_e
        value=weights[i]*p
        score=value/((rcfg.cost_floor+cost)**rcfg.cost_power)
        if rcfg.cluster_weight:
            score += rcfg.cluster_weight*weights[i]*_static_cluster_bonus(mission,i,unvisited)
        out.append((float(score),int(i),float(p),float(cost)))
    out.sort(key=lambda x:(-x[0],x[1]))
    return out


def _base_choice(mission,state,arrays,points,probs,rcfg):
    scores=_fast_scores(mission,state,arrays,points,probs,rcfg)
    return 0 if not scores else int(scores[0][1])


def _simulate_after_candidate(mission,state,candidate,scenario,arrays,points,probs,rcfg):
    """Slow reference implementation used by verification tests."""
    s=_copy_state(state)
    _apply_workload(mission,s,candidate,scenario[candidate-1])
    for _ in range(rcfg.max_future_steps):
        i=_base_choice(mission,s,arrays,points,probs,rcfg)
        if i==0: break
        _apply_workload(mission,s,i,scenario[i-1])
    finish_return(mission,s)
    return sum(mission.tasks[i-1].weight for i in s.completed)


def run_rollout_policy_reference(mission, rcfg:RolloutConfig|None=None, record_trace:bool=False):
    """Readable reference policy; the fast policy below is used for experiments."""
    rcfg=rcfg or RolloutConfig(base_modes=1,profile_count=1)
    cfg=mission.cfg
    arrays=prepare_arrays(mission)
    weights,deadline,mu,sigma,pars,aa,bb,dd,ee=arrays
    points,probs=quantile_workload_points(mu,sigma,rcfg.base_quadrature)
    state=State();traces=[];planner_time=0.0;start=time.perf_counter()
    for epoch in range(cfg.n_tasks):
        t0=time.perf_counter()
        fast=_fast_scores(mission,state,arrays,points,probs,rcfg)
        if not fast: break
        candidates=[x[1] for x in fast[:max(1,rcfg.shortlist)]]
        scenario_bank=_scenario_workloads(mu,sigma,rcfg,epoch)
        best_i=0;best_q=-1e100;best_fast=-1e100
        fast_map={i:s for s,i,_,_ in fast}
        for i in candidates:
            indiv=np.empty(len(scenario_bank),dtype=float)
            for ss,scen in enumerate(scenario_bank):
                indiv[ss]=_simulate_after_candidate(mission,state,i,scen,arrays,points,probs,rcfg)
            mean_q=float(np.mean(indiv))
            if rcfg.risk_weight>0.0:
                k=max(1,int(math.ceil(rcfg.lower_tail_fraction*len(indiv))))
                lower=float(np.mean(np.sort(indiv)[:k]))
                q=mean_q-rcfg.risk_weight*(mean_q-lower)
            else:
                q=mean_q
            fs=fast_map[i]
            if q>best_q+1e-12 or (abs(q-best_q)<=1e-12 and (fs>best_fast+1e-12 or (abs(fs-best_fast)<=1e-12 and (best_i==0 or i<best_i)))):
                best_i=i;best_q=q;best_fast=fs
        planner_time += time.perf_counter()-t0
        if best_i==0: break
        ex=_apply_workload(mission,state,best_i,mission.tasks[best_i-1].workload_gcy)
        if record_trace:
            task=mission.tasks[best_i-1]
            traces.append(dict(epoch=epoch,task=best_i,priority=task.weight,workload=task.workload_gcy,
                               completed=int(ex.feasible),mode=ex.mode,time_s=state.time_s,
                               energy_kj=state.energy_kj,rollout_value=float(best_q),
                               candidate_count=len(candidates),scenario_count=len(scenario_bank)))
    return _finish_output(mission,state,'Proposed-RCR-Reference',start,planner_time,traces if record_trace else None)


def _visited_mask_fast(state: State):
    mask=np.uint64(0)
    for i in state.visited:
        mask |= np.uint64(1)<<np.uint64(i-1)
    return mask


def _finish_output(mission,state,method,start,planner_time,traces=None):
    cfg=mission.cfg
    finish_return(mission,state)
    runtime=time.perf_counter()-start
    total_w=sum(t.weight for t in mission.tasks)
    done_w=sum(mission.tasks[i-1].weight for i in state.completed)
    out=dict(seed=mission.seed,method=method,weighted_rate=done_w/total_w,
             task_rate=len(state.completed)/cfg.n_tasks,completed_weight=done_w,
             completed_tasks=len(state.completed),visited_tasks=len(state.visited),
             time_s=state.time_s,energy_kj=state.energy_kj,
             return_ok=float(state.time_s<=cfg.t_max+1e-8 and state.energy_kj<=cfg.e_max_kj+1e-8),
             local_actions=state.local_actions,mec_actions=state.mec_actions,skipped=state.skipped,
             runtime_s=runtime,planner_runtime_s=planner_time,beam_states=0,
             route='-'.join(map(str,state.route)))
    if traces is not None: out['_trace']=traces
    return out


def run_rollout_policy_fast(mission, rcfg:RolloutConfig|None=None, record_trace:bool=False):
    """Numba-accelerated calibrated nonanticipative recourse rollout."""
    rcfg=rcfg or RolloutConfig()
    cfg=mission.cfg
    arrays=prepare_arrays(mission)
    weights,deadline,mu,sigma,pars,aa,bb,dd,ee=arrays
    points,probs=quantile_workload_points(mu,sigma,rcfg.base_quadrature)
    pars_plan=pars.copy()
    pars_plan[0]=max(1.0,pars[0]-rcfg.time_reserve_s)
    pars_plan[1]=max(1.0,pars[1]-rcfg.energy_reserve_kj)
    state=State();traces=[];planner_time=0.0;start=time.perf_counter()
    for epoch in range(cfg.n_tasks):
        bank0=_scenario_workloads(mu,sigma,rcfg,epoch)
        bank=np.zeros((bank0.shape[0],bank0.shape[1]+1),dtype=np.float64)
        bank[:,1:]=bank0
        vm=_visited_mask_fast(state);t0=time.perf_counter()

        # Fixed policy portfolio.  Every member is evaluated on the same
        # common-random-number scenario bank.
        profiles=[(rcfg.shortlist,rcfg.cost_floor,rcfg.cost_power,
                   rcfg.cluster_weight,rcfg.risk_weight)]
        if rcfg.profile_count > 1:
            profiles.append((5,0.11,1.10,0.05,0.0))   # tight-budget efficiency
        if rcfg.profile_count > 2:
            profiles.append((6,0.05,0.90,0.15,0.10)) # value/cluster seeking

        i=0;q=-1e100;ncand=0;chosen_mode=0
        for pid,(pk,pcf,pcp,pcl,prisk) in enumerate(profiles):
            ip,qp,np_=rollout_choose(state.node,state.time_s,state.energy_kj,vm,bank,
                    points,probs,mu,sigma,mission.flight_time,mission.flight_energy_kj,
                    weights,deadline,pars_plan,aa,bb,dd,ee,pk,pcf,pcp,pcl,
                    rcfg.max_future_steps,prisk,rcfg.lower_tail_fraction)
            ncand+=np_
            if qp>q+1e-12:
                i,q,chosen_mode=ip,qp,10+pid

        if rcfg.base_modes > 1:
            i1,q1,n1=rollout_choose_mean(state.node,state.time_s,state.energy_kj,vm,bank,
                    mission.flight_time,mission.flight_energy_kj,weights,deadline,mu,sigma,
                    pars_plan,aa,bb,dd,ee,rcfg.shortlist,rcfg.cost_floor,rcfg.cost_power,
                    rcfg.cluster_weight,rcfg.max_future_steps,rcfg.risk_weight,
                    rcfg.lower_tail_fraction)
            if q1>q+1e-12:
                i,q,chosen_mode=i1,q1,1
            ncand+=n1

        planner_time+=time.perf_counter()-t0;i=int(i)
        if i==0: break
        ex=_apply_workload(mission,state,i,mission.tasks[i-1].workload_gcy)
        if record_trace:
            task=mission.tasks[i-1]
            traces.append(dict(epoch=epoch,task=i,priority=task.weight,
                               workload=task.workload_gcy,completed=int(ex.feasible),
                               mode=ex.mode,time_s=state.time_s,energy_kj=state.energy_kj,
                               rollout_value=float(q),candidate_count=int(ncand),
                               scenario_count=bank.shape[0],continuation_mode=int(chosen_mode)))
    return _finish_output(mission,state,'Proposed-RCR',start,planner_time,
                          traces if record_trace else None)
