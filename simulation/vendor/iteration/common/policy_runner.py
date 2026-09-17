from __future__ import annotations
import time, math
import numpy as np
from .simulator import (State, apply_actual_task, apply_actual_task_v2,
                        finish_return, visit_feasible, solve_execution)
from .mte_core import (prepare_arrays, gh_workload_points, mean_workload_points, saa_workload_points,
                       top_choose, myopic_choose)
from .mte_v2 import quantile_workload_points, top_choose_adaptive_v2

METHOD_LABELS={
    'Proposed':'Proposed',
    'ProposedV2':'Proposed V2',
    'MeanRollout':'Mean Rollout',
    'SAA5':'SAA-5',
    'MyopicEV':'Myopic EV',
    'Greedy':'Greedy',
    'Nearest':'Nearest',
    'NoFuture':'Without Future Value',
    'NoLP':'Without LP Guidance',
    'LocalOnly':'Local Only',
    'MECOnly':'MEC Only',
}


def _visited_mask(state):
    mask=np.uint64(0)
    for i in state.visited:mask |= (np.uint64(1)<<np.uint64(i-1))
    return mask


def _planning_points(method,mu,sigma,cfg,seed,epoch,gh_order=None):
    if method in ('Proposed','NoFuture','NoLP','LocalOnly','MECOnly','MyopicEV'):
        return gh_workload_points(mu,sigma,cfg.gh_order if gh_order is None else gh_order)
    if method=='MeanRollout':return mean_workload_points(mu,sigma)
    if method=='SAA5':return saa_workload_points(mu,sigma,cfg.saa_samples,seed,epoch)
    raise ValueError(method)


def _greedy_choice(mission,state):
    """Immediate priority-to-resource-efficiency baseline.

    It uses the lognormal mean workload but has no future-route evaluation.
    """
    cand=[t.idx for t in mission.tasks if t.idx not in state.visited and visit_feasible(mission,state,t.idx)]
    if not cand:return 0
    best_i=0; best_score=-1.0
    for i in cand:
        task=mission.tasks[i-1]
        cmean=math.exp(task.mu+0.5*task.sigma*task.sigma)
        ex=solve_execution(mission,state,i,cmean,'auto')
        if not ex.feasible:
            continue
        ft=mission.flight_time[state.node,i]; fe=mission.flight_energy_kj[state.node,i]
        cost=(ft+ex.time_s)/mission.cfg.t_max + (fe+ex.energy_kj)/mission.cfg.e_max_kj
        score=task.weight/max(cost,1e-12)
        if score>best_score+1e-12 or (abs(score-best_score)<=1e-12 and (best_i==0 or ft<mission.flight_time[state.node,best_i])):
            best_i=i; best_score=score
    return best_i


def _nearest_choice(mission,state):
    cand=[t.idx for t in mission.tasks if t.idx not in state.visited and visit_feasible(mission,state,t.idx)]
    if not cand:return 0
    return min(cand,key=lambda i:(mission.flight_time[state.node,i],i))


def run_policy(mission,method='Proposed',beam_width=None,gh_order=None,record_trace=False,v2_params=None):
    cfg=mission.cfg; beam_width=cfg.beam_width if beam_width is None else int(beam_width)
    v2=dict(v2_params or {})
    v2_time_weight=float(v2.get('time_weight',1.0))
    v2_energy_weight=float(v2.get('energy_weight',1.0))
    v2_balance_weight=float(v2.get('balance_weight',0.0))
    v2_order=int(v2.get('order',gh_order if gh_order is not None else cfg.gh_order))
    v2_quadrature=str(v2.get('quadrature','quantile')).lower()
    v2_beam_width=int(v2.get('beam_width',beam_width))
    v2_use_lp=bool(v2.get('use_lp',True))
    v2_lp_weight=float(v2.get('lp_weight',1.0))
    v2_tighten_return=bool(v2.get('tighten_return',True))
    v2_min_completion_prob=float(v2.get('min_completion_prob',0.0))
    v2_risk_lambda=float(v2.get('risk_lambda',0.0))
    weights,deadline,mu,sigma,pars,aa,bb,dd,ee=prepare_arrays(mission)
    state=State(); traces=[]; expanded_total=0; planner_time=0.0
    force_mode=0; force_name='auto'
    if method=='LocalOnly':force_mode=1;force_name='local'
    if method=='MECOnly':force_mode=2;force_name='mec'
    start=time.perf_counter()
    for epoch in range(cfg.n_tasks):
        if method=='Greedy':i=_greedy_choice(mission,state); jval=jshort=jfuture=0.0; expanded=0
        elif method=='Nearest':i=_nearest_choice(mission,state); jval=jshort=jfuture=0.0; expanded=0
        else:
            vm=_visited_mask(state); t0=time.perf_counter()
            if method=='ProposedV2':
                if v2_quadrature=='gh':
                    points,probs=gh_workload_points(mu,sigma,v2_order)
                elif v2_quadrature=='quantile':
                    points,probs=quantile_workload_points(mu,sigma,v2_order)
                else:
                    raise ValueError(f'Unknown V2 quadrature: {v2_quadrature}')
                i,jval,jshort,jfuture,expanded=top_choose_adaptive_v2(
                        state.node,state.time_s,state.energy_kj,vm,points,probs,
                        mission.flight_time,mission.flight_energy_kj,weights,deadline,pars,aa,bb,dd,ee,
                        force_mode,v2_time_weight,v2_energy_weight,v2_balance_weight,
                        v2_beam_width,v2_use_lp,v2_lp_weight,v2_tighten_return,
                        v2_min_completion_prob,v2_risk_lambda)
                i=int(i)
            else:
                points,probs=_planning_points(method,mu,sigma,cfg,mission.seed,epoch,gh_order)
                if method=='MyopicEV':
                    i=int(myopic_choose(state.node,state.time_s,state.energy_kj,vm,points,probs,
                            mission.flight_time,mission.flight_energy_kj,weights,deadline,pars,aa,bb,dd,ee,force_mode))
                    jval=jshort=jfuture=0.0;expanded=0
                else:
                    with_future=(method!='NoFuture'); use_lp=(method!='NoLP')
                    i,jval,jshort,jfuture,expanded=top_choose(state.node,state.time_s,state.energy_kj,vm,points,probs,
                            mission.flight_time,mission.flight_energy_kj,weights,deadline,pars,aa,bb,dd,ee,
                            force_mode,beam_width,with_future,use_lp)
                    i=int(i)
            planner_time+=time.perf_counter()-t0; expanded_total+=int(expanded)
        if i==0:break
        if method=='ProposedV2':
            ex=apply_actual_task_v2(mission,state,i,force_name,v2_time_weight,
                                    v2_energy_weight,v2_balance_weight)
        else:
            ex=apply_actual_task(mission,state,i,force_name)
        if record_trace:
            task=mission.tasks[i-1]
            traces.append(dict(epoch=epoch,task=i,lon=task.lon,lat=task.lat,priority=task.weight,
                               workload=task.workload_gcy,completed=int(ex.feasible),mode=ex.mode,
                               cpu_ghz=ex.cpu_ghz,normalized_cost=(ex.normalized_cost if ex.feasible else np.nan),
                               time_s=state.time_s,energy_kj=state.energy_kj,
                               planning_value=float(jval),short_value=float(jshort),future_value=float(jfuture)))
    finish_return(mission,state)
    runtime=time.perf_counter()-start
    total_w=sum(t.weight for t in mission.tasks); done_w=sum(mission.tasks[i-1].weight for i in state.completed)
    nominal_return=float(state.time_s<=cfg.t_max+1e-8 and state.energy_kj<=cfg.e_max_kj+1e-8)
    out=dict(seed=mission.seed,method=method,weighted_rate=done_w/total_w,task_rate=len(state.completed)/cfg.n_tasks,
             completed_weight=done_w,completed_tasks=len(state.completed),visited_tasks=len(state.visited),
             time_s=state.time_s,energy_kj=state.energy_kj,return_ok=nominal_return,
             local_actions=state.local_actions,mec_actions=state.mec_actions,skipped=state.skipped,
             runtime_s=runtime,planner_runtime_s=planner_time,beam_states=expanded_total,
             route='-'.join(map(str,state.route)))
    if method=='ProposedV2':
        out.update(v2_quadrature=v2_quadrature,v2_order=v2_order,
                   v2_beam_width=v2_beam_width,v2_time_weight=v2_time_weight,
                   v2_energy_weight=v2_energy_weight,
                   v2_balance_weight=v2_balance_weight,
                   v2_lp_weight=v2_lp_weight,
                   v2_tighten_return=int(v2_tighten_return),
                   v2_min_completion_prob=v2_min_completion_prob,
                   v2_risk_lambda=v2_risk_lambda)
    if record_trace:out['_trace']=traces
    return out


def warm_up(mission):
    # Compile Numba kernels on a tiny state before timing paper runs.
    weights,deadline,mu,sigma,pars,aa,bb,dd,ee=prepare_arrays(mission)
    pts,pr=gh_workload_points(mu,sigma,1)
    _=top_choose(0,0.0,0.0,np.uint64(0),pts,pr,mission.flight_time,mission.flight_energy_kj,
                 weights,deadline,pars,aa,bb,dd,ee,0,1,False,False)
    _=top_choose_adaptive_v2(0,0.0,0.0,np.uint64(0),pts,pr,
                 mission.flight_time,mission.flight_energy_kj,weights,deadline,
                 pars,aa,bb,dd,ee,0,1.0,1.0,0.0,1,False,1.0,False,0.0,0.0)
