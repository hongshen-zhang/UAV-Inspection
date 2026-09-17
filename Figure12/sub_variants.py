"""Single-factor Sub ablations of frozen U9; original source stays unchanged.

Price variants change actual-arrival completion cost only. The original Top
DP uses a normalized virtual-action proxy, not actual LP prices; that existing
approximation is deliberately retained. No-proactive-skip changes both the
virtual DP comparison and actual-arrival action set. Fixed CPU changes both
forecast and actual admissible frequency bounds.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from numba import njit, types
from numba.typed import Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'simulation'))
import run_mean_workload_dp as common
import ablation_base as original_ablation
from enhanced_dp import EnhancedPlanner, EnhancedContinuation, enhanced_value, grid, greedy_tail
from planner import ConsistentACARPlanner, PlanState
from root_calibrated_core import (_maximum_capacity, _lognormal_cdf,
    _truncated_lognormal_first_moment, _lognormal_ppf)
from batch_adapter import iteration_v2, robust_core

SPEC = common.SPEC
POLICY = common.BASE_POLICY
NAMES = {
    'proposed':'Proposed',
    'fixed_prices':'Fixed resource prices',
    'time_only_cost':'Time-only cost',
    'energy_only_cost':'Energy-only cost',
    'no_proactive_skip':'No proactive skip',
    'fixed_local_cpu':'Fixed local CPU',
}
DEFINITIONS = {
    'proposed':'Frozen U9 reference; existing nominal results reused.',
    'fixed_prices':'Actual Sub completion prices are 1/T and 1/E; original Top virtual-action proxy retained.',
    'time_only_cost':'Actual Sub completion prices are 1/T and 0; original Top virtual-action proxy retained.',
    'energy_only_cost':'Actual Sub completion prices are 0 and 1/E; original Top virtual-action proxy retained.',
    'no_proactive_skip':'Complete a feasible task in every virtual DP branch and actual Sub; skip remains for infeasible workload.',
    'fixed_local_cpu':'Set allowed local frequency to f_max=2.5 GHz in actual Sub and all forecasting arrays; retain all other rules.',
}


@njit(cache=True)
def sub_value(node,it,ie,mask,tasks,tailmask,time_step,energy_step,nt,ne,cache,tail_cache,
              moments,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,tail_weight,tail_depth,power,
              rounding,integration,allow_proactive_skip):
    """Frozen enhanced_value recurrence, with one explicit skip-choice switch."""
    t,e=it*time_step,ie*energy_step
    if t+ft[node,0]>pars[0]+1e-9 or e+fe[node,0]>pars[1]+1e-9:return 0.,0
    key=((mask*len(weights)+node)*nt+it)*ne+ie
    if key in cache:return cache[key]
    best,best_task=0.,0
    if tail_weight>0. and tailmask:
        tk=(node*nt+it)*ne+ie
        if tk in tail_cache:tv,ti=tail_cache[tk]
        else:
            tv,ti=greedy_tail(node,t,e,tailmask,tail_depth,1.,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee)
            tail_cache[tk]=tv,ti
        best,best_task=tail_weight*tv,ti
    bins=moments.shape[1]+1
    for index in range(len(tasks)):
        bit=1<<index
        if mask & bit==0:continue
        task=tasks[index]
        if not iteration_v2.visit_ok_v2(node,t,e,task,ft,fe,deadline,pars):continue
        at,ae,dt,de=iteration_v2.resource_limits_v2(node,t,e,task,ft,fe,deadline,pars)
        rem=mask & ~bit
        sit,sie=grid(at,time_step,rounding),grid(ae,energy_step,rounding)
        skip,_=sub_value(task,sit,sie,rem,tasks,tailmask,time_step,energy_step,nt,ne,cache,tail_cache,
            moments,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,tail_weight,tail_depth,power,
            rounding,integration,allow_proactive_skip)
        cap=_maximum_capacity(task,dt,de,pars,aa,bb,dd,ee)
        p=_lognormal_cdf(mu[task],sigma[task],cap)
        cap_moment=_truncated_lognormal_first_moment(mu[task],sigma[task],cap)
        value=(1-p)*skip;previous=0.
        for g in range(bins):
            pr=p/bins if integration==1 else min(p,(g+1)/bins)-g/bins
            if pr<=1e-14:break
            if integration==1:
                bound=cap if g==bins-1 else _lognormal_ppf(mu[task],sigma[task],p*(g+1)/bins)
                upper=_truncated_lognormal_first_moment(mu[task],sigma[task],bound)
            else:upper=cap_moment if p<=(g+1)/bins else moments[task,g]
            c=min(cap,(upper-previous)/pr);previous=upper
            ok,st,se,mode,freq,cost=robust_core.execute_virtual(c,dt,de,pars,aa[task],bb[task],dd[task],ee[task],0)
            v=skip
            if ok:
                cit,cie=grid(at+st,time_step,rounding),grid(ae+se,energy_step,rounding)
                future,_=sub_value(task,cit,cie,rem,tasks,tailmask,time_step,energy_step,nt,ne,cache,tail_cache,
                    moments,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,tail_weight,tail_depth,power,
                    rounding,integration,allow_proactive_skip)
                v=max(v,weights[task]+future) if allow_proactive_skip else weights[task]+future
            value+=pr*v
        if value>best+1e-12 or (abs(value-best)<=1e-12 and best_task>0 and task<best_task):
            best,best_task=value,task
    cache[key]=best,best_task
    return best,best_task


class SkipContinuation(EnhancedContinuation):
    def __init__(self,base,spec,allow_proactive_skip=False):
        super().__init__(base,spec)
        assert spec.tail_weight==0, 'This matched ablation is defined for frozen U9 without a tail.'
        self.allow_proactive_skip=bool(allow_proactive_skip)
        self.signature=hashlib.sha256(json.dumps({'spec':asdict(spec),
            'allow_proactive_skip':self.allow_proactive_skip},sort_keys=True).encode()).hexdigest()[:16]

    def plan(self,state):
        key=(state.node,state.time_s,state.energy_kj,state.remaining_mask)
        if key in self.cache:self.cache_hits+=1;return self.cache[key]
        b,s=self.base,self.spec
        ft,fe=b.bundle.iteration.flight_time,b.bundle.iteration.flight_energy_kj
        scores=original_ablation.screen_rows(b,state,s)
        tasks=np.array([-x[1] for x in sorted(scores,reverse=True)[:s.shortlist]],np.int64)
        tailmask=sum(1<<(-item[1]-1) for item in scores)
        for task in tasks:tailmask &= ~(1<<(int(task)-1))
        cache=Dict.empty(types.int64,types.Tuple((types.float64,types.int64)))
        tc=Dict.empty(types.int64,types.Tuple((types.float64,types.int64)))
        nt,ne=math.ceil(b.pars[0]/s.time_step)+3,math.ceil(b.pars[1]/s.energy_step)+3
        v,task=sub_value(state.node,grid(state.time_s,s.time_step,s.rounding),
            grid(state.energy_kj,s.energy_step,s.rounding),(1<<len(tasks))-1,tasks,np.uint64(tailmask),
            s.time_step,s.energy_step,nt,ne,cache,tc,self.moments,b.mu,b.sigma,ft,fe,b.weights,
            b.deadline,b.pars,b.aa,b.bb,b.dd,b.ee,s.tail_weight,s.tail_depth,s.cost_power,
            s.rounding,s.integration,self.allow_proactive_skip)
        self.expansions+=len(cache);self.cache[key]=int(task),float(v)
        return self.cache[key]


class AuditedSubPlanner(EnhancedPlanner):
    enhanced_spec=SPEC
    variant_id='proposed'

    def __init__(self,bundle,policy):
        super().__init__(bundle,policy)
        self.sub_audit=[]
        self.last_completions=()

    def _completion_actions(self,task,workload,available_t,available_e,price_t,price_e):
        result=super()._completion_actions(task,workload,available_t,available_e,price_t,price_e)
        self.last_completions=result
        return result

    def _best_action(self,task,workload,arrival,remaining,price_t,price_e):
        if self.variant_id=='no_proactive_skip':
            completions=self._completion_actions(task,workload,arrival[2],arrival[3],price_t,price_e)
            actions=completions if completions else (self._skip_action(),)
            scored=[(self._action_value(task,arrival[0],arrival[1],remaining,a),a) for a in actions]
            value,selected=max(scored,key=lambda item:(item[0],1 if item[1].mode!='skip' else 0,
                -item[1].occupation,-item[1].service_time_s,-item[1].service_energy_kj,-item[1].mec))
        else:
            selected,value=super()._best_action(task,workload,arrival,remaining,price_t,price_e)
        self.sub_audit.append({'task':int(task),'workload':float(workload),
            'arrival_time':float(arrival[0]),'arrival_energy':float(arrival[1]),
            'available_time':float(arrival[2]),'available_energy':float(arrival[3]),
            'price_t':float(price_t),'price_e':float(price_e),'selected_mode':selected.mode,
            'selected_mec':int(selected.mec),'selected_frequency':float(selected.frequency_ghz),
            'feasible_completion':bool(self.last_completions),
            'proactive_skip':selected.mode=='skip' and bool(self.last_completions),
            'completion_candidates':[asdict(a) for a in self.last_completions]})
        return selected,float(value)


class FixedPricesPlanner(AuditedSubPlanner):
    variant_id='fixed_prices'
    def _prices(self,task,arrival_t,arrival_e,remaining):
        self.stats['fixed_price_calls']=self.stats.get('fixed_price_calls',0)+1
        return 1./float(self.pars[0]),1./float(self.pars[1])


class TimeOnlyPlanner(AuditedSubPlanner):
    variant_id='time_only_cost'
    def _prices(self,task,arrival_t,arrival_e,remaining):
        self.stats['time_only_price_calls']=self.stats.get('time_only_price_calls',0)+1
        return 1./float(self.pars[0]),0.


class EnergyOnlyPlanner(AuditedSubPlanner):
    variant_id='energy_only_cost'
    def _prices(self,task,arrival_t,arrival_e,remaining):
        self.stats['energy_only_price_calls']=self.stats.get('energy_only_price_calls',0)+1
        return 0.,1./float(self.pars[1])


class NoProactiveSkipPlanner(AuditedSubPlanner):
    variant_id='no_proactive_skip'
    def __init__(self,bundle,policy):
        super().__init__(bundle,policy)
        self.continuation=SkipContinuation(self.continuation.base,self.enhanced_spec,False)


class FixedLocalCPUPlanner(AuditedSubPlanner):
    variant_id='fixed_local_cpu'
    def __init__(self,bundle,policy):
        super().__init__(bundle,policy)
        self.pars=self.pars.copy();self.pars[2]=self.pars[3]
        self.continuation.base.pars=self.continuation.base.pars.copy()
        self.continuation.base.pars[2]=self.continuation.base.pars[3]
        self.continuation.signature=hashlib.sha256(json.dumps({'spec':asdict(self.enhanced_spec),
            'fixed_local_cpu':float(self.pars[3])},sort_keys=True).encode()).hexdigest()[:16]


CLASSES={'proposed':AuditedSubPlanner,'fixed_prices':FixedPricesPlanner,
    'time_only_cost':TimeOnlyPlanner,'energy_only_cost':EnergyOnlyPlanner,
    'no_proactive_skip':NoProactiveSkipPlanner,'fixed_local_cpu':FixedLocalCPUPlanner}


def audit_actions(instance, row):
    """Check actual completion minimizers and forced-skip/frequency hooks."""
    logs, trace = instance.sub_audit, json.loads(row['trace'])
    assert len(logs) == len(trace)
    fmax = float(instance.pars[3])
    energy_minimizer = (instance.pars[5] / (2 * instance.pars[4])) ** (1 / 3)
    assert energy_minimizer > fmax
    max_gap = 0.
    for event, actual in zip(logs, trace):
        task, c = event['task'], event['workload']
        dt, de = event['available_time'], event['available_energy']
        pt, pe = event['price_t'], event['price_e']
        assert task == actual['task'] and c == actual['revealed_workload_gcy']
        assert event['selected_mode'] == actual['mode'] and event['selected_mec'] == actual['mec']
        if instance.variant_id == 'fixed_prices': assert (pt, pe) == (1 / instance.pars[0], 1 / instance.pars[1])
        if instance.variant_id == 'time_only_cost': assert (pt, pe) == (1 / instance.pars[0], 0.)
        if instance.variant_id == 'energy_only_cost': assert (pt, pe) == (0., 1 / instance.pars[1])
        if instance.variant_id == 'no_proactive_skip': assert not event['proactive_skip']
        candidates = []
        st = c / fmax
        se = instance.pars[4] * c * fmax**2 + instance.pars[5] * st
        if st <= dt + 1e-9 and se <= de + 1e-9: candidates.append(('local', -1, st, se))
        for m in range(instance.aa.shape[1]):
            if instance.aa[task, m] >= 1e90: continue
            st = instance.aa[task, m] + instance.bb[task, m] * c
            se = instance.dd[task, m] + instance.ee[task, m] * c
            if st <= dt + 1e-9 and se <= de + 1e-9: candidates.append(('mec', m, st, se))
        assert bool(candidates) == event['feasible_completion']
        if instance.variant_id == 'no_proactive_skip':
            assert (actual['mode'] != 'skip') == bool(candidates)
            assert not instance.continuation.allow_proactive_skip
        if candidates:
            chosen = event['completion_candidates'][0]
            expected = min(candidates, key=lambda a: (pt*a[2] + pe*a[3], a[2]/dt + a[3]/de,
                                                       a[2], a[3], a[1]))
            assert chosen['mode'] == expected[0] and chosen['mec'] == expected[1]
            gap = pt*chosen['service_time_s'] + pe*chosen['service_energy_kj'] - pt*expected[2] - pe*expected[3]
            max_gap = max(max_gap, abs(gap)); assert abs(gap) < 1e-9
        if actual['mode'] == 'local': assert abs(actual['frequency_ghz'] - fmax) < 1e-9
    return dict(sub_audit=json.dumps(logs, separators=(',', ':')), sub_arrival_checks=len(logs),
                proactive_skips=sum(e['proactive_skip'] for e in logs),
                infeasible_skips=sum(e['selected_mode'] == 'skip' and not e['feasible_completion'] for e in logs),
                sub_completion_cost_checks=sum(e['feasible_completion'] for e in logs),
                max_completion_objective_gap=max_gap,
                forecast_proactive_skip_allowed=instance.variant_id != 'no_proactive_skip',
                forecast_local_fmin=float(instance.continuation.base.pars[2]),
                actual_local_fmin=float(instance.pars[2]), local_energy_minimizer_ghz=float(energy_minimizer))
