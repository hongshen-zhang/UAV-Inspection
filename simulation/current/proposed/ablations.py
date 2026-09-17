"""Published Figure 7 ablations, retaining original prediction/action rules."""
from __future__ import annotations
from dataclasses import asdict
import hashlib,json,math
import numpy as np
from numba import njit,types
from numba.typed import Dict
from . import physics as iteration_v2
from . import physics as robust_core
from .physics import (_maximum_capacity,_lognormal_cdf,
    _truncated_lognormal_first_moment,_lognormal_ppf)
from .dp import EnhancedSpec,EnhancedContinuation,enhanced_value,grid,greedy_tail
from .controller import BasePlanner
SPEC=EnhancedSpec(shortlist=9,time_step=25.,energy_step=7.5,quadrature=3,
    tail_weight=0.,tail_depth=6,cost_power=.5,rounding=1,mode='both',
    arrival_rule='original',integration=0)

def screen_rows(base, state, spec=SPEC):
    """Exact frozen U9 screen statistics, before resource-grid rounding."""
    ft, fe = base.bundle.iteration.flight_time, base.bundle.iteration.flight_energy_kj
    rows = []
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
        cost = ((ft[state.node, task] + st) / max(base.pars[0] - state.time_s, 1e-12)
              + (fe[state.node, task] + se) / max(base.pars[1] - state.energy_kj, 1e-12))
        score = float(base.weights[task] * p) / max(cost, 1e-12) ** spec.cost_power
        rows.append((score, -task))
    return rows


class ShortlistContinuation(EnhancedContinuation):
    """Same DP call as EnhancedContinuation, with an explicit ordering hook."""
    ordering = 'score'

    def __init__(self, base, spec):
        super().__init__(base, spec)
        self.shortlist_calls = self.shortlist_changed_calls = 0
        self.shortlist_set_changed_calls = 0
        self.last_tasks = []

    def selected_tasks(self, state, scores):
        original = [-item[1] for item in sorted(scores, reverse=True)[:self.spec.shortlist]]
        if self.ordering == 'distance':
            ft = self.base.bundle.iteration.flight_time
            chosen = sorted((-item[1] for item in scores),
                key=lambda task: (float(ft[state.node, task]), task))[:self.spec.shortlist]
        else:
            chosen = original
        self.shortlist_calls += 1
        self.shortlist_changed_calls += int(chosen != original)
        self.shortlist_set_changed_calls += int(set(chosen) != set(original))
        self.last_tasks = chosen
        return np.asarray(chosen, dtype=np.int64)

    def plan(self, state):
        key = (state.node, state.time_s, state.energy_kj, state.remaining_mask)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        b, s = self.base, self.spec
        ft, fe = b.bundle.iteration.flight_time, b.bundle.iteration.flight_energy_kj
        scores = screen_rows(b, state, s)
        tasks = self.selected_tasks(state, scores)
        tailmask = sum(1 << (-item[1] - 1) for item in scores)
        for task in tasks:
            tailmask &= ~(1 << (int(task) - 1))
        cache = Dict.empty(types.int64, types.Tuple((types.float64, types.int64)))
        tc = Dict.empty(types.int64, types.Tuple((types.float64, types.int64)))
        nt = math.ceil(b.pars[0] / s.time_step) + 3
        ne = math.ceil(b.pars[1] / s.energy_step) + 3
        value, task = enhanced_value(state.node, grid(state.time_s, s.time_step, s.rounding),
            grid(state.energy_kj, s.energy_step, s.rounding), (1 << len(tasks)) - 1,
            tasks, np.uint64(tailmask), s.time_step, s.energy_step, nt, ne, cache, tc,
            self.moments, b.mu, b.sigma, ft, fe, b.weights, b.deadline, b.pars,
            b.aa, b.bb, b.dd, b.ee, s.tail_weight, s.tail_depth, s.cost_power,
            s.rounding, s.integration)
        self.expansions += len(cache)
        self.cache[key] = int(task), float(value)
        return self.cache[key]


@njit(cache=True)
def immediate_value(node, it, ie, tasks, moments, mu, sigma, ft, fe, weights,
                    deadline, pars, aa, bb, dd, ee, time_step, energy_step):
    """Frozen G=3 U9 root integration with both successor values set to zero."""
    t, e = it * time_step, ie * energy_step
    if t + ft[node, 0] > pars[0] + 1e-9 or e + fe[node, 0] > pars[1] + 1e-9:
        return 0., 0, 0
    best, best_task, branches = 0., 0, 0
    bins = moments.shape[1] + 1
    for task in tasks:
        if not iteration_v2.visit_ok_v2(node, t, e, task, ft, fe, deadline, pars):
            continue
        at, ae, dt, de = iteration_v2.resource_limits_v2(node, t, e, task, ft, fe, deadline, pars)
        cap = _maximum_capacity(task, dt, de, pars, aa, bb, dd, ee)
        p = _lognormal_cdf(mu[task], sigma[task], cap)
        cap_moment = _truncated_lognormal_first_moment(mu[task], sigma[task], cap)
        value, previous = 0., 0.
        for g in range(bins):
            pr = min(p, (g + 1) / bins) - g / bins
            if pr <= 1e-14:
                break
            upper = cap_moment if p <= (g + 1) / bins else moments[task, g]
            workload = min(cap, (upper - previous) / pr)
            previous = upper
            ok, st, se, mode, freq, cost = robust_core.execute_virtual(workload, dt, de,
                pars, aa[task], bb[task], dd[task], ee[task], 0)
            value += pr * (weights[task] if ok else 0.)
            branches += 1
        if value > best + 1e-12 or (abs(value - best) <= 1e-12 and best_task > 0 and task < best_task):
            best, best_task = value, task
    return best, best_task, branches


class MyopicContinuation(ShortlistContinuation):
    def __init__(self, base, spec):
        super().__init__(base, spec)
        self.one_step_branches = self.zero_future_calls = 0

    def plan(self, state):
        key = (state.node, state.time_s, state.energy_kj, state.remaining_mask)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        b, s = self.base, self.spec
        tasks = self.selected_tasks(state, screen_rows(b, state, s))
        value, task, branches = immediate_value(state.node,
            grid(state.time_s, s.time_step, s.rounding),
            grid(state.energy_kj, s.energy_step, s.rounding), tasks,
            self.moments, b.mu, b.sigma, b.bundle.iteration.flight_time,
            b.bundle.iteration.flight_energy_kj, b.weights, b.deadline, b.pars,
            b.aa, b.bb, b.dd, b.ee, s.time_step, s.energy_step)
        self.one_step_branches += branches
        self.cache[key] = int(task), float(value)
        return self.cache[key]

    def value(self, state, immediate):
        self.calls += 1
        self.zero_future_calls += 1
        return float(immediate)


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
        scores=screen_rows(b,state,s)
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



class NoPriorityPlanner(BasePlanner):
    def __init__(self, public_arrays, **options):
        super().__init__(public_arrays, **options)
        self.continuation=ScoreContinuation(self.continuation.base,self.enhanced_spec,'no_priority')


class TopMyopicPlanner(BasePlanner):
    def __init__(self, public_arrays, **options):
        super().__init__(public_arrays, **options)
        self.top_selector=MyopicContinuation(self.continuation.base,self.enhanced_spec)

    def select_task(self, state):
        self.stats['route_calls'] += 1
        task,value=self.top_selector.plan(self._plan_state(state))
        return task, {'selection_mode':'top_only_immediate_reward','selected_value':value,
                      'continuation_signature':self.continuation.signature}


class NoProactiveSkipPlanner(BasePlanner):
    def __init__(self, public_arrays, **options):
        super().__init__(public_arrays, **options)
        self.continuation=SkipContinuation(self.continuation.base,self.enhanced_spec,False)

    def _best_action(self,task,workload,arrival,remaining,price_t,price_e):
        completions=self._completion_actions(task,workload,arrival[2],arrival[3],price_t,price_e)
        actions=completions if completions else (self._skip_action(),)
        scored=[(self._action_value(task,arrival[0],arrival[1],remaining,a),a) for a in actions]
        value,selected=max(scored,key=lambda item:(item[0],1 if item[1].mode!='skip' else 0,
            -item[1].occupation,-item[1].service_time_s,-item[1].service_energy_kj,-item[1].mec))
        return selected,float(value)
