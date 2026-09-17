"""Bounded subset DP with a public-distribution estimate of omitted tasks.

The tail is an approximate mean-resource greedy rollout, not an executable
lower bound or an optimality guarantee. Exact physical execution is unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,math
import numpy as np
from numba import njit,types
from numba.typed import Dict
from scipy.special import ndtr,ndtri
from planner import ConsistentACARPlanner,ExecutionAction
from particle_beam import ParticlePlanner
from batch_adapter import iteration_v2,robust_core
from root_calibrated_core import _maximum_capacity,_lognormal_cdf,_truncated_lognormal_first_moment,_lognormal_ppf

@njit(cache=True)
def greedy_tail(node,t,e,mask,depth,power,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee):
    value=0.;first=0
    for step in range(depth):
        best=0;best_score=0.;best_t=0.;best_e=0.;best_p=0.
        for task in range(1,len(weights)):
            bit=np.uint64(1)<<np.uint64(task-1)
            if mask & bit==0 or not iteration_v2.visit_ok_v2(node,t,e,task,ft,fe,deadline,pars):continue
            at,ae,dt,de=iteration_v2.resource_limits_v2(node,t,e,task,ft,fe,deadline,pars)
            cap=_maximum_capacity(task,dt,de,pars,aa,bb,dd,ee)
            p=_lognormal_cdf(mu[task],sigma[task],cap)
            if p<=1e-10:continue
            c=min(cap,_truncated_lognormal_first_moment(mu[task],sigma[task],cap)/p)
            ok,st,se,mode,freq,cost=robust_core.execute_virtual(c,dt,de,pars,aa[task],bb[task],dd[task],ee[task],0)
            if not ok:continue
            et,ee_used=at+p*st,ae+p*se
            occupation=(et-t)/max(pars[0]-t,1e-12)+(ee_used-e)/max(pars[1]-e,1e-12)
            score=weights[task]*p/max(occupation,1e-12)**power
            if score>best_score+1e-12:
                best,best_score,best_t,best_e,best_p=task,score,et,ee_used,p
        if best==0:break
        if first==0:first=best
        value+=weights[best]*best_p
        node,t,e=best,best_t,best_e
        mask &= ~(np.uint64(1)<<np.uint64(best-1))
    return value,first

@njit(cache=True)
def grid(x,step,rounding):
    if rounding==0:return int(math.ceil(x/step-1e-12))
    return int(math.floor(x/step+.5))

@njit(cache=True)
def enhanced_value(node,it,ie,mask,tasks,tailmask,time_step,energy_step,nt,ne,cache,tail_cache,
                   moments,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,tail_weight,tail_depth,power,rounding,integration):
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
        skip,_=enhanced_value(task,sit,sie,rem,tasks,tailmask,time_step,energy_step,nt,ne,cache,tail_cache,
            moments,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,tail_weight,tail_depth,power,rounding,integration)
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
                future,_=enhanced_value(task,cit,cie,rem,tasks,tailmask,time_step,energy_step,nt,ne,cache,tail_cache,
                    moments,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,tail_weight,tail_depth,power,rounding,integration)
                v=max(v,weights[task]+future)
            value+=pr*v
        if value>best+1e-12 or (abs(value-best)<=1e-12 and best_task>0 and task<best_task):
            best,best_task=value,task
    cache[key]=best,best_task
    return best,best_task

@dataclass(frozen=True)
class EnhancedSpec:
    shortlist:int=6
    time_step:float=25.
    energy_step:float=7.5
    quadrature:int=3
    tail_weight:float=.7
    tail_depth:int=6
    cost_power:float=.5
    rounding:int=0
    mode:str='both'
    arrival_rule:str='original'
    integration:int=0

class EnhancedContinuation:
    def __init__(self,base,spec):
        self.base,self.spec=base,spec
        self.points,self.probabilities=base.points,base.probabilities
        self.mu,self.sigma=base.mu,base.sigma
        self.conditional_uniform=base.conditional_uniform
        z=ndtri(np.arange(1,spec.quadrature)/spec.quadrature)
        self.moments=np.exp(self.mu[:,None]+.5*self.sigma[:,None]**2)*ndtr(z-self.sigma[:,None])
        self.calls=self.cache_hits=self.expansions=0;self.cache={}
        self.signature=hashlib.sha256(json.dumps(spec.__dict__,sort_keys=True).encode()).hexdigest()[:16]
    def plan(self,state):
        key=(state.node,state.time_s,state.energy_kj,state.remaining_mask)
        if key in self.cache:self.cache_hits+=1;return self.cache[key]
        b,s=self.base,self.spec
        ft,fe=b.bundle.iteration.flight_time,b.bundle.iteration.flight_energy_kj
        scores=[]
        for task in range(1,len(b.weights)):
            if not state.remaining_mask & (1<<(task-1)):continue
            if not iteration_v2.visit_ok_v2(state.node,state.time_s,state.energy_kj,task,ft,fe,b.deadline,b.pars):continue
            p,t,e=robust_core.calibrated_task_stats(state.node,state.time_s,state.energy_kj,task,
                b.points,b.probabilities,b.mu,b.sigma,ft,fe,b.deadline,b.pars,b.aa,b.bb,b.dd,b.ee,0)
            cost=(ft[state.node,task]+t)/max(b.pars[0]-state.time_s,1e-12)+(fe[state.node,task]+e)/max(b.pars[1]-state.energy_kj,1e-12)
            scores.append((float(b.weights[task]*p)/max(cost,1e-12)**s.cost_power,-task))
        tasks=np.array([-x[1] for x in sorted(scores,reverse=True)[:s.shortlist]],np.int64)
        # A rounded planning state must never authorize a physically unsafe
        # first visit. Candidates and the outside tail use the same exact
        # current-state visit filter before any resource rounding.
        tailmask=sum(1<<(-item[1]-1) for item in scores)
        for task in tasks:tailmask &= ~(1<<(int(task)-1))
        cache=Dict.empty(types.int64,types.Tuple((types.float64,types.int64)))
        tc=Dict.empty(types.int64,types.Tuple((types.float64,types.int64)))
        nt,ne=math.ceil(b.pars[0]/s.time_step)+3,math.ceil(b.pars[1]/s.energy_step)+3
        v,task=enhanced_value(state.node,grid(state.time_s,s.time_step,s.rounding),grid(state.energy_kj,s.energy_step,s.rounding),
            (1<<len(tasks))-1,tasks,np.uint64(tailmask),s.time_step,s.energy_step,nt,ne,cache,tc,
            self.moments,b.mu,b.sigma,ft,fe,b.weights,b.deadline,b.pars,b.aa,b.bb,b.dd,b.ee,
            s.tail_weight,s.tail_depth,s.cost_power,s.rounding,s.integration)
        self.expansions+=len(cache);self.cache[key]=int(task),float(v)
        return self.cache[key]
    def value(self,state,immediate):self.calls+=1;return immediate+self.plan(state)[1]

class EnhancedPlanner(ParticlePlanner):
    enhanced_spec=EnhancedSpec()
    def __init__(self,bundle,policy):
        ConsistentACARPlanner.__init__(self,bundle,policy)
        self.enhanced_spec=type(self).enhanced_spec
        s=self.enhanced_spec
        if s.mode=='local':
            self.aa=self.aa.copy();self.aa[:]=1e100
            self.continuation.aa=self.aa
        elif s.mode=='mec':
            self.pars=self.pars.copy();self.pars[2]=self.pars[3]=1e-12
            # Keep the continuation's pressure parameters (q, beta). The
            # actual-action parameter array has only the six physical fields.
            self.continuation.pars=self.continuation.pars.copy()
            self.continuation.pars[2:4]=self.pars[2:4]
        elif s.mode!='both':raise ValueError(s.mode)
        self.continuation=EnhancedContinuation(self.continuation,s)
    def _completion_actions(self,task,workload,available_t,available_e,price_t,price_e):
        actions=ConsistentACARPlanner._completion_actions(self,task,workload,available_t,available_e,price_t,price_e)
        mode=self.enhanced_spec.mode
        return tuple(a for a in actions if mode=='both' or a.mode==mode)
    def select_task(self,state):
        task,detail=ParticlePlanner.select_task(self,state)
        detail['selection_mode']='bounded_dp_with_tail'
        return task,detail
