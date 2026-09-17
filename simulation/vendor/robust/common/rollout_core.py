"""Numba kernels for the scenario-calibrated recourse rollout planner."""
from __future__ import annotations
import math
import numpy as np
from numba import njit

from .mte_core import calibrated_task_stats, execute_virtual, visit_ok, resource_limits


@njit(cache=True)
def _top3_cluster(i,mask,ft,weights):
    n=len(weights)-1
    a=0.0;b=0.0;c=0.0
    for j in range(1,n+1):
        bit=np.uint64(1)<<np.uint64(j-1)
        if j==i or (mask&bit)==0: continue
        v=weights[j]/(1.0+ft[i,j]/240.0)
        if v>a:
            c=b;b=a;a=v
        elif v>b:
            c=b;b=v
        elif v>c:
            c=v
    return (a+b+c)/45.0


@njit(cache=True)
def _fast_score(node,t,e,i,mask,points,probs,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,
                cost_floor,cost_power,cluster_weight):
    bit=np.uint64(1)<<np.uint64(i-1)
    if (mask&bit)==0: return -1e100
    if not visit_ok(node,t,e,i,ft,fe,deadline,pars): return -1e100
    p,et,en=calibrated_task_stats(node,t,e,i,points,probs,mu,sigma,ft,fe,deadline,pars,aa,bb,dd,ee,0)
    if p<=1e-12: return -1e100
    rem_t=pars[0]-t-ft[node,0]
    rem_e=pars[1]-e-fe[node,0]
    if rem_t<1e-9: rem_t=1e-9
    if rem_e<1e-9: rem_e=1e-9
    inc_ft=ft[node,i]+ft[i,0]-ft[node,0]
    inc_fe=fe[node,i]+fe[i,0]-fe[node,0]
    if inc_ft<0.0: inc_ft=0.0
    if inc_fe<0.0: inc_fe=0.0
    cost=inc_ft/rem_t+inc_fe/rem_e+et/rem_t+en/rem_e
    denom=(cost_floor+cost)**cost_power
    score=weights[i]*p/denom
    if cluster_weight!=0.0:
        score+=cluster_weight*weights[i]*_top3_cluster(i,mask,ft,weights)
    return score


@njit(cache=True)
def _fast_choose(node,t,e,mask,points,probs,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,
                 cost_floor,cost_power,cluster_weight):
    n=len(weights)-1;best_i=0;best=-1e100
    for i in range(1,n+1):
        s=_fast_score(node,t,e,i,mask,points,probs,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,
                      cost_floor,cost_power,cluster_weight)
        if s>best+1e-12 or (abs(s-best)<=1e-12 and s>-1e90 and (best_i==0 or i<best_i)):
            best=s;best_i=i
    return best_i


@njit(cache=True)
def _apply_scenario_task(node,t,e,mask,i,c,ft,fe,weights,deadline,pars,aa,bb,dd,ee):
    at,ae,dt,de=resource_limits(node,t,e,i,ft,fe,deadline,pars)
    bit=np.uint64(1)<<np.uint64(i-1)
    newmask=mask&(~bit)
    if dt<=0.0 or de<=0.0:
        return i,at,ae,newmask,0.0
    ok,xt,xe,mode,f,cost=execute_virtual(c,dt,de,pars,aa[i],bb[i],dd[i],ee[i],0)
    if ok:
        return i,at+xt,ae+xe,newmask,weights[i]
    return i,at,ae,newmask,0.0


@njit(cache=True)
def _simulate_candidate(node,t,e,mask,candidate,scenario,points,probs,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,
                        cost_floor,cost_power,cluster_weight,max_steps):
    sn,st,se,sm,reward=_apply_scenario_task(node,t,e,mask,candidate,scenario[candidate],ft,fe,weights,deadline,pars,aa,bb,dd,ee)
    total=reward
    for _ in range(max_steps):
        if sm==0: break
        i=_fast_choose(sn,st,se,sm,points,probs,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,
                       cost_floor,cost_power,cluster_weight)
        if i==0: break
        sn,st,se,sm,r=_apply_scenario_task(sn,st,se,sm,i,scenario[i],ft,fe,weights,deadline,pars,aa,bb,dd,ee)
        total+=r
    return total


@njit(cache=True)
def rollout_choose(node,t,e,visited_mask,scenarios,points,probs,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,
                   shortlist,cost_floor,cost_power,cluster_weight,max_steps,risk_weight,lower_tail_fraction):
    """Return the next task from a common-random-number scenario rollout."""
    n=len(weights)-1
    allmask=(np.uint64(1)<<np.uint64(n))-np.uint64(1)
    mask=allmask&(~visited_mask)
    if mask==0:return 0,0.0,0
    K=max(1,min(shortlist,n))
    cand=np.zeros(K,dtype=np.int64);cscore=np.full(K,-1e100)
    # Repeated maximum extraction gives deterministic top-K fast-policy actions.
    used=np.zeros(n+1,dtype=np.uint8)
    count=0
    for k in range(K):
        bi=0;bs=-1e100
        for i in range(1,n+1):
            if used[i]!=0:continue
            s=_fast_score(node,t,e,i,mask,points,probs,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,
                          cost_floor,cost_power,cluster_weight)
            if s>bs+1e-12 or (abs(s-bs)<=1e-12 and s>-1e90 and (bi==0 or i<bi)):
                bi=i;bs=s
        if bi==0 or bs<=-1e90:break
        cand[count]=bi;cscore[count]=bs;used[bi]=1;count+=1
    if count==0:return 0,0.0,0
    S=scenarios.shape[0]
    vals=np.empty(S,dtype=np.float64)
    best_i=0;best_q=-1e100;best_fast=-1e100
    for k in range(count):
        i=cand[k]
        for s in range(S):
            vals[s]=_simulate_candidate(node,t,e,mask,i,scenarios[s],points,probs,mu,sigma,ft,fe,weights,deadline,pars,aa,bb,dd,ee,
                                        cost_floor,cost_power,cluster_weight,max_steps)
        q=0.0
        for s in range(S):q+=vals[s]
        q/=S
        if risk_weight>0.0:
            vv=np.sort(vals.copy())
            ntail=max(1,int(math.ceil(lower_tail_fraction*S)))
            low=0.0
            for s in range(ntail):low+=vv[s]
            low/=ntail
            q=q-risk_weight*(q-low)
        fs=cscore[k]
        if q>best_q+1e-12 or (abs(q-best_q)<=1e-12 and (fs>best_fast+1e-12 or (abs(fs-best_fast)<=1e-12 and (best_i==0 or i<best_i)))):
            best_i=i;best_q=q;best_fast=fs
    return best_i,best_q,count
