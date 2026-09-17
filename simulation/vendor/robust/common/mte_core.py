"""Numerical implementation of the latest Section IV.

Algorithm 1 (MTE-Sub): Eqs. (49)-(75), normalized execution cost.
Algorithm 2 (MTE-Top): Eqs. (76)-(93), two-task expectation plus LP-guided Beam Search.

The LP in Eq. (86) has two resource constraints and box constraints.  Its dual is
2-dimensional.  We solve it exactly by enumerating all vertices of the dual
piecewise-linear objective.  This keeps the per-state LP cost O(N^2), matching
Section IV-D, without calling a generic LP solver at every Beam-Search state.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit
from numpy.polynomial.hermite import hermgauss
from statistics import NormalDist


def prepare_arrays(mission):
    cfg=mission.cfg; n=cfg.n_tasks+1; m=cfg.n_mec
    w=np.zeros(n); deadline=np.zeros(n); mu=np.zeros(n); sigma=np.zeros(n)
    aa=np.full((n,m),1e100); bb=np.zeros((n,m)); dd=np.full((n,m),1e100); ee=np.zeros((n,m))
    for task in mission.tasks:
        i=task.idx; w[i]=task.weight; deadline[i]=task.deadline_s; mu[i]=task.mu; sigma[i]=task.sigma
        for j,(ru,rd,fm) in enumerate(zip(task.ul_rates,task.dl_rates,cfg.mec_cpu_ghz)):
            if ru<cfg.ul_min_mbps or rd<cfg.dl_min_mbps:continue
            tu=task.data_mbit/ru; td=task.output_ratio*task.data_mbit/rd
            aa[i,j]=tu+td; bb[i,j]=1.0/fm
            ph=cfg.hover_power_w/1000.0
            dd[i,j]=cfg.tx_power_w*tu/1000.0+cfg.rx_power_w*td/1000.0+ph*(tu+td)
            ee[i,j]=ph/fm
    pars=np.array([cfg.t_max,cfg.e_max_kj,cfg.local_f_min_ghz,cfg.local_f_max_ghz,
                   cfg.kappa_kj_per_gcycle_ghz2,cfg.hover_power_w/1000.0,
                   cfg.sub_pressure_q,cfg.sub_bottleneck_beta],dtype=np.float64)
    return w,deadline,mu,sigma,pars,aa,bb,dd,ee


def gh_workload_points(mu,sigma,order):
    x,w=hermgauss(order); probs=w/math.sqrt(math.pi)
    pts=np.zeros((len(mu),order),dtype=np.float64)
    for i in range(1,len(mu)):
        pts[i]=np.exp(mu[i]+math.sqrt(2.0)*sigma[i]*x)
    return pts,probs.astype(np.float64)




def quantile_workload_points(mu, sigma, order):
    """Equal-probability midpoint quadrature for lognormal workloads.

    Unlike raw Monte Carlo samples, every marginal uses the same balanced
    probability strata, so lower and upper tails are represented deterministically.
    """
    q=max(1,int(order))
    nd=NormalDist()
    z=np.asarray([nd.inv_cdf((g+0.5)/q) for g in range(q)],dtype=np.float64)
    pts=np.zeros((len(mu),q),dtype=np.float64)
    for i in range(1,len(mu)):
        pts[i]=np.exp(mu[i]+sigma[i]*z)
    return pts,np.ones(q,dtype=np.float64)/q

def mean_workload_points(mu,sigma):
    pts=np.zeros((len(mu),1),dtype=np.float64)
    pts[:,0]=np.exp(mu+0.5*sigma*sigma); pts[0,0]=0.0
    return pts,np.ones(1,dtype=np.float64)


def saa_workload_points(mu,sigma,samples,seed,epoch):
    rng=np.random.default_rng(np.random.SeedSequence([91919,int(seed),int(epoch)]))
    z=rng.standard_normal((len(mu),samples)); z[0]=0.0
    pts=np.exp(mu[:,None]+sigma[:,None]*z); pts[0]=0.0
    return pts,np.ones(samples,dtype=np.float64)/samples


@njit(cache=True)
def execute_virtual(c,dt,de,pars,aa,bb,dd,ee,force_mode):
    """Pressure-aware MTE-Sub for one virtual workload.

    pars[6]=q and pars[7]=beta. q=0,beta=0 exactly recover the legacy
    equal-weight normalized cost. The local objective is convex and piecewise
    smooth, so all boundaries, regional stationary points, and the uT=uE kink
    are evaluated deterministically.
    """
    if c<=0.0 or dt<=0.0 or de<=0.0:
        return False,0.0,0.0,-1,0.0,1e100
    fmin=pars[2]; fmax=pars[3]; k=pars[4]; ph=pars[5]
    q=pars[6] if len(pars)>6 else 0.0
    beta=pars[7] if len(pars)>7 else 0.0
    pt=(pars[0]/max(dt,1e-12))**q
    pe=(pars[1]/max(de,1e-12))**q
    dp=pt+pe
    wt=2.0*pt/dp if dp>0.0 else 1.0
    we=2.0*pe/dp if dp>0.0 else 1.0
    best=1e100; bt=0.0; be=0.0; bm=-1; bf=0.0
    if force_mode!=2:
        lo=max(fmin,c/dt); hi=fmax
        if lo<=hi+1e-12:
            if lo>hi:lo=hi
            fm=(ph/(2.0*k))**(1.0/3.0)
            if fm<lo:fm=lo
            if fm>hi:fm=hi
            em=k*c*fm*fm+ph*c/fm
            if em<=de+1e-9:
                elo=k*c*lo*lo+ph*c/lo
                if elo>de:
                    a=lo; b=fm
                    for _ in range(55):
                        x=0.5*(a+b); ex=k*c*x*x+ph*c/x
                        if ex>de:a=x
                        else:b=x
                    lo=b
                ehi=k*c*hi*hi+ph*c/hi
                if ehi>de:
                    a=fm; b=hi
                    for _ in range(55):
                        x=0.5*(a+b); ex=k*c*x*x+ph*c/x
                        if ex>de:b=x
                        else:a=x
                    hi=a
                cand=np.empty(7,dtype=np.float64)
                cand[0]=lo; cand[1]=hi; cand[2]=fm
                a_t=(wt+beta)/dt + we*ph/de
                b_t=we*k/de
                cand[3]=(a_t/(2.0*b_t))**(1.0/3.0) if b_t>1e-30 else lo
                a_e=wt/dt + (we+beta)*ph/de
                b_e=(we+beta)*k/de
                cand[4]=(a_e/(2.0*b_e))**(1.0/3.0) if b_e>1e-30 else lo
                rhs=de/dt-ph
                cand[5]=(rhs/k)**(1.0/3.0) if rhs>0.0 else lo
                a0=wt/dt+we*ph/de
                b0=we*k/de
                cand[6]=(a0/(2.0*b0))**(1.0/3.0) if b0>1e-30 else lo
                for z in range(len(cand)):
                    f=cand[z]
                    if f<lo:f=lo
                    if f>hi:f=hi
                    t=c/f; en=k*c*f*f+ph*t
                    if t<=dt+1e-8 and en<=de+1e-8:
                        ut=t/dt; ue=en/de
                        cost=wt*ut+we*ue+beta*(ut if ut>=ue else ue)
                        if cost<best-1e-12:
                            best=cost; bt=t; be=en; bm=0; bf=f
    if force_mode!=1:
        for m in range(len(aa)):
            if aa[m]>=1e90:continue
            t=aa[m]+bb[m]*c; en=dd[m]+ee[m]*c
            if t<=dt+1e-9 and en<=de+1e-9:
                ut=t/dt; ue=en/de
                cost=wt*ut+we*ue+beta*(ut if ut>=ue else ue)
                if cost<best-1e-12:
                    best=cost; bt=t; be=en; bm=m+1; bf=0.0
    return bm>=0,bt,be,bm,bf,best


@njit(cache=True)
def visit_ok(node,t,e,i,ft,fe,deadline,pars):
    at=t+ft[node,i]; ae=e+fe[node,i]
    return (at+ft[i,0]<=pars[0]+1e-9 and ae+fe[i,0]<=pars[1]+1e-9 and deadline[i]-at>0.0)


@njit(cache=True)
def resource_limits(node,t,e,i,ft,fe,deadline,pars):
    at=t+ft[node,i]; ae=e+fe[node,i]
    dt=min(deadline[i]-at,pars[0]-at-ft[i,0]); de=pars[1]-ae-fe[i,0]
    return at,ae,dt,de


@njit(cache=True)
def expected_task_stats(node,t,e,i,points,probs,ft,fe,deadline,pars,aa,bb,dd,ee,force_mode):
    """p_i, E[T_i], E[E_i] at planning state y(P); execution terms only."""
    at,ae,dt,de=resource_limits(node,t,e,i,ft,fe,deadline,pars)
    if dt<=0.0 or de<=0.0:return 0.0,0.0,0.0
    p=0.0; et=0.0; en=0.0
    for g in range(len(probs)):
        ok,xt,xe,mode,f,cost=execute_virtual(points[i,g],dt,de,pars,aa[i],bb[i],dd[i],ee[i],force_mode)
        if ok:
            pr=probs[g]; p+=pr; et+=pr*xt; en+=pr*xe
    return p,et,en


@njit(cache=True)
def dual_value(y1,y2,cap_t,cap_e,vals,acost,bcost,nitems):
    out=cap_t*y1+cap_e*y2
    for r in range(nitems):
        z=vals[r]-acost[r]*y1-bcost[r]*y2
        if z>0.0:out+=z
    return out


@njit(cache=True)
def lp_two_resource_exact(cap_t,cap_e,vals,acost,bcost,nitems):
    """Exact value of Eq. (86) through its two-variable dual, O(N^2)."""
    if cap_t<=0.0 or cap_e<=0.0 or nitems==0:return 0.0
    best=dual_value(0.0,0.0,cap_t,cap_e,vals,acost,bcost,nitems)
    for i in range(nitems):
        if vals[i]<=0.0:continue
        if acost[i]>1e-12:
            y1=vals[i]/acost[i]; q=dual_value(y1,0.0,cap_t,cap_e,vals,acost,bcost,nitems)
            if q<best:best=q
        if bcost[i]>1e-12:
            y2=vals[i]/bcost[i]; q=dual_value(0.0,y2,cap_t,cap_e,vals,acost,bcost,nitems)
            if q<best:best=q
    for i in range(nitems):
        for j in range(i+1,nitems):
            det=acost[i]*bcost[j]-acost[j]*bcost[i]
            if abs(det)<1e-12:continue
            y1=(vals[i]*bcost[j]-vals[j]*bcost[i])/det
            y2=(acost[i]*vals[j]-acost[j]*vals[i])/det
            if y1>=-1e-12 and y2>=-1e-12:
                if y1<0:y1=0.0
                if y2<0:y2=0.0
                q=dual_value(y1,y2,cap_t,cap_e,vals,acost,bcost,nitems)
                if q<best:best=q
    return max(0.0,best)


@njit(cache=True)
def remaining_potential(node,t,e,mask,points,probs,ft,fe,weights,deadline,pars,aa,bb,dd,ee,force_mode,use_lp):
    if not use_lp:return 0.0
    n=len(weights)-1
    vals=np.zeros(n); acost=np.zeros(n); bcost=np.zeros(n); cnt=0
    for k in range(1,n+1):
        bit=np.uint64(1)<<np.uint64(k-1)
        if (mask & bit)==0:continue
        p,et,en=expected_task_stats(node,t,e,k,points,probs,ft,fe,deadline,pars,aa,bb,dd,ee,force_mode)
        v=weights[k]*p
        if v<=1e-14:continue
        mt=ft[node,k]; me=fe[node,k]
        for u in range(1,n+1):
            if u==k:continue
            bu=np.uint64(1)<<np.uint64(u-1)
            if (mask & bu)!=0:
                if ft[u,k]<mt:mt=ft[u,k]
                if fe[u,k]<me:me=fe[u,k]
        vals[cnt]=v; acost[cnt]=mt+et; bcost[cnt]=me+en; cnt+=1
    return lp_two_resource_exact(pars[0]-t,pars[1]-e,vals,acost,bcost,cnt)


@njit(cache=True)
def beam_future(node0,t0,e0,mask0,jshort,points,probs,ft,fe,weights,deadline,pars,aa,bb,dd,ee,
                force_mode,beam_width,use_lp):
    if mask0==0:return jshort,0
    n=len(weights)-1; B=max(1,beam_width)
    bnode=np.zeros(B,np.int64); bt=np.zeros(B); be=np.zeros(B); bmask=np.zeros(B,np.uint64); bj=np.zeros(B)
    bnode[0]=node0; bt[0]=t0; be[0]=e0; bmask[0]=mask0; bj[0]=jshort; bcount=1
    best_terminal=jshort; expanded=0
    for depth in range(n):
        cap=B*n
        cnode=np.zeros(cap,np.int64); ct=np.zeros(cap); ce=np.zeros(cap); cmask=np.zeros(cap,np.uint64)
        cj=np.zeros(cap); cs=np.full(cap,-1e100); ccount=0
        any_extension=False
        for h in range(bcount):
            extended_this=False
            for k in range(1,n+1):
                bit=np.uint64(1)<<np.uint64(k-1)
                if (bmask[h]&bit)==0:continue
                if not visit_ok(bnode[h],bt[h],be[h],k,ft,fe,deadline,pars):continue
                p,et,en=expected_task_stats(bnode[h],bt[h],be[h],k,points,probs,ft,fe,deadline,pars,aa,bb,dd,ee,force_mode)
                # A task with zero expected completion can still be reached, but it cannot improve J.
                # Skipping it only consumes flight resources, so it is never useful in the future-value search.
                if p<=1e-14:continue
                nt=bt[h]+ft[bnode[h],k]+et; ne=be[h]+fe[bnode[h],k]+en
                nm=bmask[h] & (~bit); nj=bj[h]+weights[k]*p
                pot=remaining_potential(k,nt,ne,nm,points,probs,ft,fe,weights,deadline,pars,aa,bb,dd,ee,force_mode,use_lp)
                cnode[ccount]=k; ct[ccount]=nt; ce[ccount]=ne; cmask[ccount]=nm; cj[ccount]=nj; cs[ccount]=nj+pot
                ccount+=1; expanded+=1; extended_this=True; any_extension=True
            if not extended_this and bj[h]>best_terminal:best_terminal=bj[h]
        if not any_extension or ccount==0:break
        # Keep B largest pruning scores, with accumulated J and smaller node as deterministic ties.
        for keep in range(min(B,ccount)):
            best_idx=-1; best_score=-1e100; best_j=-1e100; best_node=10**9
            for q in range(ccount):
                if cs[q]<=-1e90:continue
                if cs[q]>best_score+1e-12 or (abs(cs[q]-best_score)<=1e-12 and (cj[q]>best_j+1e-12 or (abs(cj[q]-best_j)<=1e-12 and cnode[q]<best_node))):
                    best_idx=q; best_score=cs[q]; best_j=cj[q]; best_node=cnode[q]
            if best_idx<0:break
            bnode[keep]=cnode[best_idx]; bt[keep]=ct[best_idx]; be[keep]=ce[best_idx]; bmask[keep]=cmask[best_idx]; bj[keep]=cj[best_idx]
            cs[best_idx]=-1e100
            if bj[keep]>best_terminal:best_terminal=bj[keep]
        bcount=min(B,ccount)
    return best_terminal,expanded


@njit(cache=True)
def top_choose(node,t,e,visited_mask,points,probs,ft,fe,weights,deadline,pars,aa,bb,dd,ee,
               force_mode,beam_width,with_future,use_lp):
    """Algorithm 2. Returns selected i, J_i, J_short_i, J_future_i, expanded states."""
    n=len(weights)-1
    allmask=(np.uint64(1)<<np.uint64(n))-np.uint64(1)
    unvisited=allmask & (~visited_mask)
    best_i=0; best_val=0.0; best_short=0.0; best_future=0.0; expanded_total=0
    for i in range(1,n+1):
        biti=np.uint64(1)<<np.uint64(i-1)
        if (unvisited&biti)==0:continue
        if not visit_ok(node,t,e,i,ft,fe,deadline,pars):continue
        at_i,ae_i,dt_i,de_i=resource_limits(node,t,e,i,ft,fe,deadline,pars)
        G=len(probs)
        ti=np.zeros(G); ei=np.zeros(G); xi=np.zeros(G); t1=np.zeros(G); e1=np.zeros(G)
        for g in range(G):
            ok,xt,xe,mode,f,cost=execute_virtual(points[i,g],dt_i,de_i,pars,aa[i],bb[i],dd[i],ee[i],force_mode)
            if ok:xi[g]=1.0; ti[g]=xt; ei[g]=xe; t1[g]=at_i+xt; e1[g]=ae_i+xe
            else:t1[g]=at_i; e1[g]=ae_i
        rem_after_i=unvisited & (~biti)
        # One-task edge case.
        if rem_after_i==0:
            js=0.0
            for g in range(G):js+=probs[g]*weights[i]*xi[g]
            val=js
            if val>best_val+1e-12 or (abs(val-best_val)<=1e-12 and (best_i==0 or i<best_i)):
                best_i=i; best_val=val; best_short=js; best_future=0.0
            continue
        best_pair=-1e100; pair_short=0.0; pair_future=0.0
        for j in range(1,n+1):
            bitj=np.uint64(1)<<np.uint64(j-1)
            if j==i or (rem_after_i&bitj)==0:continue
            js=0.0; ex_t=0.0; ex_e=0.0
            for g in range(G):
                p2,t2,e2=expected_task_stats(i,t1[g],e1[g],j,points,probs,ft,fe,deadline,pars,aa,bb,dd,ee,force_mode)
                js+=probs[g]*(weights[i]*xi[g]+weights[j]*p2)
                ex_t+=probs[g]*(ti[g]+t2); ex_e+=probs[g]*(ei[g]+e2)
            plan_t=t+ft[node,i]+ft[i,j]+ex_t; plan_e=e+fe[node,i]+fe[i,j]+ex_e
            rem=rem_after_i & (~bitj)
            if with_future:
                total,expanded=beam_future(j,plan_t,plan_e,rem,js,points,probs,ft,fe,weights,deadline,pars,aa,bb,dd,ee,force_mode,beam_width,use_lp)
                expanded_total+=expanded
            else:total=js
            if total>best_pair+1e-12 or (abs(total-best_pair)<=1e-12 and j<i):
                best_pair=total; pair_short=js; pair_future=total-js
        if best_pair<-1e90:
            # No second task: use expected first-task value.
            best_pair=0.0
            for g in range(G):best_pair+=probs[g]*weights[i]*xi[g]
            pair_short=best_pair; pair_future=0.0
        if best_pair>best_val+1e-12 or (abs(best_pair-best_val)<=1e-12 and (best_i==0 or i<best_i)):
            best_i=i; best_val=best_pair; best_short=pair_short; best_future=pair_future
    return best_i,best_val,best_short,best_future,expanded_total


@njit(cache=True)
def myopic_choose(node,t,e,visited_mask,points,probs,ft,fe,weights,deadline,pars,aa,bb,dd,ee,force_mode):
    n=len(weights)-1; best_i=0; best=-1.0
    for i in range(1,n+1):
        bit=np.uint64(1)<<np.uint64(i-1)
        if (visited_mask&bit)!=0:continue
        if not visit_ok(node,t,e,i,ft,fe,deadline,pars):continue
        p,et,en=expected_task_stats(node,t,e,i,points,probs,ft,fe,deadline,pars,aa,bb,dd,ee,force_mode)
        q=weights[i]*p
        if q>best+1e-12 or (abs(q-best)<=1e-12 and (best_i==0 or ft[node,i]<ft[node,best_i])):
            best=q;best_i=i
    return best_i

@njit(cache=True)
def max_feasible_workload(dt,de,pars,aa,bb,dd,ee,force_mode):
    """Largest workload that admits at least one feasible execution action.

    Execution feasibility is monotone in workload.  A bounded bisection therefore
    gives the exact threshold needed for the lognormal completion probability.
    """
    if dt<=0.0 or de<=0.0:
        return 0.0
    hi=dt*pars[3]
    if force_mode!=1:
        for m in range(len(aa)):
            if aa[m]>=1e90 or bb[m]<=0.0:
                continue
            ct=(dt-aa[m])/bb[m]
            if ct>hi:
                hi=ct
    if hi<=0.0:
        return 0.0
    lo=0.0
    ok,xt,xe,mode,f,cost=execute_virtual(hi,dt,de,pars,aa,bb,dd,ee,force_mode)
    if ok:
        return hi
    for _ in range(64):
        mid=0.5*(lo+hi)
        ok,xt,xe,mode,f,cost=execute_virtual(mid,dt,de,pars,aa,bb,dd,ee,force_mode)
        if ok:
            lo=mid
        else:
            hi=mid
    return lo


@njit(cache=True)
def lognormal_completion_probability(mu,sigma,cmax):
    if cmax<=0.0:
        return 0.0
    if sigma<=1e-12:
        return 1.0 if math.exp(mu)<=cmax else 0.0
    z=(math.log(cmax)-mu)/(sigma*math.sqrt(2.0))
    p=0.5*(1.0+math.erf(z))
    if p<0.0:return 0.0
    if p>1.0:return 1.0
    return p


@njit(cache=True)
def calibrated_task_stats(node,t,e,i,points,probs,mu,sigma,ft,fe,deadline,pars,aa,bb,dd,ee,force_mode):
    """Exact completion mass plus quadrature-calibrated resource moments.

    The completion probability is evaluated from the exact monotone workload
    threshold.  Feasible quadrature moments are then rescaled to this exact mass,
    removing the coarse all-or-nothing probability error caused by applying a
    low-order quadrature rule directly to the feasibility indicator.
    """
    at,ae,dt,de=resource_limits(node,t,e,i,ft,fe,deadline,pars)
    if dt<=0.0 or de<=0.0:
        return 0.0,0.0,0.0
    cmax=max_feasible_workload(dt,de,pars,aa[i],bb[i],dd[i],ee[i],force_mode)
    p=lognormal_completion_probability(mu[i],sigma[i],cmax)
    if p<=1e-15:
        return 0.0,0.0,0.0
    pq=0.0;etq=0.0;enq=0.0
    for g in range(len(probs)):
        ok,xt,xe,mode,f,cost=execute_virtual(points[i,g],dt,de,pars,aa[i],bb[i],dd[i],ee[i],force_mode)
        if ok:
            pr=probs[g];pq+=pr;etq+=pr*xt;enq+=pr*xe
    if pq>1e-14:
        scale=p/pq
        return p,etq*scale,enq*scale
    # The feasible tail lies below the smallest quadrature point.  Use a
    # representative workload inside the exact feasible interval.
    cref=0.70*cmax
    mean_c=math.exp(mu[i]+0.5*sigma[i]*sigma[i])
    if mean_c<cref:cref=mean_c
    if cref<=0.0:return p,0.0,0.0
    ok,xt,xe,mode,f,cost=execute_virtual(cref,dt,de,pars,aa[i],bb[i],dd[i],ee[i],force_mode)
    if ok:
        return p,p*xt,p*xe
    return p,0.0,0.0

