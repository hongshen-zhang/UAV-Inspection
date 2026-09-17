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
    pars=np.array([
        cfg.t_max,
        cfg.e_max_kj,
        cfg.local_f_min_ghz,
        cfg.local_f_max_ghz,
        cfg.kappa_kj_per_gcycle_ghz2,
        cfg.hover_power_w/1000.0,
        cfg.sub_pressure_power,
        cfg.sub_bottleneck_weight,
        cfg.top_risk_z,
        cfg.top_recourse_weight,
        cfg.top_recourse_lp_discount,
        cfg.top_failure_penalty,
        cfg.top_risk_e_z,
        cfg.top_recourse_blend,
        cfg.top_recourse_uncertainty_gain,
        cfg.top_downside_weight,
        cfg.top_downside_alpha,
        1.0 if cfg.beam_endpoint_diversity else 0.0,
    ],dtype=np.float64)
    return w,deadline,mu,sigma,pars,aa,bb,dd,ee


def gh_workload_points(mu,sigma,order):
    x,w=hermgauss(order); probs=w/math.sqrt(math.pi)
    pts=np.zeros((len(mu),order),dtype=np.float64)
    for i in range(1,len(mu)):
        pts[i]=np.exp(mu[i]+math.sqrt(2.0)*sigma[i]*x)
    return pts,probs.astype(np.float64)


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
def aligned_execution_cost(t,e,dt,de,pars):
    """Resource-aligned MTE-Sub cost.

    The current rule t/dt + e/de uses equal resource weights.  The aligned
    rule increases the weight of the resource whose remaining fraction is
    smaller and adds a bottleneck term.  Setting the two new coefficients to
    zero exactly recovers the original cost.
    """
    ut=t/max(dt,1e-12); ue=e/max(de,1e-12)
    q=pars[6]; beta=pars[7]
    if q<=1e-14 and beta<=1e-14:
        return ut+ue
    pt=(pars[0]/max(dt,1e-12))**q if q>1e-14 else 1.0
    pe=(pars[1]/max(de,1e-12))**q if q>1e-14 else 1.0
    den=pt+pe
    wt=2.0*pt/den; we=2.0*pe/den
    return wt*ut+we*ue+beta*max(ut,ue)


@njit(cache=True)
def execute_virtual(c,dt,de,pars,aa,bb,dd,ee,force_mode):
    """Algorithm 1 for one revealed/virtual workload.

    force_mode: 0 auto, 1 local only, 2 MEC only.
    Returns feasible, time, energy, mode (0 local, 1..M MEC), f, normalized cost.
    """
    if c<=0.0 or dt<=0.0 or de<=0.0:
        return False,0.0,0.0,-1,0.0,1e100
    fmin=pars[2]; fmax=pars[3]; k=pars[4]; ph=pars[5]
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
                fsta=((de+ph*dt)/(2.0*k*dt))**(1.0/3.0)
                fmid=min(hi,max(lo,fsta))
                for f in (lo,fmid,hi):
                    t=c/f; e=k*c*f*f+ph*t
                    if t<=dt+1e-8 and e<=de+1e-8:
                        cost=aligned_execution_cost(t,e,dt,de,pars)
                        if cost<best-1e-12:
                            best=cost; bt=t; be=e; bm=0; bf=f
    if force_mode!=1:
        for m in range(len(aa)):
            if aa[m]>=1e90:continue
            t=aa[m]+bb[m]*c; e=dd[m]+ee[m]*c
            if t<=dt+1e-9 and e<=de+1e-9:
                cost=aligned_execution_cost(t,e,dt,de,pars)
                if cost<best-1e-12:
                    best=cost; bt=t; be=e; bm=m+1; bf=0.0
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
def expected_task_moments(node,t,e,i,points,probs,ft,fe,deadline,pars,aa,bb,dd,ee,force_mode):
    """Completion probability and first two execution-resource moments.

    Execution time and energy are zero in workload branches where the task is
    not executable.  This matches the online policy: the UAV has already paid
    the flight cost, but it does not spend execution resources after a failed
    feasibility check.
    """
    at,ae,dt,de=resource_limits(node,t,e,i,ft,fe,deadline,pars)
    if dt<=0.0 or de<=0.0:return 0.0,0.0,0.0,0.0,0.0
    p=0.0; et=0.0; en=0.0; et2=0.0; en2=0.0
    for g in range(len(probs)):
        ok,xt,xe,mode,f,cost=execute_virtual(points[i,g],dt,de,pars,aa[i],bb[i],dd[i],ee[i],force_mode)
        if ok:
            pr=probs[g]; p+=pr; et+=pr*xt; en+=pr*xe; et2+=pr*xt*xt; en2+=pr*xe*xe
    st=math.sqrt(max(0.0,et2-et*et)); se=math.sqrt(max(0.0,en2-en*en))
    return p,et,en,st,se


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
        p,et,en,st,se=expected_task_moments(node,t,e,k,points,probs,ft,fe,deadline,pars,aa,bb,dd,ee,force_mode)
        fail_penalty=pars[11]
        v=weights[k]*(p-fail_penalty*(1.0-p))
        if v<=1e-14:continue
        zt=pars[8]; ze=pars[12]
        et=et+zt*st; en=en+ze*se
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
                p,et,en,st,se=expected_task_moments(bnode[h],bt[h],be[h],k,points,probs,ft,fe,deadline,pars,aa,bb,dd,ee,force_mode)
                fail_penalty=pars[11]
                reward=weights[k]*(p-fail_penalty*(1.0-p))
                # A task with no positive risk-adjusted value is not useful in the future search.
                if reward<=1e-14:continue
                zt=pars[8]; ze=pars[12]
                nt=bt[h]+ft[bnode[h],k]+et+zt*st; ne=be[h]+fe[bnode[h],k]+en+ze*se
                nm=bmask[h] & (~bit); nj=bj[h]+reward
                pot=remaining_potential(k,nt,ne,nm,points,probs,ft,fe,weights,deadline,pars,aa,bb,dd,ee,force_mode,use_lp)
                cnode[ccount]=k; ct[ccount]=nt; ce[ccount]=ne; cmask[ccount]=nm; cj[ccount]=nj; cs[ccount]=nj+pot
                ccount+=1; expanded+=1; extended_this=True; any_extension=True
            if not extended_this and bj[h]>best_terminal:best_terminal=bj[h]
        if not any_extension or ccount==0:break
        # Keep B largest pruning scores.  When endpoint diversity is enabled,
        # the first pass keeps different terminal tasks so that the beam does
        # not collapse to several nearly identical local routes.
        selected_nodes=np.full(B,-1,np.int64)
        for keep in range(min(B,ccount)):
            best_idx=-1; best_score=-1e100; best_j=-1e100; best_node=10**9
            require_unique=pars[17]>0.5
            for pass_id in range(2):
                for q in range(ccount):
                    if cs[q]<=-1e90:continue
                    if require_unique and pass_id==0:
                        duplicate=False
                        for u in range(keep):
                            if selected_nodes[u]==cnode[q]:duplicate=True;break
                        if duplicate:continue
                    if cs[q]>best_score+1e-12 or (abs(cs[q]-best_score)<=1e-12 and (cj[q]>best_j+1e-12 or (abs(cj[q]-best_j)<=1e-12 and cnode[q]<best_node))):
                        best_idx=q; best_score=cs[q]; best_j=cj[q]; best_node=cnode[q]
                if best_idx>=0 or not require_unique:break
            if best_idx<0:break
            bnode[keep]=cnode[best_idx]; bt[keep]=ct[best_idx]; be[keep]=ce[best_idx]; bmask[keep]=cmask[best_idx]; bj[keep]=cj[best_idx]
            selected_nodes[keep]=cnode[best_idx]
            cs[best_idx]=-1e100
            if bj[keep]>best_terminal:best_terminal=bj[keep]
        bcount=min(B,ccount)
    return best_terminal,expanded


@njit(cache=True)
def lower_tail_value(vals,probs,alpha):
    n=len(vals)
    if n==0:return 0.0
    if alpha<=1e-12:return np.min(vals)
    order=np.argsort(vals)
    need=min(1.0,max(alpha,1e-12)); used=0.0; out=0.0
    for q in range(n):
        idx=order[q]; take=min(probs[idx],need-used)
        if take>0.0:
            out+=take*vals[idx]; used+=take
        if used>=need-1e-14:break
    if used<=1e-14:return vals[order[0]]
    return out/used


@njit(cache=True)
def recourse_option_bonus(i,t1,e1,xi,rem_after_i,points,probs,ft,fe,weights,deadline,pars,aa,bb,dd,ee,force_mode):
    """Value of state-conditioned second-task recourse after task i.

    After the workload of task i is revealed, the next task is selected again.
    The original fixed-pair evaluation does not fully capture this option.
    This term estimates E[max_j Q_j(y_g)]-max_j E[Q_j(y_g)] using the same
    workload quadrature and the LP remaining-potential estimate.
    """
    G=len(probs)
    if G<=1 or rem_after_i==0:return 0.0,0.0,0.0
    n=len(weights)-1; fail_penalty=pars[11]; zt=pars[8]; ze=pars[12]; discount=pars[10]
    branch_best=np.zeros(G)
    fixed_best=0.0
    for g in range(G):
        base=weights[i]*(xi[g]-fail_penalty*(1.0-xi[g]))
        branch_best[g]=base; fixed_best+=probs[g]*base
    for j in range(1,n+1):
        bitj=np.uint64(1)<<np.uint64(j-1)
        if (rem_after_i&bitj)==0:continue
        acc=0.0
        rem=rem_after_i & (~bitj)
        for g in range(G):
            base=weights[i]*(xi[g]-fail_penalty*(1.0-xi[g]))
            proxy=base
            if visit_ok(i,t1[g],e1[g],j,ft,fe,deadline,pars):
                p2,t2,e2,st2,se2=expected_task_moments(i,t1[g],e1[g],j,points,probs,ft,fe,deadline,pars,aa,bb,dd,ee,force_mode)
                reward2=weights[j]*(p2-fail_penalty*(1.0-p2))
                if reward2>0.0:
                    nt=t1[g]+ft[i,j]+t2+zt*st2
                    ne=e1[g]+fe[i,j]+e2+ze*se2
                    pot=remaining_potential(j,nt,ne,rem,points,probs,ft,fe,weights,deadline,pars,aa,bb,dd,ee,force_mode,True)
                    candidate=base+reward2+discount*pot
                    if candidate>proxy:proxy=candidate
            acc+=probs[g]*proxy
            if proxy>branch_best[g]:branch_best[g]=proxy
        if acc>fixed_best:fixed_best=acc
    recourse=0.0
    for g in range(G):recourse+=probs[g]*branch_best[g]
    downside=lower_tail_value(branch_best,probs,pars[16])
    return max(0.0,recourse-fixed_best),recourse,downside


@njit(cache=True)
def top_choose(node,t,e,visited_mask,points,probs,ft,fe,weights,deadline,pars,aa,bb,dd,ee,
               force_mode,beam_width,with_future,use_lp):
    """Algorithm 2. Returns selected i, J_i, J_short_i, J_future_i, expanded states."""
    n=len(weights)-1
    allmask=(np.uint64(1)<<np.uint64(n))-np.uint64(1)
    unvisited=allmask & (~visited_mask)
    best_i=0; best_val=0.0; best_short=0.0; best_future=0.0; expanded_total=0
    candidate_scores=np.full(n+1,-1e100)
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
            js=0.0; fail_penalty=pars[11]
            for g in range(G):js+=probs[g]*weights[i]*(xi[g]-fail_penalty*(1.0-xi[g]))
            val=js
            candidate_scores[i]=val
            if val>best_val+1e-12 or (abs(val-best_val)<=1e-12 and (best_i==0 or i<best_i)):
                best_i=i; best_val=val; best_short=js; best_future=0.0
            continue
        best_pair=-1e100; pair_short=0.0; pair_future=0.0
        for j in range(1,n+1):
            bitj=np.uint64(1)<<np.uint64(j-1)
            if j==i or (rem_after_i&bitj)==0:continue
            js=0.0; ex_t=0.0; ex_e=0.0; ex_t2=0.0; ex_e2=0.0
            fail_penalty=pars[11]; zt=pars[8]; ze=pars[12]
            for g in range(G):
                p2,t2,e2,st2,se2=expected_task_moments(i,t1[g],e1[g],j,points,probs,ft,fe,deadline,pars,aa,bb,dd,ee,force_mode)
                r1=weights[i]*(xi[g]-fail_penalty*(1.0-xi[g]))
                r2=weights[j]*(p2-fail_penalty*(1.0-p2))
                js+=probs[g]*(r1+r2)
                tg=ti[g]+t2; eg=ei[g]+e2
                ex_t+=probs[g]*tg; ex_e+=probs[g]*eg
                ex_t2+=probs[g]*(tg*tg+st2*st2); ex_e2+=probs[g]*(eg*eg+se2*se2)
            pair_st=math.sqrt(max(0.0,ex_t2-ex_t*ex_t)); pair_se=math.sqrt(max(0.0,ex_e2-ex_e*ex_e))
            plan_t=t+ft[node,i]+ft[i,j]+ex_t+zt*pair_st; plan_e=e+fe[node,i]+fe[i,j]+ex_e+ze*pair_se
            rem=rem_after_i & (~bitj)
            if with_future:
                total,expanded=beam_future(j,plan_t,plan_e,rem,js,points,probs,ft,fe,weights,deadline,pars,aa,bb,dd,ee,force_mode,beam_width,use_lp)
                expanded_total+=expanded
            else:total=js
            if total>best_pair+1e-12 or (abs(total-best_pair)<=1e-12 and j<i):
                best_pair=total; pair_short=js; pair_future=total-js
        if best_pair<-1e90:
            # No second task: use expected first-task value.
            best_pair=0.0; fail_penalty=pars[11]
            for g in range(G):best_pair+=probs[g]*weights[i]*(xi[g]-fail_penalty*(1.0-xi[g]))
            pair_short=best_pair; pair_future=0.0
        option=0.0; recourse_proxy=0.0; downside_proxy=0.0
        if pars[9]>1e-14 or pars[13]>1e-14 or pars[15]>1e-14:
            option,recourse_proxy,downside_proxy=recourse_option_bonus(i,t1,e1,xi,rem_after_i,points,probs,ft,fe,weights,deadline,pars,aa,bb,dd,ee,force_mode)
        aligned_value=best_pair+pars[9]*option
        if pars[13]>1e-14 and recourse_proxy>0.0:
            pfirst=0.0
            for g in range(G):pfirst+=probs[g]*xi[g]
            blend=pars[13]+pars[14]*4.0*pfirst*(1.0-pfirst)
            if blend>1.0:blend=1.0
            recourse_target=recourse_proxy
            if pars[15]>1e-14:
                recourse_target=(1.0-pars[15])*recourse_proxy+pars[15]*downside_proxy
            aligned_value=(1.0-blend)*aligned_value+blend*recourse_target
        candidate_scores[i]=aligned_value
        if aligned_value>best_val+1e-12 or (abs(aligned_value-best_val)<=1e-12 and (best_i==0 or i<best_i)):
            best_i=i; best_val=aligned_value; best_short=pair_short; best_future=aligned_value-pair_short
    return best_i,best_val,best_short,best_future,expanded_total,candidate_scores


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
