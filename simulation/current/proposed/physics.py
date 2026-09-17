"""Exact physical kernels shared by the original six comparison methods.

Extracted without mathematical changes from the frozen research implementation.
Function bodies/decorators are preserved verbatim; only host-specific imports
and unrelated planning code are omitted. All arithmetic and original function bodies are retained.
No realized workloads are stored: callers explicitly provide each tested value.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit

# Source: experiments/mte_fusion_comparison/src/vendor/iteration/common/mte_v2.py
@njit(cache=True)
def execute_virtual_v2(c, dt, de, pars, aa, bb, dd, ee, force_mode,
                       time_weight, energy_weight, balance_weight):
    """Solve the refined MTE-Sub action for one workload value.

    ``aa + bb*c`` and ``dd + ee*c`` are MEC time and onboard-energy models.
    ``force_mode`` is 0 (auto), 1 (local only), or 2 (MEC only).
    """
    if c <= 0.0 or dt <= 0.0 or de <= 0.0:
        return False, 0.0, 0.0, -1, 0.0, 1e100

    wt = max(time_weight, 1e-12)
    we = max(energy_weight, 1e-12)
    bw = max(balance_weight, 0.0)
    fmin = pars[2]
    fmax = pars[3]
    kappa = pars[4]
    ph = pars[5]

    best = 1e100
    best_t = 0.0
    best_e = 0.0
    best_mode = -1
    best_f = 0.0

    if force_mode != 2:
        lo = max(fmin, c / dt)
        hi = fmax
        if lo <= hi + 1e-12:
            if lo > hi:
                lo = hi

            # Local CPU energy is U-shaped; first intersect [lo, hi] with its
            # energy-feasible subinterval.
            f_energy = (ph / (2.0 * kappa)) ** (1.0 / 3.0)
            f_energy = min(hi, max(lo, f_energy))
            e_min = kappa * c * f_energy * f_energy + ph * c / f_energy
            if e_min <= de + 1e-9:
                e_lo = kappa * c * lo * lo + ph * c / lo
                if e_lo > de:
                    a = lo
                    b = f_energy
                    for _ in range(55):
                        x = 0.5 * (a + b)
                        ex = kappa * c * x * x + ph * c / x
                        if ex > de:
                            a = x
                        else:
                            b = x
                    lo = b

                e_hi = kappa * c * hi * hi + ph * c / hi
                if e_hi > de:
                    a = f_energy
                    b = hi
                    for _ in range(55):
                        x = 0.5 * (a + b)
                        ex = kappa * c * x * x + ph * c / x
                        if ex > de:
                            b = x
                        else:
                            a = x
                    hi = a

                # The objective is convex and piecewise smooth.  Its optimum
                # is at a boundary, a smooth-region stationary point, or the
                # time/energy-utilization equality kink.
                f_sum = ((wt * de + we * ph * dt) /
                         (2.0 * we * kappa * dt)) ** (1.0 / 3.0)
                f_time_region = (((wt + bw) * de + we * ph * dt) /
                                 (2.0 * we * kappa * dt)) ** (1.0 / 3.0)
                f_energy_region = ((wt * de + (we + bw) * ph * dt) /
                                   (2.0 * (we + bw) * kappa * dt)) ** (1.0 / 3.0)
                rhs = de / dt - ph
                f_equal = f_energy
                if rhs > 0.0:
                    f_equal = (rhs / kappa) ** (1.0 / 3.0)

                for f0 in (lo, hi, f_energy, f_sum, f_time_region,
                           f_energy_region, f_equal):
                    f = min(hi, max(lo, f0))
                    xt = c / f
                    xe = kappa * c * f * f + ph * xt
                    if xt <= dt + 1e-8 and xe <= de + 1e-8:
                        ut = xt / dt
                        ue = xe / de
                        cost = wt * ut + we * ue + bw * max(ut, ue)
                        if cost < best - 1e-12:
                            best = cost
                            best_t = xt
                            best_e = xe
                            best_mode = 0
                            best_f = f

    if force_mode != 1:
        for m in range(len(aa)):
            if aa[m] >= 1e90:
                continue
            xt = aa[m] + bb[m] * c
            xe = dd[m] + ee[m] * c
            if xt <= dt + 1e-9 and xe <= de + 1e-9:
                ut = xt / dt
                ue = xe / de
                cost = wt * ut + we * ue + bw * max(ut, ue)
                if cost < best - 1e-12:
                    best = cost
                    best_t = xt
                    best_e = xe
                    best_mode = m + 1
                    best_f = 0.0

    return (best_mode >= 0, best_t, best_e, best_mode, best_f, best)


# Source: experiments/mte_fusion_comparison/src/vendor/iteration/common/mte_v2.py
@njit(cache=True)
def visit_ok_v2(node, t, e, i, ft, fe, deadline, pars):
    at = t + ft[node, i]
    ae = e + fe[node, i]
    return (at + ft[i, 0] <= pars[0] + 1e-9 and
            ae + fe[i, 0] <= pars[1] + 1e-9 and
            deadline[i] - at > 0.0)


# Source: experiments/mte_fusion_comparison/src/vendor/iteration/common/mte_v2.py
@njit(cache=True)
def resource_limits_v2(node, t, e, i, ft, fe, deadline, pars):
    at = t + ft[node, i]
    ae = e + fe[node, i]
    dt = min(deadline[i] - at, pars[0] - at - ft[i, 0])
    de = pars[1] - ae - fe[i, 0]
    return at, ae, dt, de


# Source: experiments/mte_fusion_comparison/src/vendor/robust/common/mte_core.py
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


# Source: experiments/mte_fusion_comparison/src/vendor/robust/common/mte_core.py
@njit(cache=True)
def resource_limits(node,t,e,i,ft,fe,deadline,pars):
    at=t+ft[node,i]; ae=e+fe[node,i]
    dt=min(deadline[i]-at,pars[0]-at-ft[i,0]); de=pars[1]-ae-fe[i,0]
    return at,ae,dt,de


# Source: experiments/mte_fusion_comparison/src/vendor/robust/common/mte_core.py
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


# Source: experiments/mte_fusion_comparison/src/vendor/robust/common/mte_core.py
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


# Source: experiments/mte_fusion_comparison/src/vendor/robust/common/mte_core.py
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


# Source: experiments/mte_policy_iteration/src/root_calibrated_core.py
@njit(cache=True)
def _local_capacity(dt, de, pars):
    """Maximum local workload from the finite analytic candidate set."""

    if dt <= 0.0 or de <= 0.0:
        return 0.0
    fmin = pars[2]
    fmax = pars[3]
    kappa = pars[4]
    hover = pars[5]
    candidates = np.zeros(4)
    count = 0
    candidates[count] = fmin
    count += 1
    candidates[count] = fmax
    count += 1
    if kappa > 0.0:
        candidates[count] = (hover / (2.0 * kappa)) ** (1.0 / 3.0)
        count += 1
        rhs = de / dt - hover
        if rhs > 0.0:
            candidates[count] = (rhs / kappa) ** (1.0 / 3.0)
            count += 1

    capacity = 0.0
    for index in range(count):
        frequency = min(fmax, max(fmin, candidates[index]))
        energy_per_workload = kappa * frequency * frequency + hover / frequency
        value = min(dt * frequency, de / max(energy_per_workload, 1e-15))
        if value > capacity:
            capacity = value
    return max(0.0, capacity)


# Source: experiments/mte_policy_iteration/src/root_calibrated_core.py
@njit(cache=True)
def _maximum_capacity(task, dt, de, pars, aa, bb, dd, ee):
    """Maximum workload executable by Local or one available MEC server."""

    capacity = _local_capacity(dt, de, pars)
    for mec in range(aa.shape[1]):
        if aa[task, mec] >= 1e90:
            continue
        if bb[task, mec] > 0.0:
            time_capacity = (dt - aa[task, mec]) / bb[task, mec]
        else:
            time_capacity = 1e100
        if ee[task, mec] > 0.0:
            energy_capacity = (de - dd[task, mec]) / ee[task, mec]
        else:
            energy_capacity = 1e100
        mec_capacity = max(0.0, min(time_capacity, energy_capacity))
        if mec_capacity > capacity:
            capacity = mec_capacity
    return max(0.0, capacity)


# Source: experiments/mte_policy_iteration/src/root_calibrated_core.py
@njit(cache=True)
def _lognormal_cdf(mu, sigma, threshold):
    if threshold <= 0.0:
        return 0.0
    if sigma <= 1e-12:
        return 1.0 if math.exp(mu) <= threshold else 0.0
    z = (math.log(threshold) - mu) / (sigma * math.sqrt(2.0))
    return min(1.0, max(0.0, 0.5 * (1.0 + math.erf(z))))



@njit(cache=True)
def expected_task_stats_v2(node, t, e, i, points, probs, ft, fe, deadline,
                           pars, aa, bb, dd, ee, force_mode,
                           time_weight, energy_weight, balance_weight):
    """Completion probability and unconditional expected execution resources."""
    at, ae, dt, de = resource_limits_v2(node, t, e, i, ft, fe, deadline, pars)
    if dt <= 0.0 or de <= 0.0:
        return 0.0, 0.0, 0.0
    p = 0.0
    et = 0.0
    en = 0.0
    for g in range(len(probs)):
        ok, xt, xe, mode, f, cost = execute_virtual_v2(
            points[i, g], dt, de, pars, aa[i], bb[i], dd[i], ee[i],
            force_mode, time_weight, energy_weight, balance_weight,
        )
        if ok:
            pr = probs[g]
            p += pr
            et += pr * xt
            en += pr * xe
    return p, et, en


@njit(cache=True)
def _lognormal_ppf(mu, sigma, probability):
    """Deterministic inverse LogNormal CDF for conditional quadrature."""

    if sigma <= 1e-12:
        return math.exp(mu)
    target = min(1.0 - 1e-15, max(1e-15, probability))
    left = -12.0
    right = 12.0
    for _ in range(58):
        middle = 0.5 * (left + right)
        cdf = 0.5 * (1.0 + math.erf(middle / math.sqrt(2.0)))
        if cdf < target:
            left = middle
        else:
            right = middle
    return math.exp(mu + sigma * (0.5 * (left + right)))


@njit(cache=True)
def _truncated_lognormal_first_moment(mu, sigma, threshold):
    """Return E[C 1(C <= threshold)] for a LogNormal workload."""

    if threshold <= 0.0:
        return 0.0
    if sigma <= 1e-12:
        workload = math.exp(mu)
        return workload if workload <= threshold else 0.0
    z = (
        math.log(threshold) - mu - sigma * sigma
    ) / (sigma * math.sqrt(2.0))
    return math.exp(mu + 0.5 * sigma * sigma) * (
        0.5 * (1.0 + math.erf(z))
    )

