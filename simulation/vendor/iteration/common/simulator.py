from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from .config import Config, BASE_LONLAT, MEC_LONLAT

EARTH_R = 6371000.0
C0 = 299792458.0


def project_lonlat(lon: float, lat: float, lon0=BASE_LONLAT[0], lat0=BASE_LONLAT[1]):
    x = math.radians(lon-lon0)*EARTH_R*math.cos(math.radians(lat0))
    y = math.radians(lat-lat0)*EARTH_R
    return np.array([x,y], dtype=float)

@dataclass
class Task:
    idx:int; lon:float; lat:float; xy:np.ndarray; region:int
    data_mbit:float; deadline_s:float; weight:float
    mu:float; sigma:float; workload_gcy:float; output_ratio:float
    ul_rates:np.ndarray; dl_rates:np.ndarray

@dataclass
class Mission:
    seed:int; cfg:Config; tasks:list[Task]
    base_xy:np.ndarray; mec_xy:np.ndarray
    dist:np.ndarray; flight_time:np.ndarray; flight_energy_kj:np.ndarray

@dataclass
class State:
    node:int=0; time_s:float=0.0; energy_kj:float=0.0
    visited:set[int]=field(default_factory=set)
    completed:set[int]=field(default_factory=set)
    route:list[int]=field(default_factory=list)
    local_actions:int=0; mec_actions:int=0; skipped:int=0

@dataclass
class ExecResult:
    feasible:bool; mode:str; time_s:float=0.0; energy_kj:float=0.0
    cpu_ghz:float=0.0; mec_idx:int=-1; normalized_cost:float=1e100


def _channel_rates(task_xy, mec_xy, cfg):
    ru=[]; rd=[]
    for s in mec_xy:
        d=float(np.linalg.norm(task_xy-s)); h=math.sqrt(d*d+cfg.uav_altitude_m**2)
        theta=180/math.pi*math.asin(cfg.uav_altitude_m/h)
        p=1/(1+cfg.los_a*math.exp(-cfg.los_b*(theta-cfg.los_a)))
        loss=(4*math.pi*cfg.fc_hz*h/C0)**2*(p*cfg.eta_los+(1-p)*cfg.eta_nlos)
        g=1/loss
        snru=cfg.tx_power_w*g/(cfg.noise_psd_w_hz*cfg.ul_bw_hz)
        snrd=cfg.mec_tx_power_w*g/(cfg.noise_psd_w_hz*cfg.dl_bw_hz)
        ru.append(cfg.ul_bw_hz*math.log2(1+snru)/1e6)
        rd.append(cfg.dl_bw_hz*math.log2(1+snrd)/1e6)
    return np.asarray(ru),np.asarray(rd)


def generate_mission(seed:int,cfg:Config)->Mission:
    """Generate one mission on the fixed coastal Macau map used in Fig. 4.

    Geography is fixed across missions. Priority is spatially structured: tasks
    farther from the nearest MEC, especially the explicit offshore tasks, receive
    higher inspection priority. Workload means are kept fixed when uncertainty
    changes so that the Mean-vs-distribution comparison isolates uncertainty.
    """
    rng=np.random.default_rng(seed)
    base_xy=np.zeros(2); mec_xy=np.stack([project_lonlat(*p) for p in MEC_LONLAT])

    task_lonlat=(
        (113.5355,22.2100),(113.5454,22.2162),(113.5558,22.2126),(113.5594,22.2040),(113.5358,22.1978),
        (113.5480,22.1616),(113.5572,22.1641),(113.5691,22.1588),(113.5475,22.1525),
        (113.5538,22.1454),(113.5648,22.1442),(113.5752,22.1396),
        (113.5538,22.1322),(113.5689,22.1288),(113.5805,22.1222),(113.5712,22.1092),(113.5560,22.1090),(113.5488,22.1166),
        (113.5912,22.1448),(113.5943,22.1212),
    )
    regions=(0,0,0,0,0, 1,1,1,1, 2,2,2, 3,3,3,3,3,3, 2,3)
    if cfg.n_tasks!=len(task_lonlat):
        raise ValueError(f"Main-comparison map fixes n_tasks={len(task_lonlat)}; got {cfg.n_tasks}")

    task_xy=np.stack([project_lonlat(*p) for p in task_lonlat])
    d_near=np.min(np.linalg.norm(task_xy[:,None,:]-mec_xy[None,:,:],axis=2),axis=1)
    d_norm=(d_near-d_near.min())/max(d_near.max()-d_near.min(),1e-12)
    offshore=np.zeros(len(task_lonlat))
    offshore[18:20]=1.0  # T19-T20 are explicit nearshore/offshore inspection points.
    score=cfg.priority_distance_weight*d_norm+cfg.priority_offshore_weight*offshore

    # Exactly four tasks in each priority level. The rule depends only on map
    # geometry and is fixed before workload realizations are generated.
    order=np.argsort(score)
    weight_by_idx=np.zeros(len(task_lonlat),dtype=float)
    urgency_by_idx=np.zeros(len(task_lonlat),dtype=int)
    if cfg.priority_scheme=="critical":
        # Coastal-critical profile: most near-MEC tasks are low priority, while
        # the eight most remote coastal tasks dominate mission value.
        rank_weights=[1.]*8+[3.]*4+[10.]*4+[15.]*4
    elif cfg.priority_scheme=="critical2":
        rank_weights=[1.]*10+[3.]*2+[10.]*4+[15.]*4
    else:
        rank_weights=[1.]*4+[3.]*4+[6.]*4+[10.]*4+[15.]*4
    for rank,idx0 in enumerate(order):
        w=float(rank_weights[rank])
        weight_by_idx[idx0]=w
        urgency_by_idx[idx0]={1.:1,3.:2,6.:3,10.:4,15.:5}[w]

    tasks=[]
    for idx,((lon,lat),r) in enumerate(zip(task_lonlat,regions),start=1):
        xy=task_xy[idx-1]; data=50.0
        urgency=int(urgency_by_idx[idx-1]); weight=float(weight_by_idx[idx-1])

        # Fixed target mean workload. Changing sigma does not change E[C_i].
        region_shift=(45.,20.,0.,-5.)[int(r)]
        mean_target=(95.+1.35*data+region_shift)*cfg.workload_scale
        mean_target*=1.0+0.045*(urgency-1)
        # Remote critical coastal tasks can require more detailed processing.
        # This is a spatial scenario property and is fixed before any test run.
        mean_target*=1.0+cfg.mean_spatial_span*score[idx-1]
        sigma=(cfg.sigma_min+cfg.sigma_span*score[idx-1])*cfg.uncertainty_scale
        mu=math.log(mean_target)-0.5*sigma*sigma
        workload=float(math.exp(mu+sigma*rng.normal()))

        ul,dl=_channel_rates(xy,mec_xy,cfg)
        # High-priority remote coastal tasks can have tighter response windows.
        # The deadline rule depends only on the spatial criticality score and
        # the task's mean computational demand, never on the realized workload.
        base_dist=float(np.linalg.norm(xy)); nominal_flight=base_dist/cfg.speed_mps
        if cfg.deadline_mode=="critical":
            best_exec=mean_target/cfg.local_f_max_ghz
            for mm,(ru,rd,fm) in enumerate(zip(ul,dl,cfg.mec_cpu_ghz)):
                if ru>=cfg.ul_min_mbps and rd>=cfg.dl_min_mbps:
                    tm=data/ru+mean_target/fm+cfg.output_ratio*data/rd
                    if tm<best_exec:best_exec=tm
            fac=cfg.deadline_factor_high+(cfg.deadline_factor_low-cfg.deadline_factor_high)*(1.0-score[idx-1])
            deadline=nominal_flight+cfg.deadline_slack_s+fac*best_exec
            deadline*=cfg.deadline_scale
            deadline=float(np.clip(deadline,430.,cfg.task_deadline_cap_s))
        else:
            slack=500.+rng.uniform(230.,620.)+(6-urgency)*70.
            deadline=.55*nominal_flight+cfg.deadline_scale*slack
            deadline=float(np.clip(deadline,620.,cfg.task_deadline_cap_s))
        tasks.append(Task(idx,lon,lat,xy,int(r),data,deadline,weight,mu,sigma,workload,cfg.output_ratio,ul,dl))

    nodes=np.vstack([base_xy,task_xy])
    diff=nodes[:,None,:]-nodes[None,:,:]; dist=np.linalg.norm(diff,axis=2)
    ft=dist/cfg.speed_mps; fe=ft*cfg.flight_power_w/1000.
    return Mission(seed,cfg,tasks,base_xy,mec_xy,dist,ft,fe)


def available_resources(mission:Mission,state:State,i:int):
    cfg=mission.cfg
    at=state.time_s+mission.flight_time[state.node,i]
    ae=state.energy_kj+mission.flight_energy_kj[state.node,i]
    dt=min(mission.tasks[i-1].deadline_s-at,cfg.t_max-at-mission.flight_time[i,0])
    de=cfg.e_max_kj-ae-mission.flight_energy_kj[i,0]
    return at,ae,dt,de


def visit_feasible(mission:Mission,state:State,i:int):
    at,ae,dt,de=available_resources(mission,state,i)
    return dt>0 and de>0


def solve_execution(mission:Mission,state:State,i:int,c:float,force_mode:str='auto')->ExecResult:
    cfg=mission.cfg; at,ae,dt,de=available_resources(mission,state,i)
    if dt<=0 or de<=0:return ExecResult(False,'skip')
    task=mission.tasks[i-1]; best=None
    if force_mode in ('auto','local'):
        k=cfg.kappa_kj_per_gcycle_ghz2; ph=cfg.hover_power_w/1000.
        lo=max(cfg.local_f_min_ghz,c/dt); hi=cfg.local_f_max_ghz
        if lo<=hi+1e-12:
            lo=min(lo,hi)
            def energy(f):return k*c*f*f+ph*c/f
            fm=min(hi,max(lo,(ph/(2*k))**(1/3)))
            if energy(fm)<=de+1e-9:
                if energy(lo)>de:
                    a,b=lo,fm
                    for _ in range(60):
                        x=(a+b)/2
                        if energy(x)>de:a=x
                        else:b=x
                    lo=b
                if energy(hi)>de:
                    a,b=fm,hi
                    for _ in range(60):
                        x=(a+b)/2
                        if energy(x)>de:b=x
                        else:a=x
                    hi=a
                fsta=((de+ph*dt)/(2*k*dt))**(1/3)
                fmid=min(hi,max(lo,fsta))
                for f in (lo,fmid,hi):
                    t=c/f; e=energy(f)
                    if t<=dt+1e-8 and e<=de+1e-8:
                        cost=t/dt+e/de
                        rr=ExecResult(True,'local',t,e,f,-1,cost)
                        if best is None or rr.normalized_cost<best.normalized_cost-1e-12:best=rr
    if force_mode in ('auto','mec'):
        for m,(ru,rd,fm) in enumerate(zip(task.ul_rates,task.dl_rates,cfg.mec_cpu_ghz)):
            if ru<cfg.ul_min_mbps or rd<cfg.dl_min_mbps:continue
            tu=task.data_mbit/ru; td=task.output_ratio*task.data_mbit/rd
            t=tu+c/fm+td
            e=cfg.tx_power_w*tu/1000+cfg.rx_power_w*td/1000+cfg.hover_power_w*t/1000
            if t<=dt+1e-9 and e<=de+1e-9:
                rr=ExecResult(True,'mec',t,e,0.,m,t/dt+e/de)
                if best is None or rr.normalized_cost<best.normalized_cost-1e-12:best=rr
    return best if best is not None else ExecResult(False,'skip')


def apply_actual_task(mission:Mission,state:State,i:int,force_mode='auto'):
    old=State(state.node,state.time_s,state.energy_kj,set(state.visited),set(state.completed),list(state.route),state.local_actions,state.mec_actions,state.skipped)
    state.time_s += mission.flight_time[old.node,i]
    state.energy_kj += mission.flight_energy_kj[old.node,i]
    state.node=i; state.visited.add(i); state.route.append(i)
    ex=solve_execution(mission,old,i,mission.tasks[i-1].workload_gcy,force_mode)
    if ex.feasible:
        state.time_s += ex.time_s; state.energy_kj += ex.energy_kj; state.completed.add(i)
        if ex.mode=='local':state.local_actions+=1
        elif ex.mode=='mec':state.mec_actions+=1
    else:state.skipped+=1
    return ex


def finish_return(mission:Mission,state:State):
    if state.node!=0:
        state.time_s+=mission.flight_time[state.node,0]; state.energy_kj+=mission.flight_energy_kj[state.node,0]; state.node=0
    return state


def solve_execution_v2(mission:Mission,state:State,i:int,c:float,
                       force_mode:str='auto',time_weight:float=1.0,
                       energy_weight:float=1.0,balance_weight:float=0.0)->ExecResult:
    """Refined MTE-Sub rule used by ProposedV2.

    Feasibility is identical to ``solve_execution``.  The only change is the
    action ranking cost: weighted normalized time/energy utilization plus an
    optional bottleneck-utilization term.
    """
    cfg=mission.cfg
    at,ae,dt,de=available_resources(mission,state,i)
    if dt<=0 or de<=0:
        return ExecResult(False,'skip')

    wt=max(float(time_weight),1e-12)
    we=max(float(energy_weight),1e-12)
    bw=max(float(balance_weight),0.0)
    task=mission.tasks[i-1]
    best=None

    if force_mode in ('auto','local'):
        k=cfg.kappa_kj_per_gcycle_ghz2
        ph=cfg.hover_power_w/1000.0
        lo=max(cfg.local_f_min_ghz,c/dt)
        hi=cfg.local_f_max_ghz
        if lo<=hi+1e-12:
            lo=min(lo,hi)
            def energy(f):
                return k*c*f*f+ph*c/f
            f_energy=min(hi,max(lo,(ph/(2*k))**(1/3)))
            if energy(f_energy)<=de+1e-9:
                if energy(lo)>de:
                    a,b=lo,f_energy
                    for _ in range(60):
                        x=(a+b)/2
                        if energy(x)>de:a=x
                        else:b=x
                    lo=b
                if energy(hi)>de:
                    a,b=f_energy,hi
                    for _ in range(60):
                        x=(a+b)/2
                        if energy(x)>de:b=x
                        else:a=x
                    hi=a

                f_sum=((wt*de+we*ph*dt)/(2*we*k*dt))**(1/3)
                f_time=(((wt+bw)*de+we*ph*dt)/(2*we*k*dt))**(1/3)
                f_energy_region=((wt*de+(we+bw)*ph*dt)/(2*(we+bw)*k*dt))**(1/3)
                rhs=de/dt-ph
                f_equal=f_energy if rhs<=0 else (rhs/k)**(1/3)
                for f0 in (lo,hi,f_energy,f_sum,f_time,f_energy_region,f_equal):
                    f=min(hi,max(lo,f0))
                    xt=c/f
                    xe=energy(f)
                    if xt<=dt+1e-8 and xe<=de+1e-8:
                        ut=xt/dt; ue=xe/de
                        cost=wt*ut+we*ue+bw*max(ut,ue)
                        rr=ExecResult(True,'local',xt,xe,f,-1,cost)
                        if best is None or rr.normalized_cost<best.normalized_cost-1e-12:
                            best=rr

    if force_mode in ('auto','mec'):
        for m,(ru,rd,fm) in enumerate(zip(task.ul_rates,task.dl_rates,cfg.mec_cpu_ghz)):
            if ru<cfg.ul_min_mbps or rd<cfg.dl_min_mbps:
                continue
            tu=task.data_mbit/ru
            td=task.output_ratio*task.data_mbit/rd
            xt=tu+c/fm+td
            xe=(cfg.tx_power_w*tu/1000.0+
                cfg.rx_power_w*td/1000.0+
                cfg.hover_power_w*xt/1000.0)
            if xt<=dt+1e-9 and xe<=de+1e-9:
                ut=xt/dt; ue=xe/de
                cost=wt*ut+we*ue+bw*max(ut,ue)
                rr=ExecResult(True,'mec',xt,xe,0.0,m,cost)
                if best is None or rr.normalized_cost<best.normalized_cost-1e-12:
                    best=rr

    return best if best is not None else ExecResult(False,'skip')


def apply_actual_task_v2(mission:Mission,state:State,i:int,force_mode='auto',
                         time_weight:float=1.0,energy_weight:float=1.0,
                         balance_weight:float=0.0):
    old=State(state.node,state.time_s,state.energy_kj,set(state.visited),
              set(state.completed),list(state.route),state.local_actions,
              state.mec_actions,state.skipped)
    state.time_s += mission.flight_time[old.node,i]
    state.energy_kj += mission.flight_energy_kj[old.node,i]
    state.node=i
    state.visited.add(i)
    state.route.append(i)
    ex=solve_execution_v2(mission,old,i,mission.tasks[i-1].workload_gcy,
                          force_mode,time_weight,energy_weight,balance_weight)
    if ex.feasible:
        state.time_s += ex.time_s
        state.energy_kj += ex.energy_kj
        state.completed.add(i)
        if ex.mode=='local':
            state.local_actions+=1
        elif ex.mode=='mec':
            state.mec_actions+=1
    else:
        state.skipped+=1
    return ex
