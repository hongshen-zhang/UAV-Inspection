"""Execute the three literature baselines using freshly generated inputs."""
from pathlib import Path
import importlib
import json
import sys

HERE = Path(__file__).resolve().parent

def literature_run(public,hidden,method,seed):
    path=str(HERE/'literature')
    if path not in sys.path:sys.path.insert(0,path)
    from environment import Environment
    module=importlib.import_module({'IO':'baselines.shiri','Rollout':'baselines.novoa','ADAPT':'adapt_policy'}[method])
    env=Environment(public);policy=module.make_policy(env,int(seed))
    state=env.initial_state();route=[0];trace=[];reward=0.;local=mec=completed=0
    for epoch in range(env.n_tasks):
        task=int(policy.select_task(state))
        if task==0:break
        assert task in env.tasks(state)
        workload=float(hidden[task])
        allow=policy.arrival_decision(state,task,workload) if hasattr(policy,'arrival_decision') else True
        tr=env.step(state,task,workload) if allow else env.skip_step(state,task)
        state=tr.state;route.append(task);reward+=tr.reward
        completed+=int(tr.completed);local+=int(tr.mode=='local');mec+=int(tr.mode=='mec')
        assert state.time_s+env.flight_time[task,0]<=env.t_max+1e-7
        assert state.energy_kj+env.flight_energy[task,0]<=env.e_max+1e-7
        if tr.completed:assert state.time_s<=env.deadline[task]+1e-7
        trace.append(dict(task=task,revealed_workload_gcy=workload,mode=tr.mode,frequency_ghz=tr.frequency_ghz,mec=tr.mec,time_s=state.time_s,energy_kj=state.energy_kj))
    rt,re=env.return_cost(state)
    return dict(wcr=reward/float(env.weights.sum()),weighted_completed=reward,total_weight=float(env.weights.sum()),completed_tasks=completed,visited_tasks=len(trace),skipped_tasks=len(trace)-completed,local_actions=local,mec_actions=mec,final_time_s=state.time_s+rt,final_energy_kj=state.energy_kj+re,return_success=1,route='-'.join(map(str,route+[0])),trace=json.dumps(trace,separators=(',',':')))
