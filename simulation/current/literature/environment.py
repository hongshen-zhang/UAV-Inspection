"""Public-only UAV environment. Units: seconds, kJ, Gcycles and GHz.

All numerical kernels are identical to the audited frozen experiment. The
constructor copies only public arrays; it stores no actual workload table.
"""
from __future__ import annotations
import os,sys
from pathlib import Path
from dataclasses import dataclass
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ.setdefault('NUMBA_CACHE_DIR',str(HERE/'runtime/numba_cache'))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
 os.environ.setdefault(key,'1')
import numpy as np
from numba import njit
import portable_physics as iteration_v2
import portable_physics as robust_core
from portable_physics import _maximum_capacity, _lognormal_cdf

@dataclass(frozen=True)
class State:
 node:int=0
 time_s:float=0.0
 energy_kj:float=0.0
 remaining_mask:int=0

@dataclass(frozen=True)
class Transition:
 state:State
 reward:float
 completed:bool
 completion_feasible:bool
 mode:str
 service_time_s:float
 service_energy_kj:float
 frequency_ghz:float=0.0
 mec:int=-1

@njit(cache=True)
def step_kernel(node,t,e,mask,task,c,ft,fe,weights,deadline,pars,aa,bb,dd,ee):
 """Return (next_t,next_e,next_mask,reward,completion_feasible,mode,freq,service_t,service_e).
 mode=-2 invalid visit, -1 safe skip, 0 local, m+1 MEC. Never reads real workload.
 Matches existing HeuristicPlanner's minimum-occupation feasible executor.
 """
 if task<1 or not (mask & (1<<(task-1))) or not iteration_v2.visit_ok_v2(node,t,e,task,ft,fe,deadline,pars):
  return t,e,mask,0.0,False,-2,0.0,0.0,0.0
 at,ae,dt,de=iteration_v2.resource_limits_v2(node,t,e,task,ft,fe,deadline,pars)
 # Exact shared Local optimizer; MEC tie-breaking matches _completion_actions.
 ok,st,se,mode,freq,cost=iteration_v2.execute_virtual_v2(c,dt,de,pars,aa[task],bb[task],dd[task],ee[task],1,1.,1.,0.)
 best_mode=0 if ok else -1
 best_cost=cost if ok else 1e100
 for m in range(aa.shape[1]):
  if aa[task,m]>=1e90:continue
  mt=aa[task,m]+bb[task,m]*c;me=dd[task,m]+ee[task,m]*c
  if mt>dt+1e-9 or me>de+1e-9:continue
  mc=mt/max(dt,1e-12)+me/max(de,1e-12)
  # Python min key: (occupation, service_time, service_energy, mec).
  old_mec=-1 if best_mode==0 else best_mode-1
  if best_mode<0 or (mc,mt,me,m)<(best_cost,st,se,old_mec):
   best_cost=mc;st=mt;se=me;best_mode=m+1;freq=0.
 nm=mask & ~(1<<(task-1))
 if best_mode<0:return at,ae,nm,0.0,False,-1,0.0,0.0,0.0
 return at+st,ae+se,nm,weights[task],True,best_mode,freq,st,se

@njit(cache=True)
def probability_kernel(node,t,e,task,ft,fe,deadline,pars,mu,sigma,aa,bb,dd,ee):
 if not iteration_v2.visit_ok_v2(node,t,e,task,ft,fe,deadline,pars):return 0.
 at,ae,dt,de=iteration_v2.resource_limits_v2(node,t,e,task,ft,fe,deadline,pars)
 cap=_maximum_capacity(task,dt,de,pars,aa,bb,dd,ee)
 return _lognormal_cdf(mu[task],sigma[task],cap)

@njit(cache=True)
def expected_cost_kernel(node,t,e,task,ft,fe,weights,deadline,pars,mu,sigma,aa,bb,dd,ee,points,probabilities):
 p,st,se=robust_core.calibrated_task_stats(node,t,e,task,points,probabilities,mu,sigma,ft,fe,deadline,pars,aa,bb,dd,ee,0)
 return (ft[node,task]+st)/max(pars[0]-t,1e-12)+(fe[node,task]+se)/max(pars[1]-e,1e-12)

class Environment:
 """Only public constants/distribution arrays; no bundle, tasks or true workloads retained."""
 def __init__(self,public_arrays):
  for name in ('weights','deadline','mu','sigma','pars','aa','bb','dd','ee','flight_time','flight_energy','points','probabilities','mec_frequencies','mean_workloads'):
   setattr(self,name,np.array(public_arrays[name],copy=True))
  self.n_tasks=len(self.weights)-1
  self.t_max=float(self.pars[0]);self.e_max=float(self.pars[1])
  for value in vars(self).values():
   if isinstance(value,np.ndarray):value.flags.writeable=False
 def initial_state(self):return State(0,0.,0.,(1<<self.n_tasks)-1)
 def tasks(self,state):
  return tuple(i for i in range(1,self.n_tasks+1) if state.remaining_mask & (1<<(i-1)) and iteration_v2.visit_ok_v2(state.node,state.time_s,state.energy_kj,i,self.flight_time,self.flight_energy,self.deadline,self.pars))
 def sample_workloads(self,rng,count):
  a=np.exp(self.mu[None,:]+self.sigma[None,:]*rng.standard_normal((int(count),self.n_tasks+1)));a[:,0]=0.;return a
 def step(self,state,task,workload):
  if task==0:
   rt,re=self.return_cost(state)
   return Transition(State(0,state.time_s+rt,state.energy_kj+re,state.remaining_mask),0.,False,True,'return',0.,0.)
  nt,ne,nm,reward,ok,mode,freq,st,se=step_kernel(state.node,state.time_s,state.energy_kj,state.remaining_mask,int(task),float(workload),self.flight_time,self.flight_energy,self.weights,self.deadline,self.pars,self.aa,self.bb,self.dd,self.ee)
  if mode==-2:raise ValueError('Task is not a legal unvisited safe visit')
  if mode>0:freq=float(self.mec_frequencies[mode-1])
  return Transition(State(int(task),nt,ne,int(nm)),reward,bool(ok),bool(ok),'skip' if mode<0 else 'local' if mode==0 else 'mec',st,se,freq,mode-1 if mode>0 else -1)
 def completion_probability(self,state,task):
  return float(probability_kernel(state.node,state.time_s,state.energy_kj,int(task),self.flight_time,self.flight_energy,self.deadline,self.pars,self.mu,self.sigma,self.aa,self.bb,self.dd,self.ee))
 def expected_cost(self,state,task):
  return float(expected_cost_kernel(state.node,state.time_s,state.energy_kj,int(task),self.flight_time,self.flight_energy,self.weights,self.deadline,self.pars,self.mu,self.sigma,self.aa,self.bb,self.dd,self.ee,self.points,self.probabilities))
 def return_cost(self,state):return float(self.flight_time[state.node,0]),float(self.flight_energy[state.node,0])

def _skip_step(self,state,task):
 if int(task) not in self.tasks(state):raise ValueError('Illegal forced-skip visit')
 return Transition(State(int(task),state.time_s+float(self.flight_time[state.node,task]),state.energy_kj+float(self.flight_energy[state.node,task]),state.remaining_mask & ~(1<<(int(task)-1))),0.,False,False,'skip',0.,0.)
Environment.skip_step=_skip_step
