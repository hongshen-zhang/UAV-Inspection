"""Simple online controls using the identical physical executor."""
import math
from planner import ConsistentACARPlanner
from batch_adapter import robust_core,iteration_v2

class HeuristicPlanner(ConsistentACARPlanner):
    rule='density'
    def select_task(self,state):
        self.stats['route_calls']+=1
        s=self._plan_state(state);best=0;value=0
        for i in range(1,self.n_tasks+1):
            if not s.remaining_mask & (1<<(i-1)):continue
            arrival=self._arrival(s,i)
            if arrival is None:continue
            p,t,e=robust_core.calibrated_task_stats(s.node,s.time_s,s.energy_kj,i,
                self.price_points,self.price_probabilities,self.mu,self.sigma,
                self.mission.flight_time,self.mission.flight_energy_kj,self.deadline,self.pars,
                self.aa,self.bb,self.dd,self.ee,0)
            if p<=1e-12:continue
            cost=(self.mission.flight_time[s.node,i]+t)/max(self.pars[0]-s.time_s,1e-9)
            cost+=(self.mission.flight_energy_kj[s.node,i]+e)/max(self.pars[1]-s.energy_kj,1e-9)
            v=self.weights[i]*p/max(cost,1e-9)
            if self.rule=='nearest':v=1/max(self.mission.flight_time[s.node,i],1e-9)
            if v>value+1e-12:best,value=i,v
        return best,{'selection_mode':self.rule,'selected_value':float(value)}
    def _prices(self,*args):return 0.0,0.0
    def _best_action(self,task,workload,arrival,remaining,price_t,price_e):
        actions=self._completion_actions(task,workload,arrival[2],arrival[3],0.0,0.0)
        return (actions[0],float(self.weights[task])) if actions else (self._skip_action(),0.0)
