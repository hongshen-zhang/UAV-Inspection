#!/usr/bin/env python3
"""Paired nominal Nearest Neighbor and EDF controls using the shared executor.

Only the route-ranking rule differs from Priority Greedy. No realized
workload is used before arrival. All frozen seeds are evaluated by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_baselines_formal import (
    HERE, BASE_CLASS, BASE_POLICY, BatchCase, HeuristicPlanner, planner,
)


class NearestNeighborPlanner(HeuristicPlanner):
    def select_task(self, state):
        self.stats["route_calls"] += 1
        plan = self._plan_state(state)
        candidates = [
            task for task in range(1, self.n_tasks + 1)
            if plan.remaining_mask & (1 << (task - 1))
            and self._arrival(plan, task) is not None
        ]
        best = min(
            candidates,
            key=lambda task: (float(self.mission.flight_time[plan.node, task]), task),
            default=0,
        )
        return int(best), {"selection_mode": "nearest_neighbor"}


class EarliestDeadlinePlanner(HeuristicPlanner):
    def select_task(self, state):
        self.stats["route_calls"] += 1
        plan = self._plan_state(state)
        candidates = [
            task for task in range(1, self.n_tasks + 1)
            if plan.remaining_mask & (1 << (task - 1))
            and self._arrival(plan, task) is not None
        ]
        best = min(
            candidates,
            key=lambda task: (
                float(self.deadline[task]),
                float(self.mission.flight_time[plan.node, task]), task,
            ),
            default=0,
        )
        return int(best), {"selection_mode": "earliest_deadline_first"}


