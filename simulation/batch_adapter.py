"""Load the realistic batch into the three isolated algorithm snapshots.

The loader treats the batch files as data only.  One canonical task table is
copied into three native ``Mission`` classes so package specific planners can
run in the same process without importing their conflicting ``common`` names.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, fields
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np


FUSION_SRC = Path(__file__).resolve().parent
if str(FUSION_SRC) not in sys.path:
    sys.path.insert(0, str(FUSION_SRC))

from vendor.optimized.common import config as optimized_config  # noqa: E402
from vendor.optimized.common import mte_core as optimized_core  # noqa: E402
from vendor.optimized.common import policy_runner as optimized_policy  # noqa: E402
from vendor.optimized.common import simulator as optimized_sim  # noqa: E402
from vendor.iteration.common import config as iteration_config  # noqa: E402
from vendor.iteration.common import mte_core as iteration_core  # noqa: E402
from vendor.iteration.common import mte_v2 as iteration_v2  # noqa: E402
from vendor.iteration.common import policy_runner as iteration_policy  # noqa: E402
from vendor.iteration.common import simulator as iteration_sim  # noqa: E402
from vendor.robust.common import config as robust_config  # noqa: E402
from vendor.robust.common import mte_core as robust_core  # noqa: E402
from vendor.robust.common import rollout_core as robust_rollout_core  # noqa: E402
from vendor.robust.common import rollout_mean_core as robust_rollout_mean_core  # noqa: E402
from vendor.robust.common import rollout_policy as robust_policy  # noqa: E402
from vendor.robust.common import simulator as robust_sim  # noqa: E402

import root_calibrated_core  # noqa: E402


@dataclass
class PlannerBundle:
    """Common mission objects and precomputed arrays for one case and seed."""

    optimized: Any
    iteration: Any
    robust: Any
    optimized_arrays: tuple
    iteration_arrays: tuple
    robust_arrays: tuple
    optimized_points: np.ndarray
    optimized_probs: np.ndarray
    iteration_points: np.ndarray
    iteration_probs: np.ndarray
    robust_points: np.ndarray
    robust_probs: np.ndarray
    robust_scenarios: np.ndarray
    tnse_parameters: dict[str, Any] = field(default_factory=dict)
    bloom_parameters: dict[str, Any] = field(default_factory=dict)
    algorithm_overrides: dict[str, Any] = field(default_factory=dict)


def scenario_digest(mission: Any) -> str:
    """Hash a mission in a representation independent form for adapter audits."""

    tasks = list(mission.tasks)
    task_fields: dict[str, Any] = {
        "task_xy": np.asarray([task.xy for task in tasks], dtype=float),
        "priorities": np.asarray([task.weight for task in tasks], dtype=float),
        "deadlines_s": np.asarray([task.deadline_s for task in tasks], dtype=float),
        "workload_mu": np.asarray([task.mu for task in tasks], dtype=float),
        "workload_sigma": np.asarray([task.sigma for task in tasks], dtype=float),
        "true_log_workload": np.asarray(
            [math.log(max(float(task.workload_gcy), 1e-300)) for task in tasks],
            dtype=float,
        ),
        "uplink_rate_mbps": np.asarray([task.ul_rates for task in tasks], dtype=float),
        "downlink_rate_mbps": np.asarray([task.dl_rates for task in tasks], dtype=float),
    }
    digest = hashlib.sha256()
    for name in (
        "task_xy",
        "mec_xy",
        "base_xy",
        "priorities",
        "deadlines_s",
        "workload_mu",
        "workload_sigma",
        "true_log_workload",
        "uplink_rate_mbps",
        "downlink_rate_mbps",
        "flight_time",
        "flight_energy_kj",
    ):
        if name in task_fields:
            value = task_fields[name]
        elif hasattr(mission, name):
            value = getattr(mission, name)
        else:
            continue
        array = np.asarray(value)
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def scenario_workload_bank(
    mu: np.ndarray, sigma: np.ndarray, *, count: int, seed: int
) -> np.ndarray:
    """Build a deterministic Latin hypercube workload bank."""

    from statistics import NormalDist

    n = max(2, int(count))
    rng = np.random.default_rng(int(seed))
    base = np.asarray(
        [NormalDist().inv_cdf((index + 0.5) / n) for index in range(n)],
        dtype=float,
    )
    result = np.zeros((n, len(mu)), dtype=float)
    for task in range(1, len(mu)):
        result[:, task] = np.exp(mu[task] + sigma[task] * base[rng.permutation(n)])
    return result


def visited_mask(state: Any) -> np.uint64:
    mask = np.uint64(0)
    for index in state.visited:
        mask |= np.uint64(1) << np.uint64(int(index) - 1)
    return mask


def optimized_scores(
    bundle: PlannerBundle, state: Any, mask: np.uint64
) -> tuple[int, np.ndarray, int]:
    """Return the frozen optimized planner choice and its native score vector."""

    mission = bundle.optimized
    weights, deadline, _, _, pars, aa, bb, dd, ee = bundle.optimized_arrays
    result = optimized_core.top_choose(
        int(state.node),
        float(state.time_s),
        float(state.energy_kj),
        mask,
        bundle.optimized_points,
        bundle.optimized_probs,
        mission.flight_time,
        mission.flight_energy_kj,
        weights,
        deadline,
        pars,
        aa,
        bb,
        dd,
        ee,
        0,
        int(mission.cfg.beam_width),
        True,
        True,
    )
    return int(result[0]), np.asarray(result[5], dtype=float), int(result[4])


def iteration_choice(
    bundle: PlannerBundle, state: Any, mask: np.uint64
) -> tuple[int, float, int]:
    """Return the frozen scenario conditioned planner choice."""

    mission = bundle.iteration
    weights, deadline, _, _, pars, aa, bb, dd, ee = bundle.iteration_arrays
    started = perf_counter()
    result = iteration_v2.top_choose_adaptive_v2(
        int(state.node),
        float(state.time_s),
        float(state.energy_kj),
        mask,
        bundle.iteration_points,
        bundle.iteration_probs,
        mission.flight_time,
        mission.flight_energy_kj,
        weights,
        deadline,
        pars,
        aa,
        bb,
        dd,
        ee,
        0,
        1.0,
        1.0,
        0.0,
        int(mission.cfg.beam_width),
        True,
        0.5,
        True,
        0.0,
        0.0,
    )
    return int(result[0]), float(result[1]), int((perf_counter() - started) * 1e6)


def root_calibrated_iteration_choice(
    bundle: PlannerBundle,
    state: Any,
    mask: np.uint64,
    *,
    calibration_mode: int,
) -> tuple[int, float, int]:
    """Return the AR route choice with one analytic root mass correction."""

    mission = bundle.iteration
    weights, deadline, mu, sigma, pars, aa, bb, dd, ee = bundle.iteration_arrays
    result = root_calibrated_core.top_choose_root_calibrated_v2(
        int(state.node),
        float(state.time_s),
        float(state.energy_kj),
        mask,
        bundle.iteration_points,
        bundle.iteration_probs,
        mu,
        sigma,
        mission.flight_time,
        mission.flight_energy_kj,
        weights,
        deadline,
        pars,
        aa,
        bb,
        dd,
        ee,
        int(mission.cfg.beam_width),
        0.5,
        int(calibration_mode),
    )
    return int(result[0]), float(result[1]), int(result[2])


def action_consistent_iteration_choice(
    bundle: PlannerBundle,
    state: Any,
    mask: np.uint64,
    *,
    quadrature_order: int = 0,
    support_rule: str = "quantile",
    beam_width: int = 0,
    lp_weight: float = 0.5,
    dual_price_local: bool = False,
    dual_action_rule: str = "none",
    analytic_root_mass: bool = False,
    return_prices: bool = False,
) -> tuple[int, float, int] | tuple[int, float, int, float, float]:
    """Return one action consistent policy improvement of the AR route."""

    mission = bundle.iteration
    weights, deadline, mu, sigma, pars, aa, bb, dd, ee = bundle.iteration_arrays
    if support_rule not in {"quantile", "gauss_hermite"}:
        raise ValueError(f"unknown workload support rule: {support_rule}")
    if int(quadrature_order) > 0 and support_rule == "quantile":
        points, probabilities = iteration_v2.quantile_workload_points(
            mu, sigma, int(quadrature_order)
        )
    elif int(quadrature_order) > 0:
        points, probabilities = optimized_core.gh_workload_points(
            mu, sigma, int(quadrature_order)
        )
    else:
        points, probabilities = bundle.iteration_points, bundle.iteration_probs
    width = int(beam_width) if int(beam_width) > 0 else int(mission.cfg.beam_width)
    action_modes = {
        "none": 0,
        "reduced_cost": 1,
        "lexicographic": 2,
        "mode_separated": 3,
        "pareto_separated": 4,
        "mode_separated_cached": 5,
        "mode_separated_full_cache": 5,
    }
    if dual_action_rule not in action_modes:
        raise ValueError(f"unknown dual action rule: {dual_action_rule}")
    result = root_calibrated_core.top_choose_action_consistent_v2(
        int(state.node),
        float(state.time_s),
        float(state.energy_kj),
        mask,
        points,
        probabilities,
        mu,
        sigma,
        mission.flight_time,
        mission.flight_energy_kj,
        weights,
        deadline,
        pars,
        aa,
        bb,
        dd,
        ee,
        width,
        float(lp_weight),
        int(bool(dual_price_local)),
        int(action_modes[dual_action_rule]),
        int(bool(analytic_root_mass)),
    )
    if return_prices:
        return (
            int(result[0]),
            float(result[1]),
            int(result[2]),
            float(result[3]),
            float(result[4]),
        )
    return int(result[0]), float(result[1]), int(result[2])


def state_lp_dual_prices(
    bundle: PlannerBundle,
    *,
    node: int,
    time_s: float,
    energy_kj: float,
    remaining_mask: int,
    quadrature_order: int = 0,
    support_rule: str = "quantile",
) -> tuple[float, float, float]:
    """Return state dependent time and energy prices for realized recourse."""

    mission = bundle.iteration
    weights, deadline, mu, sigma, pars, aa, bb, dd, ee = bundle.iteration_arrays
    if support_rule not in {"quantile", "gauss_hermite"}:
        raise ValueError(f"unknown workload support rule: {support_rule}")
    if int(quadrature_order) > 0 and support_rule == "quantile":
        points, probabilities = iteration_v2.quantile_workload_points(
            mu, sigma, int(quadrature_order)
        )
    elif int(quadrature_order) > 0:
        points, probabilities = optimized_core.gh_workload_points(
            mu, sigma, int(quadrature_order)
        )
    else:
        points, probabilities = bundle.iteration_points, bundle.iteration_probs
    result = root_calibrated_core.state_lp_dual_prices_v2(
        int(node),
        float(time_s),
        float(energy_kj),
        np.uint64(remaining_mask),
        points,
        probabilities,
        mission.flight_time,
        mission.flight_energy_kj,
        weights,
        deadline,
        pars,
        aa,
        bb,
        dd,
        ee,
    )
    return float(result[0]), float(result[1]), float(result[2])


def threshold_stratified_action_consistent_choice(
    bundle: PlannerBundle,
    state: Any,
    mask: np.uint64,
    *,
    support_budget: int,
    beam_width: int,
    lp_weight: float,
    max_depth: int = 0,
) -> tuple[int, float, int]:
    """Return the analytical threshold stratified ACAR route choice."""

    mission = bundle.iteration
    weights, deadline, mu, sigma, pars, aa, bb, dd, ee = bundle.iteration_arrays
    width = int(beam_width) if int(beam_width) > 0 else int(mission.cfg.beam_width)
    result = root_calibrated_core.top_choose_threshold_stratified_action_consistent_v2(
        int(state.node),
        float(state.time_s),
        float(state.energy_kj),
        mask,
        mu,
        sigma,
        mission.flight_time,
        mission.flight_energy_kj,
        weights,
        deadline,
        pars,
        aa,
        bb,
        dd,
        ee,
        max(2, int(support_budget)),
        width,
        float(lp_weight),
        int(max_depth),
    )
    return int(result[0]), float(result[1]), int(result[2])


def threshold_bellman_action_consistent_choice(
    bundle: PlannerBundle,
    state: Any,
    mask: np.uint64,
    *,
    support_budget: int,
    lp_weight: float,
    continuation_depth: int,
) -> tuple[int, float, int]:
    """Return the finite-depth threshold Bellman route choice."""

    mission = bundle.iteration
    weights, deadline, mu, sigma, pars, aa, bb, dd, ee = bundle.iteration_arrays
    result = root_calibrated_core.top_choose_threshold_bellman_v2(
        int(state.node),
        float(state.time_s),
        float(state.energy_kj),
        mask,
        mu,
        sigma,
        mission.flight_time,
        mission.flight_energy_kj,
        weights,
        deadline,
        pars,
        aa,
        bb,
        dd,
        ee,
        max(2, int(support_budget)),
        float(lp_weight),
        max(0, int(continuation_depth)),
    )
    return int(result[0]), float(result[1]), int(result[2])


def threshold_bellman_action_consistent_scores(
    bundle: PlannerBundle,
    state: Any,
    mask: np.uint64,
    *,
    support_budget: int,
    lp_weight: float,
    continuation_depth: int,
) -> tuple[int, float, int, np.ndarray]:
    """Return the branch-preserving Bellman score of every root task."""

    mission = bundle.iteration
    weights, deadline, mu, sigma, pars, aa, bb, dd, ee = bundle.iteration_arrays
    result = root_calibrated_core.top_choose_threshold_bellman_v2(
        int(state.node),
        float(state.time_s),
        float(state.energy_kj),
        mask,
        mu,
        sigma,
        mission.flight_time,
        mission.flight_energy_kj,
        weights,
        deadline,
        pars,
        aa,
        bb,
        dd,
        ee,
        max(2, int(support_budget)),
        float(lp_weight),
        max(0, int(continuation_depth)),
    )
    return (
        int(result[0]),
        float(result[1]),
        int(result[2]),
        np.asarray(result[3], dtype=float),
    )


def threshold_bellman_future_value(
    bundle: PlannerBundle,
    *,
    node: int,
    time_s: float,
    energy_kj: float,
    remaining_mask: int,
    support_budget: int,
    lp_weight: float,
    continuation_depth: int,
) -> tuple[float, int]:
    """Evaluate the common finite-depth continuation from a reached state."""

    mission = bundle.iteration
    weights, deadline, mu, sigma, pars, aa, bb, dd, ee = bundle.iteration_arrays
    result = root_calibrated_core._threshold_bellman_state_value(
        int(node),
        float(time_s),
        float(energy_kj),
        np.uint64(remaining_mask),
        mu,
        sigma,
        mission.flight_time,
        mission.flight_energy_kj,
        weights,
        deadline,
        pars,
        aa,
        bb,
        dd,
        ee,
        max(2, int(support_budget)),
        float(lp_weight),
        max(0, int(continuation_depth)),
    )
    return float(result[0]), int(result[1])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _config(config_type: type, physics: dict[str, Any], changes: dict[str, Any]) -> Any:
    available = {field.name for field in fields(config_type)}
    values: dict[str, Any] = {}
    for key, value in physics.items():
        if key in available:
            values[key] = value
    if "eta_los" in available and "eta_los_db" in physics:
        values["eta_los"] = 10.0 ** (float(physics["eta_los_db"]) / 10.0)
    if "eta_nlos" in available and "eta_nlos_db" in physics:
        values["eta_nlos"] = 10.0 ** (float(physics["eta_nlos_db"]) / 10.0)
    if "mec_cpu_ghz" in available:
        values["mec_cpu_ghz"] = tuple(float(value) for value in physics["mec_cpu_ghz"])
    for key, value in changes.items():
        if key in available:
            values[key] = value
    return config_type(**values)


class BatchCase:
    """Cached reader for all missions in one generated case directory."""

    def __init__(self, case_dir: str | Path) -> None:
        self.case_dir = Path(case_dir).resolve()
        self.payload = json.loads((self.case_dir / "config.json").read_text(encoding="utf-8"))
        self.physics = dict(self.payload["physics"])
        self.algorithm_parameters = dict(self.payload.get("algorithm_parameters", {}))
        self.algorithm_overrides = dict(self.payload.get("algorithm_overrides", {}))

        rows = _read_csv(self.case_dir / "tasks.csv")
        self.tasks_by_seed: dict[int, list[dict[str, str]]] = {}
        for row in rows:
            self.tasks_by_seed.setdefault(int(row["seed"]), []).append(row)
        for task_rows in self.tasks_by_seed.values():
            task_rows.sort(key=lambda row: int(row["task_id"]))

        link_rows = _read_csv(self.case_dir / "links.csv")
        self.links_by_task: dict[int, list[dict[str, str]]] = {}
        for row in link_rows:
            self.links_by_task.setdefault(int(row["task_id"]), []).append(row)
        for task_links in self.links_by_task.values():
            task_links.sort(key=lambda row: int(row["mec_id"]))

        nodes = _read_csv(self.case_dir / "nodes.csv")
        base = next(row for row in nodes if row["node_type"] == "base")
        mecs = sorted(
            (row for row in nodes if row["node_type"] == "mec"),
            key=lambda row: int(row["mec_id"]),
        )
        self.base_xy = np.asarray([float(base["x_m"]), float(base["y_m"])], dtype=float)
        self.mec_xy = np.asarray(
            [[float(row["x_m"]), float(row["y_m"])] for row in mecs], dtype=float
        )
        self.fingerprints = {
            int(row["seed"]): row["mission_fingerprint"]
            for row in _read_csv(self.case_dir / "mission_summary.csv")
        }

        mte = dict(self.algorithm_parameters.get("MTE-Optimized", {}))
        tnse = dict(self.algorithm_parameters.get("TNSE-SC-MTE", {}))
        bloom = dict(self.algorithm_parameters.get("BLOOM-Proposed-RCR", {}))
        self.optimized_cfg = _config(optimized_config.Config, self.physics, mte)
        iteration_changes = {
            "beam_width": int(tnse.get("beam_width", 4)),
            "gh_order": int(tnse.get("order", 9)),
            "task_deadline_cap_s": float(self.physics["t_max"])
            - float(self.physics.get("deadline_cap_margin_s", 90.0)),
        }
        self.iteration_cfg = _config(iteration_config.Config, self.physics, iteration_changes)
        robust_changes = {
            "sub_pressure_q": float(bloom.get("sub_pressure_q", 1.0)),
            "sub_bottleneck_beta": float(bloom.get("sub_bottleneck_beta", 0.5)),
            "beam_width": int(bloom.get("beam_width", 2)),
        }
        self.robust_cfg = _config(robust_config.Config, self.physics, robust_changes)
        self.tnse_parameters = tnse
        self.bloom_parameters = bloom

    @property
    def case_id(self) -> str:
        return str(self.payload["case_id"])

    @property
    def suite(self) -> str:
        return str(self.payload["suite"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted(self.tasks_by_seed))

    def _task(self, row: dict[str, str], sim: Any) -> Any:
        task_id = int(row["task_id"])
        links = self.links_by_task[task_id]
        return sim.Task(
            task_id,
            float(row["lon"]),
            float(row["lat"]),
            np.asarray([float(row["x_m"]), float(row["y_m"])], dtype=float),
            int(row["region"]),
            float(row["data_mbit"]),
            float(row["deadline_s"]),
            float(row["weight"]),
            float(row["mu"]),
            float(row["sigma"]),
            float(row["workload_gcy"]),
            float(row["output_ratio"]),
            np.asarray([float(link["ul_rate_mbps"]) for link in links], dtype=float),
            np.asarray([float(link["dl_rate_mbps"]) for link in links], dtype=float),
        )

    def _mission(self, seed: int, cfg: Any, sim: Any) -> Any:
        if seed not in self.tasks_by_seed:
            raise KeyError(f"seed {seed} is absent from {self.case_dir}")
        tasks = [self._task(row, sim) for row in self.tasks_by_seed[seed]]
        coordinates = np.vstack(
            [self.base_xy, np.asarray([task.xy for task in tasks], dtype=float)]
        )
        delta = coordinates[:, None, :] - coordinates[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        flight_time = distance / float(cfg.speed_mps)
        flight_energy = flight_time * float(cfg.flight_power_w) / 1000.0
        return sim.Mission(
            int(seed),
            cfg,
            tasks,
            self.base_xy.copy(),
            self.mec_xy.copy(),
            distance,
            flight_time,
            flight_energy,
        )

    def build_bundle(self, seed: int) -> PlannerBundle:
        optimized = self._mission(seed, self.optimized_cfg, optimized_sim)
        iteration = self._mission(seed, self.iteration_cfg, iteration_sim)
        robust = self._mission(seed, self.robust_cfg, robust_sim)
        digests = {
            scenario_digest(optimized),
            scenario_digest(iteration),
            scenario_digest(robust),
        }
        if len(digests) != 1:
            raise RuntimeError("native mission adapters do not have a common digest")

        optimized_arrays = optimized_core.prepare_arrays(optimized)
        iteration_arrays = iteration_core.prepare_arrays(iteration)
        robust_arrays = robust_core.prepare_arrays(robust)
        _, _, opt_mu, opt_sigma, *_ = optimized_arrays
        _, _, iter_mu, iter_sigma, *_ = iteration_arrays
        _, _, robust_mu, robust_sigma, *_ = robust_arrays
        opt_points, opt_probs = optimized_core.gh_workload_points(
            opt_mu, opt_sigma, int(optimized.cfg.gh_order)
        )
        order = int(self.tnse_parameters.get("order", 9))
        iter_points, iter_probs = iteration_v2.quantile_workload_points(
            iter_mu, iter_sigma, order
        )
        base_quadrature = int(self.bloom_parameters.get("base_quadrature", 3))
        robust_points, robust_probs = robust_core.quantile_workload_points(
            robust_mu, robust_sigma, base_quadrature
        )
        scenario_count = int(self.bloom_parameters.get("scenarios", 16))
        robust_scenarios = scenario_workload_bank(
            robust_mu,
            robust_sigma,
            count=scenario_count,
            seed=20260831 + 7919 * int(seed),
        )
        return PlannerBundle(
            optimized=optimized,
            iteration=iteration,
            robust=robust,
            optimized_arrays=optimized_arrays,
            iteration_arrays=iteration_arrays,
            robust_arrays=robust_arrays,
            optimized_points=opt_points,
            optimized_probs=np.asarray(opt_probs, dtype=float),
            iteration_points=iter_points,
            iteration_probs=np.asarray(iter_probs, dtype=float),
            robust_points=robust_points,
            robust_probs=np.asarray(robust_probs, dtype=float),
            robust_scenarios=robust_scenarios,
            tnse_parameters=dict(self.tnse_parameters),
            bloom_parameters=dict(self.bloom_parameters),
            algorithm_overrides=dict(self.algorithm_overrides),
        )

    def metadata(self, seed: int) -> dict[str, Any]:
        axis_value = self.payload.get("axis_value")
        factors = axis_value if isinstance(axis_value, dict) else {}
        return {
            "suite": self.suite,
            "case_id": self.case_id,
            "seed": int(seed),
            "mission_fingerprint": self.fingerprints[int(seed)],
            "axis": str(self.payload.get("axis", "")),
            "axis_value": json.dumps(
                axis_value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            "t_max_s": float(self.physics["t_max"]),
            "e_max_kj": float(self.physics["e_max_kj"]),
            "n_tasks": int(self.physics["n_tasks"]),
            "n_mec": int(self.physics["n_mec"]),
            "geometry_profile": str(
                factors.get(
                    "geometry", self.payload.get("geometry_profile", "")
                )
            ),
            "path_loss_L_profile": str(
                factors.get(
                    "L_profile", self.payload.get("path_loss_L_profile", "")
                )
            ),
            "communication_profile": str(factors.get("communication", "")),
            "local_mec_balance": str(factors.get("balance", "")),
            "uncertainty_scale": float(
                factors.get(
                    "uncertainty_scale", self.physics.get("uncertainty_scale", 1.0)
                )
            ),
            "load_scale": float(
                factors.get(
                    "load_scale", float(self.physics.get("workload_scale", 2.5)) / 2.5
                )
            ),
            "deadline_scale_factor": float(
                factors.get(
                    "deadline_scale", self.physics.get("deadline_scale", 1.0)
                )
            ),
            "lookahead_depth_L": int(
                self.algorithm_overrides.get("lookahead_depth_L", 0)
            ),
        }
