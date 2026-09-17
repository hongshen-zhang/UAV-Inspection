"""Strict configuration objects for the consistent ACAR redesign.

The central design rule is that one :class:`ContinuationSpec` instance is
owned by one planner and is used for both pre-arrival route evaluation and
post-arrival recourse.  There is no second route-side guidance constant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


GUIDANCE_RULES = {"blended", "reward_only", "lp_upper"}
SUPPORT_RULES = {"gh", "quantile", "mean"}
ACTION_RULES = {"full_actions", "mode_separated"}
CONTINUATION_EVALUATORS = {"calibrated_beam", "executable_rollout"}
ANCHOR_RULES = {"none", "conditioned"}


@dataclass(frozen=True)
class GuidanceSpec:
    """Define the single Beam guidance rule used everywhere.

    ``reward_only`` has effective LP weight zero. ``lp_upper`` uses the LP
    opportunity value in its native reward unit and therefore has effective
    weight one. Only ``blended`` accepts an explicit empirical weight.
    """

    rule: str
    weight: float | None = None

    def __post_init__(self) -> None:
        if self.rule not in GUIDANCE_RULES:
            raise ValueError(f"unknown guidance rule: {self.rule}")
        if self.rule == "blended":
            if self.weight is None:
                raise ValueError("blended guidance requires one explicit weight")
            if not 0.0 <= float(self.weight) <= 1.0:
                raise ValueError("blended guidance weight must lie in [0, 1]")
        elif self.weight is not None:
            raise ValueError(
                f"{self.rule} guidance has a fixed mathematical weight; "
                "do not provide a second weight"
            )

    @property
    def effective_weight(self) -> float:
        if self.rule == "reward_only":
            return 0.0
        if self.rule == "lp_upper":
            return 1.0
        assert self.weight is not None
        return float(self.weight)


@dataclass(frozen=True)
class ContinuationSpec:
    """One immutable continuation definition shared by both decision stages."""

    support_rule: str
    support_order: int
    beam_width: int
    lookahead_depth: int
    cache_expected_stats: bool
    guidance: GuidanceSpec
    analytic_mass_calibration: bool = False
    evaluator: str = "calibrated_beam"
    candidate_width: int = 0
    rollout_profiles: int = 1
    rollout_steps: int = 20
    base_cost_floor: float = 0.08
    base_cost_power: float = 1.0
    base_cluster_weight: float = 0.10
    base_deadline_weight: float = 0.0
    anchor_rule: str = "none"
    improvement_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.support_rule not in SUPPORT_RULES:
            raise ValueError(f"unknown support rule: {self.support_rule}")
        if int(self.support_order) <= 0:
            raise ValueError("support order must be positive")
        if self.support_rule == "mean" and int(self.support_order) != 1:
            raise ValueError("mean support requires support_order=1")
        if int(self.beam_width) <= 0:
            raise ValueError("beam width must be positive")
        if int(self.lookahead_depth) <= 0:
            raise ValueError("lookahead depth must be positive")
        if not isinstance(self.cache_expected_stats, bool):
            raise ValueError("cache_expected_stats must be true or false")
        if not isinstance(self.analytic_mass_calibration, bool):
            raise ValueError("analytic_mass_calibration must be true or false")
        if self.evaluator not in CONTINUATION_EVALUATORS:
            raise ValueError(f"unknown continuation evaluator: {self.evaluator}")
        if int(self.candidate_width) < 0:
            raise ValueError("candidate_width must be nonnegative")
        if int(self.rollout_profiles) <= 0:
            raise ValueError("rollout_profiles must be positive")
        if int(self.rollout_steps) <= 0:
            raise ValueError("rollout_steps must be positive")
        for name, value in (
            ("base_cost_floor", self.base_cost_floor),
            ("base_cost_power", self.base_cost_power),
            ("base_cluster_weight", self.base_cluster_weight),
            ("base_deadline_weight", self.base_deadline_weight),
            ("improvement_margin", self.improvement_margin),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.anchor_rule not in ANCHOR_RULES:
            raise ValueError(f"unknown anchor rule: {self.anchor_rule}")
        if self.anchor_rule != "none" and self.evaluator != "executable_rollout":
            raise ValueError(
                "a conservative anchor is only defined for executable rollout"
            )
        if self.evaluator == "executable_rollout":
            if self.support_rule != "quantile":
                raise ValueError(
                    "executable rollout requires equal probability quantile support"
                )
            if int(self.candidate_width) <= 0:
                raise ValueError(
                    "executable rollout requires a positive candidate_width"
                )
            if self.guidance.rule != "reward_only":
                raise ValueError(
                    "executable rollout has no LP guidance; use reward_only"
                )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PolicySpec:
    """Complete definition of one unified ACAR policy."""

    config_id: str
    method_key: str
    label: str
    action_rule: str
    continuation: ContinuationSpec

    def __post_init__(self) -> None:
        if not self.config_id or not self.method_key or not self.label:
            raise ValueError("policy identifiers must be nonempty")
        if self.action_rule not in ACTION_RULES:
            raise ValueError(f"unknown action rule: {self.action_rule}")


def policy_from_mapping(config_id: str, values: Mapping[str, Any]) -> PolicySpec:
    """Parse one policy and reject ambiguous or unused fields."""

    allowed = {"method_key", "label", "action_rule", "continuation"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{config_id} has unknown fields: {', '.join(unknown)}")

    continuation_values = dict(values["continuation"])
    allowed_continuation = {
        "support_rule",
        "support_order",
        "beam_width",
        "lookahead_depth",
        "cache_expected_stats",
        "analytic_mass_calibration",
        "evaluator",
        "candidate_width",
        "rollout_profiles",
        "rollout_steps",
        "base_cost_floor",
        "base_cost_power",
        "base_cluster_weight",
        "base_deadline_weight",
        "anchor_rule",
        "improvement_margin",
        "guidance",
    }
    unknown_continuation = sorted(set(continuation_values) - allowed_continuation)
    if unknown_continuation:
        raise ValueError(
            f"{config_id} has unknown continuation fields: "
            + ", ".join(unknown_continuation)
        )
    guidance_values = dict(continuation_values.pop("guidance"))
    unknown_guidance = sorted(set(guidance_values) - {"rule", "weight"})
    if unknown_guidance:
        raise ValueError(
            f"{config_id} has unknown guidance fields: "
            + ", ".join(unknown_guidance)
        )
    guidance = GuidanceSpec(**guidance_values)
    continuation = ContinuationSpec(guidance=guidance, **continuation_values)
    return PolicySpec(
        config_id=config_id,
        method_key=str(values["method_key"]),
        label=str(values["label"]),
        action_rule=str(values["action_rule"]),
        continuation=continuation,
    )


def load_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported consistent ACAR configuration schema")
    required = {
        "schema_version",
        "design_name",
        "status",
        "design_contract",
        "default_methods",
        "unified_methods",
        "legacy_methods",
        "control_methods",
    }
    allowed = required | {"formal_evaluation_lock"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError("unknown top-level configuration fields: " + ", ".join(unknown))
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("missing top-level configuration fields: " + ", ".join(missing))

    groups = {
        name: payload[name]
        for name in ("unified_methods", "legacy_methods", "control_methods")
    }
    if any(not isinstance(values, dict) for values in groups.values()):
        raise ValueError("all method groups must be JSON objects")
    seen: set[str] = set()
    for name, values in groups.items():
        overlap = sorted(seen & set(values))
        if overlap:
            raise ValueError(f"method IDs repeated before {name}: {', '.join(overlap)}")
        seen.update(values)

    defaults = payload["default_methods"]
    if not isinstance(defaults, list) or not defaults:
        raise ValueError("default_methods must be a nonempty list")
    if len(defaults) != len(set(defaults)):
        raise ValueError("default_methods contains a repeated method ID")
    missing_defaults = sorted(set(defaults) - seen)
    if missing_defaults:
        raise ValueError("unknown default methods: " + ", ".join(missing_defaults))

    contract = payload["design_contract"]
    if not isinstance(contract, dict):
        raise ValueError("design_contract must be a JSON object")
    required_contract = {
        "one_continuation_object_per_planner": True,
        "current_task_reward_coefficient": 1.0,
        "hidden_workload_access_before_arrival": False,
        "guidance_applies_only_to_lp_opportunity": True,
        "route_and_arrival_signature_must_match": True,
    }
    for key, expected in required_contract.items():
        if key not in contract or contract[key] != expected:
            raise ValueError(f"design contract violation for {key}")
    return payload


def load_unified_policies(path: str | Path) -> dict[str, PolicySpec]:
    payload = load_payload(path)
    return {
        config_id: policy_from_mapping(config_id, values)
        for config_id, values in payload["unified_methods"].items()
    }
