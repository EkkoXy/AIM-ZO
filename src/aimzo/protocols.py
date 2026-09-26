from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


FRESH_ROLLOUT_PROTOCOL = "regenerate_rollout_per_perturbation"
FIXED_ROLLOUT_PROTOCOL = "fixed_rollout_per_step"
ZOREGULAR_SUPERVISED_PROTOCOL = "zoregular_supervised_ce"
ZOREGULAR_SUPERVISED_OBJECTIVES = frozenset(
    {
        "source_sst2_candidate_scoring",
        "zoregular_classification_ce",
        "zoregular_generation_ce",
    }
)
SUPPORTED_OBJECTIVES = frozenset(
    {
        "reward",
        "policy_loss",
        "grpo_like_without_kl",
        "grpo_with_kl",
        "grpo_surrogate_without_kl",
        "grpo_surrogate_with_kl",
        *ZOREGULAR_SUPERVISED_OBJECTIVES,
    }
)
SUPPORTED_ROLLOUT_PROTOCOLS = frozenset(
    {FRESH_ROLLOUT_PROTOCOL, FIXED_ROLLOUT_PROTOCOL}
)
FIXED_ROLLOUT_OBJECTIVES = frozenset(
    {
        "policy_loss",
        "grpo_like_without_kl",
        "grpo_with_kl",
        "grpo_surrogate_without_kl",
        "grpo_surrogate_with_kl",
    }
)
SUPPORTED_LOSS_AGGREGATIONS = frozenset(
    {
        "sequence_sum_then_sample_mean",
        "sequence_mean",
        "group_mean",
        "token_mean_then_sample_mean",
    }
)
SUPPORTED_REFERENCE_POLICIES = frozenset(
    {"initial_adapter", "center_at_step_start"}
)
SUPPORTED_LOGPROB_GRANULARITIES = frozenset({"sequence", "token"})


@dataclass(frozen=True)
class ObjectiveProtocol:
    objective_name: str
    name: str
    rollout_policy: str
    fixed_rollout: bool
    regenerates_rollout: bool
    requires_logprobs: bool
    old_logprob_policy: str
    reference_policy: str
    advantage_policy: str
    aggregation: str
    include_kl: bool
    objective_normalization: str
    logprob_granularity: str = "sequence"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_objective_protocol(
    objective_name: str,
    rollout_protocol: str,
    *,
    loss_aggregation: str = "sequence_sum_then_sample_mean",
    reference_policy: str = "initial_adapter",
    logprob_granularity: str = "sequence",
) -> ObjectiveProtocol:
    _validate_names(objective_name=objective_name, rollout_protocol=rollout_protocol)
    _validate_loss_aggregation(loss_aggregation)
    _validate_reference_policy(reference_policy)
    _validate_logprob_granularity(logprob_granularity)
    if objective_name in ZOREGULAR_SUPERVISED_OBJECTIVES:
        return ObjectiveProtocol(
            objective_name=objective_name,
            name=ZOREGULAR_SUPERVISED_PROTOCOL,
            rollout_policy="fixed_supervised_batch",
            fixed_rollout=True,
            regenerates_rollout=False,
            requires_logprobs=True,
            old_logprob_policy="unused",
            reference_policy="unused",
            advantage_policy="unused",
            aggregation="sample_mean",
            include_kl=False,
            objective_normalization="none",
            logprob_granularity="token",
        )
    if objective_name == "reward":
        return ObjectiveProtocol(
            objective_name=objective_name,
            name=rollout_protocol,
            rollout_policy="evaluated_policy_per_perturbation",
            fixed_rollout=False,
            regenerates_rollout=True,
            requires_logprobs=False,
            old_logprob_policy="unused",
            reference_policy="unused",
            advantage_policy="unused",
            aggregation="mean_reward",
            include_kl=False,
            objective_normalization="none",
            logprob_granularity=logprob_granularity,
        )
    if objective_name == "policy_loss":
        return ObjectiveProtocol(
            objective_name=objective_name,
            name=rollout_protocol,
            rollout_policy="step_start_policy",
            fixed_rollout=True,
            regenerates_rollout=False,
            requires_logprobs=True,
            old_logprob_policy="unused",
            reference_policy="unused",
            advantage_policy="fixed_from_step_start_rewards",
            aggregation=_metadata_aggregation(loss_aggregation),
            include_kl=False,
            objective_normalization="none",
            logprob_granularity=logprob_granularity,
        )
    if objective_name == "grpo_with_kl":
        return ObjectiveProtocol(
            objective_name=objective_name,
            name=rollout_protocol,
            rollout_policy="step_start_policy",
            fixed_rollout=True,
            regenerates_rollout=False,
            requires_logprobs=True,
            old_logprob_policy="center_policy",
            reference_policy=reference_policy,
            advantage_policy="group_reward_normalized",
            aggregation=loss_aggregation,
            include_kl=True,
            objective_normalization="group_reward_normalized_advantages",
            logprob_granularity=logprob_granularity,
        )
    if objective_name in {"grpo_surrogate_without_kl", "grpo_surrogate_with_kl"}:
        if logprob_granularity != "token":
            raise ValueError(
                f"{objective_name} requires objective.logprob_granularity='token'"
            )
        if loss_aggregation != "token_mean_then_sample_mean":
            raise ValueError(
                f"{objective_name} requires "
                "objective.loss_aggregation='token_mean_then_sample_mean'"
            )
        return ObjectiveProtocol(
            objective_name=objective_name,
            name=rollout_protocol,
            rollout_policy="step_start_policy",
            fixed_rollout=True,
            regenerates_rollout=False,
            requires_logprobs=True,
            old_logprob_policy="center_policy",
            reference_policy=(
                reference_policy
                if objective_name == "grpo_surrogate_with_kl"
                else "unused"
            ),
            advantage_policy="group_reward_normalized",
            aggregation=loss_aggregation,
            include_kl=objective_name == "grpo_surrogate_with_kl",
            objective_normalization="group_reward_normalized_advantages",
            logprob_granularity=logprob_granularity,
        )
    return ObjectiveProtocol(
        objective_name=objective_name,
        name=rollout_protocol,
        rollout_policy="step_start_policy",
        fixed_rollout=True,
        regenerates_rollout=False,
        requires_logprobs=True,
        old_logprob_policy="unused",
        reference_policy="unused",
        advantage_policy="group_reward_normalized",
        aggregation=_metadata_aggregation(loss_aggregation),
        include_kl=False,
        objective_normalization="group_reward_normalized_advantages",
        logprob_granularity=logprob_granularity,
    )


def validate_objective_protocol(objective: Any, protocol: ObjectiveProtocol) -> None:
    objective_name = getattr(objective, "name", None)
    if objective_name != protocol.objective_name:
        raise ValueError(
            "objective name does not match protocol: "
            f"objective={objective_name!r}, protocol={protocol.objective_name!r}"
        )
    requires_fresh_rollout = bool(getattr(objective, "requires_fresh_rollout", False))
    if requires_fresh_rollout != protocol.regenerates_rollout:
        raise ValueError(
            "objective fresh-rollout requirement does not match protocol: "
            f"objective.requires_fresh_rollout={requires_fresh_rollout}, "
            f"protocol.regenerates_rollout={protocol.regenerates_rollout}"
        )
    requires_logprobs = bool(getattr(objective, "requires_logprobs", False))
    if requires_logprobs != protocol.requires_logprobs:
        raise ValueError(
            "objective logprob requirement does not match protocol: "
            f"objective.requires_logprobs={requires_logprobs}, "
            f"protocol.requires_logprobs={protocol.requires_logprobs}"
        )
    logprob_granularity = getattr(objective, "logprob_granularity", None)
    if logprob_granularity is not None:
        if str(logprob_granularity) != protocol.logprob_granularity:
            raise ValueError(
                "objective logprob_granularity does not match protocol: "
                f"objective.logprob_granularity={logprob_granularity!r}, "
                f"protocol.logprob_granularity={protocol.logprob_granularity!r}"
            )
    loss_aggregation = getattr(objective, "loss_aggregation", None)
    if loss_aggregation is not None:
        expected_aggregation = (
            str(loss_aggregation)
            if protocol.objective_name in {
                "grpo_with_kl",
                "grpo_surrogate_without_kl",
                "grpo_surrogate_with_kl",
            }
            else _metadata_aggregation(str(loss_aggregation))
        )
        if expected_aggregation != protocol.aggregation:
            raise ValueError(
                "objective loss aggregation does not match protocol: "
                f"objective.loss_aggregation={loss_aggregation!r}, "
                f"protocol.aggregation={protocol.aggregation!r}"
            )
    reference_policy = getattr(objective, "reference_policy", None)
    if reference_policy is not None and protocol.include_kl:
        if str(reference_policy) != protocol.reference_policy:
            raise ValueError(
                "objective reference policy does not match protocol: "
                f"objective.reference_policy={reference_policy!r}, "
                f"protocol.reference_policy={protocol.reference_policy!r}"
            )


def _validate_names(*, objective_name: str, rollout_protocol: str) -> None:
    if objective_name not in SUPPORTED_OBJECTIVES:
        supported = ", ".join(sorted(SUPPORTED_OBJECTIVES))
        raise ValueError(
            f"unknown objective {objective_name!r}; expected one of: {supported}"
        )
    if objective_name in ZOREGULAR_SUPERVISED_OBJECTIVES:
        return
    if rollout_protocol not in SUPPORTED_ROLLOUT_PROTOCOLS:
        supported = ", ".join(sorted(SUPPORTED_ROLLOUT_PROTOCOLS))
        raise ValueError(
            f"unknown rollout protocol {rollout_protocol!r}; expected one of: {supported}"
        )
    if objective_name == "reward" and rollout_protocol != FRESH_ROLLOUT_PROTOCOL:
        raise ValueError(
            "reward objective requires fresh/regenerated rollouts; "
            "fixed_rollout_per_step would make the reward objective constant"
        )
    if (
        objective_name in FIXED_ROLLOUT_OBJECTIVES
        and rollout_protocol != FIXED_ROLLOUT_PROTOCOL
    ):
        raise ValueError(
            f"{objective_name} objective currently only supports {FIXED_ROLLOUT_PROTOCOL}"
        )


def _validate_loss_aggregation(value: str) -> None:
    if value not in SUPPORTED_LOSS_AGGREGATIONS:
        supported = ", ".join(sorted(SUPPORTED_LOSS_AGGREGATIONS))
        raise ValueError(f"loss_aggregation must be one of: {supported}")


def _validate_reference_policy(value: str) -> None:
    if value not in SUPPORTED_REFERENCE_POLICIES:
        supported = ", ".join(sorted(SUPPORTED_REFERENCE_POLICIES))
        raise ValueError(f"reference_policy must be one of: {supported}")


def _validate_logprob_granularity(value: str) -> None:
    if value not in SUPPORTED_LOGPROB_GRANULARITIES:
        supported = ", ".join(sorted(SUPPORTED_LOGPROB_GRANULARITIES))
        raise ValueError(f"logprob_granularity must be one of: {supported}")


def _metadata_aggregation(loss_aggregation: str) -> str:
    if loss_aggregation == "sequence_sum_then_sample_mean":
        return "sequence_logprob_sum_then_sample_mean"
    return loss_aggregation
