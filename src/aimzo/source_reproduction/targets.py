from __future__ import annotations

from dataclasses import dataclass


SOURCE_GAP_STATUSES = {"pending", "matched", "gap", "unresolved"}
PAPER_GAP_STATUSES = {"not_applicable", "pending", "matched", "gap", "unresolved"}

AGZO_SOURCE_TARGET_ACCURACY = 0.7786697247706422
AGZO_SOURCE_AUXILIARY_ACCURACY = 0.7775229215621948
AGZO_LONG_RUN_FINAL_ACCURACY = 0.6135321100917431
AGZO_PAPER_TARGET_ACCURACY = 0.778
AGZO_SOURCE_GAP_STATUS = "pending"
AGZO_PAPER_GAP_STATUS = "matched"


@dataclass(frozen=True)
class TargetValues:
    source_observed_accuracy: float | None
    source_auxiliary_accuracy: float | None
    paper_target_accuracy: float | None
    source_gap_status: str = "pending"
    paper_gap_status: str = "not_applicable"

    def __post_init__(self) -> None:
        if self.source_gap_status not in SOURCE_GAP_STATUSES:
            raise ValueError(
                f"source_gap_status must be one of {sorted(SOURCE_GAP_STATUSES)}"
            )
        if self.paper_gap_status not in PAPER_GAP_STATUSES:
            raise ValueError(
                f"paper_gap_status must be one of {sorted(PAPER_GAP_STATUSES)}"
            )

    @property
    def has_paper_target(self) -> bool:
        return self.paper_target_accuracy is not None
