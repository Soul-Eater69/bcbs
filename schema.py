from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


EpicField = Literal["title", "description", "success_criteria"]


class StrictModel(BaseModel):
    """Reject unexpected fields returned by the evaluator."""

    model_config = ConfigDict(extra="forbid")


class EvidenceItem(StrictModel):
    epic_field: EpicField
    success_criteria_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Zero-based success-criterion index. Must be null for title "
            "and description evidence."
        ),
    )
    text: str = Field(
        min_length=1,
        description="Exact supporting text copied from the epic.",
    )

    @model_validator(mode="after")
    def validate_success_criteria_index(self) -> "EvidenceItem":
        if (
            self.epic_field == "success_criteria"
            and self.success_criteria_index is None
        ):
            raise ValueError(
                "success_criteria_index is required when "
                "epic_field is 'success_criteria'."
            )

        if (
            self.epic_field != "success_criteria"
            and self.success_criteria_index is not None
        ):
            raise ValueError(
                "success_criteria_index must be null for title "
                "or description evidence."
            )

        return self


class CoverageByStageField(StrictModel):
    stage_name: float = Field(ge=0.0, le=1.0)
    stage_description: float = Field(ge=0.0, le=1.0)
    entrance_criteria: float = Field(ge=0.0, le=1.0)
    exit_criteria: float = Field(ge=0.0, le=1.0)


class StageUsageLocations(StrictModel):
    title: bool
    description: bool
    success_criteria: bool


class EvidenceByStageField(StrictModel):
    stage_name: list[EvidenceItem] = Field(default_factory=list)
    stage_description: list[EvidenceItem] = Field(default_factory=list)
    entrance_criteria: list[EvidenceItem] = Field(default_factory=list)
    exit_criteria: list[EvidenceItem] = Field(default_factory=list)


class StageCoverageOutput(StrictModel):
    epic_id: str = Field(min_length=1)

    overall_stage_coverage: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Proportion of all available stage information reflected "
            "anywhere in the epic."
        ),
    )

    coverage_by_stage_field: CoverageByStageField
    stage_usage_locations: StageUsageLocations
    evidence: EvidenceByStageField

    @model_validator(mode="after")
    def validate_evidence_and_locations(self) -> "StageCoverageOutput":
        coverage = self.coverage_by_stage_field
        evidence = self.evidence

        field_pairs = [
            ("stage_name", coverage.stage_name, evidence.stage_name),
            (
                "stage_description",
                coverage.stage_description,
                evidence.stage_description,
            ),
            (
                "entrance_criteria",
                coverage.entrance_criteria,
                evidence.entrance_criteria,
            ),
            (
                "exit_criteria",
                coverage.exit_criteria,
                evidence.exit_criteria,
            ),
        ]

        # A positive field score must have evidence.
        # A zero field score must not have evidence.
        for field_name, score, items in field_pairs:
            if score > 0 and not items:
                raise ValueError(
                    f"{field_name} has positive coverage but no evidence."
                )

            if score == 0 and items:
                raise ValueError(
                    f"{field_name} has zero coverage but contains evidence."
                )

        all_evidence = (
            evidence.stage_name
            + evidence.stage_description
            + evidence.entrance_criteria
            + evidence.exit_criteria
        )

        detected_locations = {
            "title": any(x.epic_field == "title" for x in all_evidence),
            "description": any(
                x.epic_field == "description" for x in all_evidence
            ),
            "success_criteria": any(
                x.epic_field == "success_criteria" for x in all_evidence
            ),
        }

        supplied_locations = self.stage_usage_locations.model_dump()

        if detected_locations != supplied_locations:
            raise ValueError(
                "stage_usage_locations does not match the evidence locations. "
                f"Expected {detected_locations}."
            )

        return self
