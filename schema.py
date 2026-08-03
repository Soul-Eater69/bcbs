from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EpicField = Literal["title", "description", "success_criteria"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceItem(StrictModel):
    epic_field: EpicField
    text: str = Field(
        min_length=1,
        description="Exact supporting text copied from the epic field.",
    )


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
    epic_id: str

    overall_stage_coverage: float = Field(
        ge=0.0,
        le=1.0,
    )

    coverage_by_stage_field: CoverageByStageField
    stage_usage_locations: StageUsageLocations
    evidence: EvidenceByStageField
