from typing import Literal

from pydantic import Field

from cogentic.orchestration.models.action import CogenticAction
from cogentic.orchestration.models.artifact import CogenticArtifact
from cogentic.orchestration.models.base import CogenticBaseModel
from cogentic.orchestration.models.evidence import CogenticEvidence
from cogentic.orchestration.models.hypothesis import CogenticHypothesis
from cogentic.orchestration.models.issue import CogenticIssue

CogenticPlanState = Literal[
    "in_progress",
    "completed",
    "failed",
]


class CogenticPlan(CogenticBaseModel):
    """Plan for the cogentic system."""

    state: CogenticPlanState = Field(
        description="State of the plan",
        default="in_progress",
    )
    hypotheses: list[CogenticHypothesis] = Field(
        description="Hypotheses to be tested", default_factory=list
    )
    artifacts: list[CogenticArtifact] = Field(
        description="Artifacts we have created as part of our work e.g., scripts, files, etc.",
        default_factory=list,
    )
    issues: list[CogenticIssue] = Field(
        description="Issues we have encountered as part of our work e.g., bugs, errors, etc.",
        default_factory=list,
    )
    evidence: list[CogenticEvidence] = Field(
        description="Evidence we have collected as part of our work e.g., sources, content, etc.",
        default_factory=list,
    )
    actions: list[CogenticAction] = Field(
        description="History of all actions taken and their outcomes as part of this plan.",
        default_factory=list,
    )

    @property
    def current_hypothesis(self) -> CogenticHypothesis | None:
        """Get the current hypothesis to be tested."""
        for hypothesis in self.hypotheses:
            if hypothesis.state == "unverified":
                return hypothesis
        return None
