from typing import Union

from pydantic import Field

from cogentic.orchestration.models.base import CogenticBaseModel


class CogenticQuestionEvidence(CogenticBaseModel):
    """Evidence for the cogentic system."""

    description: str = Field(
        description="Description of the evidence, including it's relevance to the question"
    )
    content: str = Field(description="Relevant content from the source")


class CogenticTestEvidence(CogenticBaseModel):
    """Evidence for the cogentic system."""

    test_name: str = Field(description="Name of the test which produced this evidence")
    team_member_name: str = Field(
        description="Name of the team member who produced this evidence during their work on the test"
    )
    content: str = Field(description="The evidence itself")


CogenticEvidence = Union[CogenticQuestionEvidence, CogenticTestEvidence]


class CogenticEvidenceContainer(CogenticBaseModel):
    """Container for evidence."""

    entries: list[CogenticEvidence] = Field(
        description="List of evidence collected from the question or test",
    )


class CogenticInitialEvidence(CogenticBaseModel):
    """Initial evidence for the cogentic system."""

    evidence: list[CogenticQuestionEvidence] = Field(
        description="List of evidence collected from the question",
    )
