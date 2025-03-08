from typing import Literal

from pydantic import Field

from cogentic.orchestration.models.base import CogenticBaseModel


class CogenticIssue(CogenticBaseModel):
    """Issue for the cogentic system."""

    name: str = Field(description="Name of the issue")
    description: str = Field(description="Description of the issue")
    severity: Literal["low", "medium", "high"] = Field(
        description="Severity of the issue"
    )


class CogenticIssueContainer(CogenticBaseModel):
    """Container for issues."""

    entries: list[CogenticIssue] = Field(
        description="List of issues encountered during the work",
    )
