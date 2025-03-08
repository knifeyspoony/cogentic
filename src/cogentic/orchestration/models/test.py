from typing import Literal

from pydantic import Field

from cogentic.orchestration.models.base import CogenticBaseModel

CogenticTestState = Literal["complete", "incomplete", "abandoned"]


class CogenticTestTeamMemberPlan(CogenticBaseModel):
    """Role of the team member in a test."""

    name: str = Field(description="Name of the team member")
    plan: str = Field(description="How we envision the team member solving the test")
    rationale: str = Field(
        description="Why we believe this team member will be able to perform the test, based on their description"
    )


class CogenticTest(CogenticBaseModel):
    """A test which is part of a hypothesis."""

    name: str = Field(description="Name of the test")
    description: str = Field(description="Description of the test")
    state: CogenticTestState = Field(description="State of the test")
    plan: list[CogenticTestTeamMemberPlan] = Field(
        description="Plan for the test. This should include a list of team members and how we envision them solving the test"
    )
    result_summary: str | None = Field(description="A summary of  the results")
