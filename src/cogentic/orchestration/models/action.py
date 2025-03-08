from pydantic import Field

from cogentic.orchestration.models.base import CogenticBaseModel


class CogenticAction(CogenticBaseModel):
    goal: str = Field(
        description="Short summary of the goal of the action",
    )
    outcome: str = Field(
        description="Short summary of the outcome of the action",
    )
    test_name: str = Field(description="Name of the test this action was a part of")
    team_member_name: str = Field(
        description="Name of the team member who performed the action"
    )
