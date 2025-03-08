from pathlib import Path

from pydantic import Field

from cogentic.orchestration.models.base import CogenticBaseModel


class CogenticArtifact(CogenticBaseModel):
    """Artifact for the cogentic system."""

    name: str = Field(description="Name of the artifact")
    description: str = Field(description="Description of the artifact")
    path: Path = Field(
        description="Path to the artifact. This should be a relative path to the current working directory."
    )
    test_name: str = Field(description="Name of the test which produced this artifact")
    team_member_name: str = Field(
        description="Name of the team member who produced this artifact"
    )
