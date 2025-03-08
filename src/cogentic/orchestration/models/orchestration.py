from __future__ import annotations

from typing import Literal, Self, Type

from pydantic import Field, model_validator

from cogentic.orchestration.models.base import CogenticBaseModel
from cogentic.orchestration.models.hypothesis import (
    CogenticHypothesis,
    CogenticHypothesisState,
)
from cogentic.orchestration.models.plan import CogenticPlanState
from cogentic.orchestration.models.reasoning import (
    CogenticReasonedChoiceAnswer,
    CogenticReasonedStringAnswer,
)
from cogentic.orchestration.models.test import CogenticTest


class CogenticNextStep(CogenticBaseModel):
    """Next step for the cogentic system."""

    goal: CogenticReasonedStringAnswer = Field(
        description="What is the goal of our next action? Provide your reasoning along with your answer.",
    )
    next_speaker: CogenticReasonedChoiceAnswer = Field(
        description="Who are we communicating with? Provide your reasoning and select the team member to communicate with.",
    )
    instruction_or_question: CogenticReasonedStringAnswer = Field(
        description="What instruction or question are we giving this team member? Provide your reasoning, and phrase your answer as it you're speaking to them directly.",
    )

    @classmethod
    def with_speaker_choices(cls, choices: list[str]) -> Type["CogenticNextStep"]:
        """Create a new type from our class, where the next speaker is limited to a set of choices."""
        # Create the choice type with proper annotation
        next_speaker_type = CogenticReasonedChoiceAnswer[Literal[tuple(choices)]]

        return type(
            "CogenticNextStepWithSpeakerChoices",
            (cls,),
            {
                "__annotations__": {
                    "next_speaker": next_speaker_type,
                },
            },
        )


class CogenticFinalAnswer(CogenticBaseModel):
    """Final answer for the cogentic system."""

    result: str = Field(description="The result of our work")
    completed_by_team_members: bool = Field(
        description="Whether the answer was completed by team members or by yourself"
    )
    status: Literal["complete", "incomplete"] = Field(
        description="Whether we were able to fully answer the question"
    )
    failure_reason: str | None = Field(
        description="Reason for the status (if incomplete)"
    )


class CogenticHypothesisUpdate(CogenticBaseModel):
    hypothesis_state: CogenticReasonedChoiceAnswer[CogenticHypothesisState] = Field(
        description="The state of the hypothesis.",
    )
    new_tests: list[CogenticTest] = Field(
        description="New tests to be added to the hypothesis if you're marking the state as `unverified`",
    )

    @model_validator(mode="after")
    def validate_hypothesis_state(self) -> Self:
        """Validate the hypothesis state."""
        if self.hypothesis_state.answer == "unverified" and not self.new_tests:
            raise ValueError(
                "If you're marking the hypothesis as `unverified`, you must provide new tests."
            )
        return self


class CogenticPlanUpdate(CogenticBaseModel):
    """Plan update for the cogentic system."""

    plan_state: CogenticReasonedChoiceAnswer[CogenticPlanState] = Field(
        description="The state of the plan.",
    )
    new_hypotheses: list[CogenticHypothesis] = Field(
        description="New hypotheses to be added to the plan if you're marking the state as `in_progress`",
    )

    @model_validator(mode="after")
    def validate_plan_state(self) -> Self:
        """Validate the plan state."""
        if self.plan_state.answer == "in_progress" and not self.new_hypotheses:
            raise ValueError(
                "If you're marking the plan as `in_progress`, you must provide new hypotheses."
            )
        return self
