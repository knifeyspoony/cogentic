from typing import Self, Type

from pydantic import Field, model_validator

from cogentic.orchestration.models.base import CogenticBaseModel
from cogentic.orchestration.models.evidence import CogenticTestEvidence
from cogentic.orchestration.models.issue import CogenticIssue
from cogentic.orchestration.models.orchestration import CogenticNextStep
from cogentic.orchestration.models.reasoning import (
    CogenticReasonedBooleanAnswer,
    CogenticReasonedChoiceAnswer,
)
from cogentic.orchestration.models.test import CogenticTestState


class CogenticProgressLedger(CogenticBaseModel):
    """Progress ledger for the cogentic system."""

    original_question_answered: CogenticReasonedBooleanAnswer = Field(
        description="Is the original question fully answered? (True if complete, or False if the original question has yet to be SUCCESSFULLY and FULLY addressed)",
    )
    test_state: CogenticReasonedChoiceAnswer[CogenticTestState] = Field(
        description="What is the state of the current test? Have we completed it? Are we stuck, and need to abandon it? Or are we still working on it?",
    )
    stuck_in_loop: CogenticReasonedBooleanAnswer = Field(
        description="Are we in a loop where we keep repeating the same request without making progress? ",
    )
    forward_progress: CogenticReasonedBooleanAnswer = Field(
        description="Are we making forward progress? (True if just starting, or recent messages are adding value. False if recent messages show evidence of being stuck in a loop or if there is evidence of significant barriers to success such as the inability to read from a required file)",
    )
    new_test_evidence: list[CogenticTestEvidence] = Field(
        description="Include any new evidence from our work on the current test.",
    )
    new_issues: list[CogenticIssue] = Field(
        description="Include any new issues we have encountered.",
    )
    next_step: CogenticNextStep | None = Field(
        description="The next step to take. This can be null if we're marking the original question as answered.",
    )

    @classmethod
    def with_speakers(cls, choices: list[str]) -> Type["CogenticProgressLedger"]:
        """Create a new type from our class, where the next speaker is limited to a set of choices."""
        # Create the choice type with proper annotation
        next_step_type = CogenticNextStep.with_speaker_choices(choices)

        return type(
            "CogenticProgressLedgerWithSpeakers",
            (cls,),
            {
                "__annotations__": {"next_step": next_step_type | None},
            },
        )

    @model_validator(mode="after")
    def validate_next_step(self) -> Self:
        """Validate that the next step is not None if the original question is not answered."""
        if self.next_step is None and self.original_question_answered is False:
            raise ValueError(
                "If we're not marking the original question as answered, we must have a next step!",
            )
        return self
