from typing import Literal, Type

from autogen_agentchat.state import BaseGroupChatManagerState
from pydantic import BaseModel, Field


class CogenticBaseModel(BaseModel):
    def model_dump_markdown(self, indent: int = 2) -> str:
        return f"```json\n{self.model_dump_json(indent=indent)}\n```"


class CogenticEvidence(CogenticBaseModel):
    """Evidence for the cogentic system."""

    source: str = Field(description="Source of the evidence e.g., file name, URL")
    content: str = Field(description="Relevant content from the source")


class CogenticTest(CogenticBaseModel):
    """A test which is part of a hypothesis."""

    description: str = Field(description="Description of the test")
    completed: bool = Field(description="Whether the test has been completed")
    result: str | None = Field(description="Result of the test")
    supporting_evidence: list[CogenticEvidence] | None = Field(
        description="Supporting evidence for the test"
    )


class CogenticHypothesis(CogenticBaseModel):
    """Hypothesis for the cogentic system."""

    hypothesis: str = Field(description="Hypothesis to be tested")
    state: Literal["unverified", "verified", "unverifiable"] = Field(
        description="State of the hypothesis"
    )
    notes: str | None = Field(description="Notes about the hypothesis")
    tests: list[CogenticTest] = Field(description="Tests for the hypothesis")


class CogenticFact(CogenticBaseModel):
    """Fact for the cogentic system."""

    content: str = Field(description="Fact content")
    state: Literal["provided", "verified"] = Field(description="State of the fact")
    notes: str | None = Field(description="Any applicable notes")
    supporting_evidence: list[CogenticEvidence] = Field(
        description="Supporting evidence for the fact"
    )


class CogenticPlan(CogenticBaseModel):
    """Plan for the cogentic system."""

    hypotheses: list[CogenticHypothesis] = Field(description="Hypotheses to be tested")


class CogenticFactSheet(CogenticBaseModel):
    """Fact sheet for the cogentic system."""

    facts: list[CogenticFact] = Field(
        description="All known facts relevant to our task"
    )


class CogenticReasonedStringAnswer(CogenticBaseModel):
    """Reasoned answer for the progress ledger."""

    reason: str = Field(description="Reason for the answer")
    answer: str = Field(description="Answer to the question")


class CogenticReasonedBooleanAnswer(CogenticBaseModel):
    """Reasoned answer for the progress ledger."""

    reason: str = Field(description="Reason for the answer")
    answer: bool = Field(description="Answer to the question")


class CogenticReasonedChoiceAnswer(CogenticBaseModel):
    """Reasoned answer for the progress ledger."""

    reason: str = Field(description="Reason for the answer")
    answer: Literal[""] = Field(description="Answer to the question")


class CogenticProgressLedger(CogenticBaseModel):
    """Progress ledger for the cogentic system."""

    is_current_hypothesis_work_complete: CogenticReasonedBooleanAnswer = Field(
        description="Is the current hypothesis work complete? (True if the current hypothesis is verified, or unverifiable. False if the current hypothesis is unverified and there is still work to be done) ",
    )
    is_request_satisfied: CogenticReasonedBooleanAnswer = Field(
        description="Is the original question fully answered? (True if complete, or False if the original question has yet to be SUCCESSFULLY and FULLY addressed)",
    )
    is_in_loop: CogenticReasonedBooleanAnswer = Field(
        description="Are we in a loop where we are repeating the same requests and / or getting the same responses as before? Loops can span multiple turns, and can include repeated actions like scrolling up or down more than a handful of times.",
    )
    is_progress_being_made: CogenticReasonedBooleanAnswer = Field(
        description="Are we making forward progress? (True if just starting, or recent messages are adding value. False if recent messages show evidence of being stuck in a loop or if there is evidence of significant barriers to success such as the inability to read from a required file)",
    )
    next_speaker: CogenticReasonedChoiceAnswer = Field(
        description="Who should speak next?",
    )
    instruction_or_question: CogenticReasonedStringAnswer = Field(
        description="What instruction or question would you give this team member? (Phrase as if speaking directly to them, and include any specific information they may need)",
    )

    @classmethod
    def with_speakers(cls, names: list[str]) -> Type["CogenticProgressLedger"]:
        """Create a new type where next speaker is also a new type containing a fixed set of choices."""
        # Create the choice type with proper annotation
        choices_type = Literal[tuple(names)]  # type: ignore

        speaker_choice = type(
            "CogenticReasonedSpeakerChoice",
            (CogenticReasonedChoiceAnswer,),
            {
                "__annotations__": {"answer": choices_type},
            },
        )

        return type(
            "CogenticProgressLedgerWithSpeakers",
            (cls,),
            {
                "__annotations__": {"next_speaker": speaker_choice},
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
    reason: str | None = Field(description="Reason for the status (if incomplete)")


class CogenticState(BaseGroupChatManagerState):
    """State for the cogentic system."""

    question: str = Field(default="")
    fact_sheet: CogenticFactSheet | None = Field(default=None)
    plan: CogenticPlan | None = Field(default=None)
    current_hypothesis: CogenticHypothesis | None = Field(default=None)
    n_rounds: int = Field(default=0)
    n_stalls: int = Field(default=0)
    type: str = Field(default="CogenticState")
