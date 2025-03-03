from typing import Literal, Type

from autogen_agentchat.state import BaseGroupChatManagerState
from pydantic import BaseModel, Field


class RigorousBaseModel(BaseModel):
    def model_dump_markdown(self, indent: int = 2) -> str:
        return f"```json\n{self.model_dump_json(indent=indent)}\n```"


class RigorousEvidence(RigorousBaseModel):
    """Evidence for the rigorous system."""

    source: str = Field(description="Source of the evidence e.g., file name, URL")
    content: str = Field(description="Relevant content from the source")


class RigorousTest(RigorousBaseModel):
    """A test which is part of a hypothesis."""

    description: str = Field(description="Description of the test")
    completed: bool = Field(description="Whether the test has been completed")
    result: str | None = Field(description="Result of the test")
    supporting_evidence: list[RigorousEvidence] | None = Field(
        description="Supporting evidence for the test"
    )


class RigorousHypothesis(RigorousBaseModel):
    """Hypothesis for the rigorous system."""

    hypothesis: str = Field(description="Hypothesis to be tested")
    state: Literal["unverified", "verified", "unverifiable"] = Field(
        description="State of the hypothesis"
    )
    notes: str | None = Field(description="Notes about the hypothesis")
    tests: list[RigorousTest] = Field(description="Tests for the hypothesis")


class RigorousFact(RigorousBaseModel):
    """Fact for the rigorous system."""

    content: str = Field(description="Fact content")
    state: Literal["provided", "verified"] = Field(description="State of the fact")
    notes: str | None = Field(description="Any applicable notes")
    supporting_evidence: list[RigorousEvidence] = Field(
        description="Supporting evidence for the fact"
    )


class RigorousPlan(RigorousBaseModel):
    """Plan for the rigorous system."""

    hypotheses: list[RigorousHypothesis] = Field(description="Hypotheses to be tested")


class RigorousFactSheet(RigorousBaseModel):
    """Fact sheet for the rigorous system."""

    facts: list[RigorousFact] = Field(
        description="All known facts relevant to our task"
    )


class RigorousReasonedStringAnswer(RigorousBaseModel):
    """Reasoned answer for the progress ledger."""

    reason: str = Field(description="Reason for the answer")
    answer: str = Field(description="Answer to the question")


class RigorousReasonedBooleanAnswer(RigorousBaseModel):
    """Reasoned answer for the progress ledger."""

    reason: str = Field(description="Reason for the answer")
    answer: bool = Field(description="Answer to the question")


class RigorousReasonedChoiceAnswer(RigorousBaseModel):
    """Reasoned answer for the progress ledger."""

    reason: str = Field(description="Reason for the answer")
    answer: Literal[""] = Field(description="Answer to the question")


class RigorousProgressLedger(RigorousBaseModel):
    """Progress ledger for the rigorous system."""

    is_current_hypothesis_work_complete: RigorousReasonedBooleanAnswer = Field(
        description="Is the current hypothesis work complete? (True if the current hypothesis is verified, or unverifiable. False if the current hypothesis is unverified and there is still work to be done) ",
    )
    is_request_satisfied: RigorousReasonedBooleanAnswer = Field(
        description="Is the original question fully answered? (True if complete, or False if the original question has yet to be SUCCESSFULLY and FULLY addressed)",
    )
    is_in_loop: RigorousReasonedBooleanAnswer = Field(
        description="Are we in a loop where we are repeating the same requests and / or getting the same responses as before? Loops can span multiple turns, and can include repeated actions like scrolling up or down more than a handful of times.",
    )
    is_progress_being_made: RigorousReasonedBooleanAnswer = Field(
        description="Are we making forward progress? (True if just starting, or recent messages are adding value. False if recent messages show evidence of being stuck in a loop or if there is evidence of significant barriers to success such as the inability to read from a required file)",
    )
    next_speaker: RigorousReasonedChoiceAnswer = Field(
        description="Who should speak next?",
    )
    instruction_or_question: RigorousReasonedStringAnswer = Field(
        description="What instruction or question would you give this team member? (Phrase as if speaking directly to them, and include any specific information they may need)",
    )

    @classmethod
    def with_speakers(cls, names: list[str]) -> Type["RigorousProgressLedger"]:
        """Create a new type where next speaker is also a new type containing a fixed set of choices."""
        # Create the choice type with proper annotation
        choices_type = Literal[tuple(names)]  # type: ignore

        speaker_choice = type(
            "RigorousReasonedSpeakerChoice",
            (RigorousReasonedChoiceAnswer,),
            {
                "__annotations__": {"answer": choices_type},
            },
        )

        return type(
            "RigorousProgressLedgerWithSpeakers",
            (cls,),
            {
                "__annotations__": {"next_speaker": speaker_choice},
            },
        )


class RigorousState(BaseGroupChatManagerState):
    """State for the rigorous system."""

    question: str = Field(default="")
    fact_sheet: RigorousFactSheet = Field(default_factory=RigorousFactSheet)
    plan: RigorousPlan = Field(default_factory=RigorousPlan)
    current_hypothesis: RigorousHypothesis | None = Field(default=None)
    n_rounds: int = Field(default=0)
    n_stalls: int = Field(default=0)
    type: str = Field(default="RigorousState")
