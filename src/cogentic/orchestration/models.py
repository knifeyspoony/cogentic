from typing import Literal, Type

from autogen_agentchat.state import BaseGroupChatManagerState
from pydantic import BaseModel, Field


class CogenticBaseModel(BaseModel):
    def model_dump_markdown(
        self, title: str | None = None, title_level: int = 2, indent: int = 2
    ) -> str:
        """Dump the model as a markdown string."""
        if title:
            title = f"{'#' * title_level} {title}\n\n"
        else:
            title = ""
        return f"{title}```json\n{self.model_dump_json(indent=indent)}\n```"


class CogenticEvidence(CogenticBaseModel):
    """Evidence for the cogentic system."""

    source: str = Field(description="Source of the evidence e.g., file name, URL")
    content: str = Field(description="Relevant content from the source")


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
    state: Literal["complete", "incomplete", "abandoned"] = Field(
        description="State of the test"
    )
    plan: list[CogenticTestTeamMemberPlan] = Field(
        description="Plan for the test. This should include a list of team members and how we envision them solving the test"
    )
    result: str | None = Field(description="Result of the test")
    supporting_evidence: list[CogenticEvidence] | None = Field(
        description="Supporting evidence for the test. For example, the source could be the name of the agent that provided the result, and the content could be a summary of their response."
    )


class CogenticHypothesis(CogenticBaseModel):
    """Hypothesis for the cogentic system."""

    hypothesis: str = Field(description="Hypothesis to be tested")
    state: Literal["unverified", "verified", "unverifiable"] = Field(
        description="State of the hypothesis"
    )
    completion_summary: str | None = Field(
        description="When completed, a summary of the results"
    )
    tests: list[CogenticTest] = Field(
        description="Tests for the hypothesis. Hypotheses must have at least one test",
        min_length=1,
    )

    @property
    def all_tests_finished(self) -> bool:
        """Check if all tests are completed or we're in a completed state."""
        return all(test.state != "incomplete" for test in self.tests)

    @property
    def all_tests_completed(self) -> bool:
        """Check if all tests are completed."""
        return all(test.state == "complete" for test in self.tests)

    @property
    def current_test(self) -> CogenticTest | None:
        """Get the current test to be completed."""
        for test in self.tests:
            if test.state == "incomplete":
                return test
        return None

    def update_test(self, test: CogenticTest) -> None:
        """Update the test in the hypothesis."""
        for i, t in enumerate(self.tests):
            if t.name == test.name:
                self.tests[i] = test
                return
        raise ValueError(f"Test '{test.name}' not found in the hypothesis.")


class CogenticFact(CogenticBaseModel):
    """Fact for the cogentic system."""

    content: str = Field(description="Fact content")
    source: Literal["question", "test_result"] = Field(
        description="How this fact was derived"
    )
    notes: str | None = Field(description="Any applicable notes")
    supporting_test: str | None = Field(
        description="If the source is `test_result`, include the test name that produced this fact"
    )


class CogenticPlan(CogenticBaseModel):
    """Plan for the cogentic system."""

    hypotheses: list[CogenticHypothesis] = Field(
        description="Hypotheses to be tested", min_length=1
    )
    benched_team_members: list[str] = Field(
        description="Team members who are benched due to repeated unrecoverable errors"
    )

    def update_hypothesis(self, hypothesis: CogenticHypothesis) -> None:
        """Update the hypothesis in the plan."""
        for i, h in enumerate(self.hypotheses):
            if h.hypothesis == hypothesis.hypothesis:
                self.hypotheses[i] = hypothesis
                break
        else:
            raise ValueError(
                f"Hypothesis '{hypothesis.hypothesis}' not found in the plan."
            )

    @property
    def current_hypothesis(self) -> CogenticHypothesis | None:
        """Get the current hypothesis to be tested."""
        for hypothesis in self.hypotheses:
            if hypothesis.state == "unverified":
                return hypothesis
        return None


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
    answer: Literal[""] | None = Field(
        description="Answer to the question, or None if no choice is applicable. If None, provide a reason why no choice is applicable.",
    )


class CogenticNextSpeaker(CogenticBaseModel):
    """Next speaker for the cogentic system."""

    next_speaker: CogenticReasonedChoiceAnswer = Field(
        description="Who should speak next?",
    )
    instruction_or_question: CogenticReasonedStringAnswer = Field(
        description="What instruction or question would you give this team member? (Phrase as if speaking directly to them, and include any specific information they may need)",
    )

    @classmethod
    def with_choices(cls, choices: list[str]) -> Type["CogenticNextSpeaker"]:
        """Create a new type where next speaker is also a new type containing a fixed set of choices."""
        # Create the choice type with proper annotation
        choices_type = Literal[tuple(choices)]  # type: ignore

        speaker_choice = type(
            "CogenticReasonedSpeakerChoice",
            (CogenticReasonedChoiceAnswer,),
            {
                "__annotations__": {"answer": choices_type},
            },
        )

        return type(
            "CogenticNextSpeakerWithChoices",
            (cls,),
            {
                "__annotations__": {"next_speaker": speaker_choice},
            },
        )


class CogenticProgressLedger(CogenticBaseModel):
    """Progress ledger for the cogentic system."""

    is_request_satisfied: CogenticReasonedBooleanAnswer = Field(
        description="Is the original question fully answered? (True if complete, or False if the original question has yet to be SUCCESSFULLY and FULLY addressed)",
    )
    is_in_loop: CogenticReasonedBooleanAnswer = Field(
        description="Are we in a loop where we are repeating the same requests and / or getting the same responses as before? Loops can span multiple turns, and can include repeated actions like scrolling up or down more than a handful of times.",
    )
    is_progress_being_made: CogenticReasonedBooleanAnswer = Field(
        description="Are we making forward progress? (True if just starting, or recent messages are adding value. False if recent messages show evidence of being stuck in a loop or if there is evidence of significant barriers to success such as the inability to read from a required file)",
    )
    new_facts: list[CogenticFact] = Field(
        description="Any remarkable facts that have been discovered since the last progress ledger",
    )
    current_test: CogenticTest = Field(
        description="Current test being worked on",
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


class CogenticState(BaseGroupChatManagerState):
    """State for the cogentic system."""

    question: str = Field(default="")
    fact_sheet: CogenticFactSheet | None = Field(default=None)
    plan: CogenticPlan | None = Field(default=None)
    current_ledger: CogenticProgressLedger | None = Field(default=None)
    n_rounds: int = Field(default=0)
    n_stalls: int = Field(default=0)
    type: str = Field(default="CogenticState")
