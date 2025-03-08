from typing import Generic, TypeVar

from pydantic import Field

from cogentic.orchestration.models.base import CogenticBaseModel

T = TypeVar("T")


class CogenticReasonedStringAnswer(CogenticBaseModel):
    """Reasoned answer for the progress ledger."""

    reason: str = Field(description="Reason for the answer")
    answer: str = Field(description="Answer to the question")


class CogenticReasonedBooleanAnswer(CogenticBaseModel):
    """Reasoned answer for the progress ledger."""

    reason: str = Field(description="Reason for the answer")
    answer: bool = Field(description="Answer to the question")


class CogenticReasonedChoiceAnswer(CogenticBaseModel, Generic[T]):
    """Reasoned answer for the progress ledger from a set of choices."""

    reason: str = Field(description="Reason for the answer")
    answer: T = Field(
        description="Answer to the question.",
    )
