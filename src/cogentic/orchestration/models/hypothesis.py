from __future__ import annotations

from typing import Literal

from pydantic import Field

from cogentic.orchestration.models.base import CogenticBaseModel
from cogentic.orchestration.models.test import CogenticTest

CogenticHypothesisState = Literal[
    "unverified",
    "verified",
    "unverifiable",
]


class CogenticHypothesis(CogenticBaseModel):
    """Hypothesis for the cogentic system."""

    hypothesis: str = Field(description="Hypothesis to be tested")
    state: CogenticHypothesisState = Field(description="State of the hypothesis")
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


class CogenticInitialHypotheses(CogenticBaseModel):
    """Initial hypotheses for the cogentic system."""

    hypotheses: list[CogenticHypothesis] = Field(
        description="Hypotheses to be tested", min_length=1
    )
