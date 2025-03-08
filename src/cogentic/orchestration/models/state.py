from autogen_agentchat.state import BaseGroupChatManagerState
from pydantic import Field

from cogentic.orchestration.models.ledger import CogenticProgressLedger
from cogentic.orchestration.models.plan import CogenticPlan


class CogenticState(BaseGroupChatManagerState):
    """State for the cogentic system."""

    question: str = Field(default="")
    plan: CogenticPlan | None = Field(default=None)
    ledger: CogenticProgressLedger | None = Field(default=None)
    total_turns: int = Field(default=0)
    stalls: int = Field(default=0)
    type: str = Field(default="CogenticState")
