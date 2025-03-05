import logging
from datetime import datetime

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_core.tools import FunctionTool
from cogentic.llm.entra import get_model_client
from cogentic.orchestration import CogenticGroupChat


class SimpleMathAssistant(AssistantAgent):
    def __init__(self, name: str, model_client: ChatCompletionClient):
        add_two_ints_tool = FunctionTool(
            name="add_two_ints",
            description="Add two integers.",
            func=self.add_two_ints,
        )
        super().__init__(
            name=name,
            model_client=model_client,
            description="Simple assistant that can add two integers.",
            tools=[add_two_ints_tool],
        )

    def add_two_ints(self, a: int, b: int) -> int:
        return a + b


@pytest.mark.asyncio
async def test_cogentic_group_chat():
    session_id = f"test-simple-math-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    orchestrator_model = get_model_client("gpt-4o-mini", session_id=session_id)
    assistant_model = get_model_client("gpt-4o-mini", session_id=session_id)

    assistant = SimpleMathAssistant("SimpleMathAssistant", model_client=assistant_model)

    team = CogenticGroupChat(
        participants=[assistant],
        model_client=orchestrator_model,
        max_turns=10,
        max_stalls=3,
    )

    response = await team.run(task="What is 33 + 22?")

    assert response is not None
    last_message = response.messages[-1]
    assert "55" in last_message.content


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    asyncio.run(test_cogentic_group_chat())
