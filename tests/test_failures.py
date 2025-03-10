from datetime import datetime

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_core.tools import FunctionTool

from cogentic import CogenticGroupChat
from cogentic.llm import get_model_client


class SimpleAdditionAssistant(AssistantAgent):
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
        raise Exception(f"Unexpected error when attempting to add {a} and {b}.")


@pytest.mark.asyncio
async def test_cogentic_group_chat():
    base_session_id = f"test-grass-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    orchestrator_session_id = f"{base_session_id}-orchestrator"
    orchestrator_json_session_id = f"{orchestrator_session_id}-json-helper"
    orchestrator_model = get_model_client(
        "gpt-4o-mini", session_id=orchestrator_session_id
    )
    json_model = get_model_client(
        "gpt-4o-mini", session_id=orchestrator_json_session_id
    )
    assistant_session_id = f"{base_session_id}-assistant"
    assistant_model = get_model_client("gpt-4o-mini", session_id=assistant_session_id)

    assistant = SimpleAdditionAssistant(
        "SimpleAdditionAssistant", model_client=assistant_model
    )

    team = CogenticGroupChat(
        participants=[assistant],
        model_client=orchestrator_model,
        json_model_client=json_model,
        max_turns_total=32,
        max_turns_per_hypothesis=8,
        max_stalls=3,
    )

    response = await team.run(task="What is 33 + 22?")

    assert response is not None


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)

    asyncio.run(test_cogentic_group_chat())
