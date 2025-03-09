import logging
from datetime import datetime

import pytest

from cogentic.llm import get_model_client
from cogentic.orchestration import CogenticGroupChat

from .common import SimpleAdditionAssistant


@pytest.mark.asyncio
async def test_cogentic_group_chat():
    session_id = f"test-simple-math-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    orchestrator_model = get_model_client("gpt-4o-mini", session_id=session_id)
    assistant_model = get_model_client("gpt-4o-mini", session_id=session_id)

    assistant = SimpleAdditionAssistant(
        "SimpleAdditionAssistant", model_client=assistant_model
    )

    team = CogenticGroupChat(
        participants=[assistant],
        model_client=orchestrator_model,
        max_turns_total=32,
        max_turns_per_hypothesis=8,
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
