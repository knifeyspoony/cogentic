from datetime import datetime

import pytest

from cogentic import CogenticGroupChat
from cogentic.llm import get_model_client

from .common import SimpleAdditionAssistant


@pytest.mark.asyncio
async def test_cogentic_group_chat():
    session_id = f"test-failures-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    json_session_id = f"{session_id}-json-requests"
    orchestrator_model = get_model_client("gpt-4o-mini", session_id=session_id)
    json_model = get_model_client("gpt-4o-mini", session_id=json_session_id)
    assistant_model = get_model_client("gpt-4o-mini", session_id=session_id)

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
