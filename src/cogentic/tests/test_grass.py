import asyncio
import logging
import sys
from datetime import datetime

import pytest
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from cogentic.llm.entra import get_model_client
from cogentic.orchestration import CogenticGroupChat

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


@pytest.mark.asyncio
async def test_cogentic_group_chat():
    session_id = f"test-grass-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    orchestrator_model = get_model_client("gpt-4o-mini", session_id=session_id)
    assistant_model = get_model_client("gpt-4o-mini", session_id=session_id)

    assistant = MultimodalWebSurfer("MultimodalWebSurfer", model_client=assistant_model)

    team = CogenticGroupChat(
        participants=[assistant],
        model_client=orchestrator_model,
        max_turns=10,
        max_stalls=3,
    )

    response = await team.run(task="Why is grass green?")

    assert response is not None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_cogentic_group_chat())
