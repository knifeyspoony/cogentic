import asyncio
import logging
import sys
from datetime import datetime

import pytest
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

from rigorous.llm import get_model_client
from rigorous.orchestration import RigorousGroupChat

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


@pytest.mark.asyncio
async def test_rigorous_group_chat():
    session_id = f"test-grass-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    orchestrator_model = get_model_client("gpt-4o-mini", session_id=session_id)
    assistant_model = get_model_client("gpt-4o-mini", session_id=session_id)

    assistant = MultimodalWebSurfer("MultimodalWebSurfer", model_client=assistant_model)

    team = RigorousGroupChat(
        participants=[assistant],
        model_client=orchestrator_model,
        max_turns=10,
        max_stalls=3,
    )

    response = await team.run(task="Why is grass green?")

    assert response is not None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_rigorous_group_chat())
