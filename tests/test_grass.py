import asyncio
import logging
import sys
from datetime import datetime

import pytest
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

from cogentic.llm import get_model_client
from cogentic.orchestration import CogenticGroupChat

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


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

    web_surfer_session_id = f"{base_session_id}-web-surfer"
    web_surfer_model = get_model_client("gpt-4o-mini", session_id=web_surfer_session_id)

    assistant = MultimodalWebSurfer(
        "MultimodalWebSurfer", model_client=web_surfer_model
    )

    team = CogenticGroupChat(
        participants=[assistant],
        model_client=orchestrator_model,
        json_model_client=json_model,
        max_turns_total=32,
        max_turns_per_hypothesis=8,
        max_stalls=3,
    )

    response = await team.run(task="Why is grass green? Give me a simple answer.")

    assert response is not None

    print(response.stop_reason)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_cogentic_group_chat())
