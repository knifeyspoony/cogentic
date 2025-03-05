import os
import sys

from cogentic.observability.client import CogenticChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

# Replace openai with langfuse instrumented version if LANGFUSE_HOST is set
if os.environ.get("LANGFUSE_HOST"):
    from autogen_ext.models.openai._openai_client import create_kwargs
    from langfuse import openai  # type: ignore

    # This lets us use the added langfuse create_kwargs in their openai module
    create_kwargs.update(set(("name", "session_id")))
    sys.modules["openai"] = openai


__all__ = ["CogenticChatCompletionClient"]
