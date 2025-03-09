import os

if os.environ.get("AZURE_OPENAI_API_KEY") is not None:
    from cogentic.llm.api import get_model_client
else:
    from cogentic.llm.entra import get_model_client

__all__ = [
    "get_model_client",
]
