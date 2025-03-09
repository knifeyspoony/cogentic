import os

from autogen_core.models import ModelInfo
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from cogentic.observability import CogenticChatCompletionClient
from pydantic import BaseModel

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_R1_ENDPOINT = os.environ.get("AZURE_R1_ENDPOINT")
AZURE_R1_API_KEY = os.environ.get("AZURE_R1_API_KEY")

assert AZURE_OPENAI_ENDPOINT, "Azure OpenAI endpoint not set."
assert AZURE_OPENAI_API_VERSION, "Azure OpenAI API version not set."
assert AZURE_OPENAI_API_KEY, "Azure OpenAI API key not set."

assert AZURE_R1_ENDPOINT, "Azure R1 endpoint not set."
assert AZURE_R1_API_KEY, "Azure R1 API key not set."


class ModelDetails(BaseModel):
    name: str
    info: ModelInfo
    endpoint: str = AZURE_OPENAI_ENDPOINT
    api_version: str = AZURE_OPENAI_API_VERSION
    api_key: str = AZURE_OPENAI_API_KEY


MODELS: dict[str, ModelDetails] = {
    "gpt-4o": ModelDetails(
        name="gpt-4o",
        info=ModelInfo(
            family="gpt-4o",
            function_calling=True,
            json_output=True,
            vision=True,
        ),
    ),
    "gpt-4o-mini": ModelDetails(
        name="gpt-4o-mini",
        info=ModelInfo(
            family="gpt-4o",
            function_calling=True,
            json_output=True,
            vision=True,
        ),
    ),
    "o1-mini": ModelDetails(
        name="o1-mini",
        info=ModelInfo(
            family="o1",
            function_calling=False,
            json_output=False,
            vision=False,
        ),
    ),
    "r1": ModelDetails(
        name="r1",
        info=ModelInfo(
            family="r1",
            function_calling=False,
            json_output=False,
            vision=False,
        ),
        api_key=AZURE_R1_API_KEY,
        endpoint=AZURE_R1_ENDPOINT,
    ),
}


def get_model_client(
    model: str,
    name: str | None = None,
    session_id: str | None = None,
) -> CogenticChatCompletionClient:
    model_details = MODELS[model]

    model_client = AzureOpenAIChatCompletionClient(
        model=model,
        azure_endpoint=model_details.endpoint,
        api_version=model_details.api_version,
        api_key=model_details.api_key,
    )

    return CogenticChatCompletionClient(
        model_client=model_client,
        name=name or model_details.name,
        session_id=session_id,
    )
