import os
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine

from autogen_core.models import ModelInfo
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.core.credentials import AccessToken
from azure.identity.aio import AzureCliCredential
from cogentic.observability import CogenticChatCompletionClient
from pydantic import BaseModel

BASE_CREDENTIAL = AzureCliCredential()

CACHED_TOKENS: dict[str, AccessToken] = {}


async def __get_entra_token(scope: str) -> str:
    token = CACHED_TOKENS.get(scope)
    if token is None or token.expires_on < int(
        (datetime.now() - timedelta(minutes=5)).timestamp()
    ):
        token = await BASE_CREDENTIAL.get_token(scope)
        CACHED_TOKENS[scope] = token
        return token.token
    return token.token


def get_token_provider(scope: str) -> Callable[[], Coroutine[Any, Any, str]]:
    async def get_token():
        return await __get_entra_token(scope)

    return get_token


def get_aoai_token_provider() -> Callable[[], Coroutine[Any, Any, str]]:
    return get_token_provider("https://cognitiveservices.azure.com/.default")


AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_R1_ENDPOINT = os.environ.get("AZURE_R1_ENDPOINT")
AZURE_R1_API_KEY = os.environ.get("AZURE_R1_API_KEY")

assert AZURE_OPENAI_ENDPOINT, "Azure OpenAI endpoint not set."
assert AZURE_OPENAI_API_VERSION, "Azure OpenAI API version not set."

assert AZURE_R1_ENDPOINT, "Azure R1 endpoint not set."
assert AZURE_R1_API_KEY, "Azure R1 API key not set."


class ModelDetails(BaseModel):
    name: str
    info: ModelInfo
    endpoint: str = AZURE_OPENAI_ENDPOINT
    api_version: str = AZURE_OPENAI_API_VERSION
    api_key: str | None = None


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
    "o3-mini": ModelDetails(
        name="o3-mini",
        info=ModelInfo(
            family="o3",
            function_calling=True,
            json_output=True,
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

    if model_details.api_key:
        # Use the API key for authentication
        model_client = AzureOpenAIChatCompletionClient(
            model=model,
            azure_endpoint=model_details.endpoint,
            api_version=model_details.api_version,
            api_key=model_details.api_key,
        )
    else:
        model_client = AzureOpenAIChatCompletionClient(
            model=model,
            azure_endpoint=model_details.endpoint,
            api_version=model_details.api_version,
            azure_ad_token_provider=get_aoai_token_provider(),
        )

    return CogenticChatCompletionClient(
        model_client=model_client,
        name=name or model_details.name,
        session_id=session_id,
    )
