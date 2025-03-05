import os
from typing import Any, AsyncGenerator, Mapping, Optional, Sequence, Union
from uuid import uuid4

from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage
from autogen_core.tools import Tool, ToolSchema


class CogenticChatCompletionClient(ChatCompletionClient):
    def __init__(
        self,
        model_client: ChatCompletionClient,
        name: str,
        session_id: str | None = None,
    ):
        self.model_client = model_client
        self.name = name
        self.session_id = session_id or uuid4().hex

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        # None means do not override the default
        # A value means to override the client default - often specified in the constructor
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        if os.environ.get("LANGFUSE_HOST"):
            context_aware_extra_create_args = {
                k: v for k, v in extra_create_args.items()
            }
            context_aware_extra_create_args["name"] = self.name
            context_aware_extra_create_args["session_id"] = self.session_id
        else:
            context_aware_extra_create_args = extra_create_args

        # Call model client
        result = await self.model_client.create(
            messages=messages,
            tools=tools,
            json_output=json_output,
            extra_create_args=context_aware_extra_create_args,
            cancellation_token=cancellation_token,
        )
        return result

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        # None means do not override the default
        # A value means to override the client default - often specified in the constructor
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        if os.environ.get("LANGFUSE_HOST"):
            context_aware_extra_create_args = {
                k: v for k, v in extra_create_args.items()
            }
            context_aware_extra_create_args["name"] = self.name
            context_aware_extra_create_args["session_id"] = self.session_id
        else:
            context_aware_extra_create_args = extra_create_args

        # Yield from model client
        async for result in self.model_client.create_stream(
            messages=messages,
            tools=tools,
            json_output=json_output,
            extra_create_args=context_aware_extra_create_args,
            cancellation_token=cancellation_token,
        ):
            yield result

    def remaining_tokens(
        self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []
    ) -> int:
        return self.model_client.remaining_tokens(messages=messages, tools=tools)

    def count_tokens(
        self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []
    ) -> int:
        return self.model_client.count_tokens(messages=messages, tools=tools)

    def actual_usage(self):
        return self.model_client.actual_usage()

    def total_usage(self):
        return self.model_client.total_usage()

    @property
    def capabilities(self):
        return self.model_client.capabilities

    @property
    def model_info(self):
        return self.model_client.model_info
