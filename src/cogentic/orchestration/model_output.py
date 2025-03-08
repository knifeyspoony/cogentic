import json
import re
from typing import Type, TypeVar

from autogen_core import CancellationToken
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    UserMessage,
)
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

FORMAT_PROMPT = """\

### Response Output Schema

Now, I need you to format your previous response in a specific format.

Here is the SCHEMA of the response format:

```json
{response_schema}
```

- Note that you need to create an instance of the model that ADHERES to this schema; not the schema itself.


### Note

Please output your response in a json-formatted code block adhering to the schema. Make sure to wrap your json in markdown tags, e.g.:

```json
... your json content here ...
```

"""

RETRY_MESSAGE = """\

## Response Format Error

- We were unable to parse the JSON output from your response. 
- The output was not in the expected format. 

### Here was your input:

```json
{input}
```

### Error

The error was:

```text
{error}
```

- Consider whether you accidentally output an instance of the schema itself instead of a json object that adheres to the schema.
- Please try to adjust the JSON and respond correctly.
"""


class CogenticOutputParsingError(Exception):
    """Exception raised for errors in the output parsing."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


def _extract_json_from_response(response: str) -> str | None:
    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        # Remove any leading/trailing whitespace
        json_str = json_str.strip()
        return json_str
    return None


async def _reason_and_request_model_via_markdown(
    model_client: ChatCompletionClient,
    json_model_client: ChatCompletionClient,
    messages: list[LLMMessage],
    cancellation_token: CancellationToken,
    response_model: Type[T],
    retries: int = 3,
):
    errors = []
    # First, get the model to respond using the original prompt
    first_response = await model_client.create(
        messages=messages,
        cancellation_token=cancellation_token,
    )
    assert isinstance(first_response.content, str)
    create_messages = messages[:] + [
        AssistantMessage(content=first_response.content, source="assistant"),
        UserMessage(
            content=FORMAT_PROMPT.format(
                response_schema=json.dumps(response_model.model_json_schema(), indent=2)
            ),
            source="assistant",
        ),
    ]
    retry_messages = create_messages[:]
    for _ in range(retries):
        try:
            response = await json_model_client.create(
                messages=retry_messages,
                cancellation_token=cancellation_token,
            )
            assert isinstance(response.content, str)
            json_object = None
            try:
                json_content = _extract_json_from_response(response.content)
                if json_content is None:
                    raise CogenticOutputParsingError(
                        "No JSON markdown block found in the response. Please ensure your response is formatted correctly."
                    )
                json_object = json.loads(json_content)
                return response_model.model_validate(json_object)
            except (CogenticOutputParsingError, ValidationError) as e:
                # We don't want to include multiple error messages in the retry
                input_str = (
                    json.dumps(json_object, indent=2)
                    if json_object
                    else response.content
                )
                retry_messages = create_messages[:]
                retry_message = RETRY_MESSAGE.format(
                    input=input_str,
                    error=str(e),
                )
                errors.append(e)
                retry_messages.append(UserMessage(content=retry_message, source="user"))

        except Exception as e:
            errors.append(e)
            # We don't want to include multiple error messages in the retry
            retry_messages = create_messages[:]
            retry_message = f"Unexpected error. Please try again.\n\nError: {e}"
            retry_messages.append(UserMessage(content=retry_message, source="user"))

    raise ValueError(
        f"Failed to get a valid response after multiple attempts:\n{errors}"
    )


async def _reason_and_request_model_directly(
    model_client: ChatCompletionClient,
    json_model_client: ChatCompletionClient,
    messages: list[LLMMessage],
    cancellation_token: CancellationToken,
    response_model: Type[T],
    retries: int = 3,
) -> T:
    # First, get the model to respond using the original prompt
    first_response = await model_client.create(
        messages=messages,
        cancellation_token=cancellation_token,
    )
    assert isinstance(first_response.content, str)
    create_messages = messages + [
        AssistantMessage(content=first_response.content, source="assistant"),
        UserMessage(
            content="Please format your response using the provided format",
            source="user",
        ),
    ]
    errors = []
    retry_messages = create_messages[:]
    for _ in range(retries):
        try:
            # Now use the json model client to get the JSON output
            model_response = await json_model_client.create(
                messages=retry_messages,
                cancellation_token=cancellation_token,
                extra_create_args={"response_format": response_model},
            )
            assert isinstance(model_response.content, str)
            return response_model.model_validate_json(model_response.content)
        except Exception as e:
            # We don't want to include multiple error messages in the retry
            retry_messages = create_messages[:]
            retry_message = RETRY_MESSAGE.format(
                error=e,
            )
            retry_messages.append(UserMessage(content=retry_message, source="user"))
            errors.append(e)
            continue
    raise ValueError(
        f"Failed to get a valid response after multiple attempts:\n{errors}"
    )


async def reason_and_output_model(
    model_client: ChatCompletionClient,
    json_model_client: ChatCompletionClient,
    messages: list[LLMMessage],
    cancellation_token: CancellationToken,
    response_model: Type[T],
    retries: int = 3,
) -> T:
    """
    Reason and output the model.

    Args:
        model_client (ChatCompletionClient): The model client used for reasoning/inference
        json_model_client (ChatCompletionClient): The model client used to request/extract JSON output
        messages (list[LLMMessage]): The messages to send to the model
        cancellation_token (CancellationToken): The cancellation token to use for the request
        response_model (Type[T]): The model to use for the response
        retries (int, optional): The number of retries to attempt. Defaults to 3.

    Returns:
        T: The model output type
    """
    return await _reason_and_request_model_via_markdown(
        model_client=model_client,
        json_model_client=json_model_client,
        messages=messages,
        cancellation_token=cancellation_token,
        response_model=response_model,
        retries=retries,
    )
