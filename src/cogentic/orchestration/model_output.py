import json
import re
from typing import Type, TypeVar

from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, LLMMessage, UserMessage
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

REASON_AND_FORMAT_PROMPT = """\

## Response Output Format

Now think step by step. Output your rationale and thoughts in a human-readable format. After you have explained yourself, output a json-formatted markdown code block with the response model.



### Response Output Schema

Here is the SCHEMA of the response format. Note that you need to create an instance of the model that ADHERES to this schema. Don't output the schema itself.

```json
{response_schema}
```

### Note

It is important that you output both plain text reasoning AND the json-formatted code block adhering to the schema. Make sure to wrap your json in markdown tags, e.g.:

```json
... your json content here ...
```

"""

RETRY_MESSAGE = """\

## Response Format Error

We were unable to parse the JSON output from your response. The output was not in the expected format. 

### Error

The error was {error}

Please try to adjust the JSON and respond correctly.
"""


class CogenticOutputParsingError(Exception):
    """Exception raised for errors in the output parsing."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def _parse_model_from_response(response: str, model: Type[T]) -> T:
    """Extract the json from the response string."""
    # Use regex to find the JSON block in the response
    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            return model.model_validate_json(json_str)
        except ValidationError as e:
            raise CogenticOutputParsingError(
                message=f"Failed to parse provided JSON: {e}.\n"
            )
    else:
        raise CogenticOutputParsingError(
            message="Failed to find JSON block in the response."
        )


async def reason_and_output_model(
    model_client: ChatCompletionClient,
    messages: list[LLMMessage],
    cancellation_token: CancellationToken,
    response_model: Type[T],
    retries: int = 3,
) -> T:
    reason_and_format_message = REASON_AND_FORMAT_PROMPT.format(
        response_schema=json.dumps(response_model.model_json_schema(), indent=2)
    )
    create_messages = messages + [
        UserMessage(content=reason_and_format_message, source="user"),
    ]

    errors = []
    retry_messages = create_messages[:]
    for _ in range(retries):
        response = await model_client.create(
            messages=retry_messages,
            cancellation_token=cancellation_token,
        )
        assert isinstance(response.content, str)
        try:
            return _parse_model_from_response(
                response.content,
                response_model,
            )
        except CogenticOutputParsingError as e:
            # We don't want to include multiple error messages in the retry
            retry_messages = create_messages[:]
            retry_message = RETRY_MESSAGE.format(
                response=response.content,
                error=e.message,
            )
            retry_messages.append(UserMessage(content=retry_message, source="user"))

    raise ValueError(
        f"Failed to get a valid response after multiple attempts:\n{errors}"
    )
