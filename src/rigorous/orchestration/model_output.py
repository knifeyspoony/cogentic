import json
import re
from typing import Type

from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, LLMMessage, UserMessage
from pydantic import BaseModel, ValidationError

REASON_AND_FORMAT_PROMPT = """\

## Response Output Format

Now think step by step. Output your rationale and thoughts in a human-readable format. After you have explained yourself, output a json-formatted markdown code block with the response model.



### Response Output Schema

Here is the SCHEMA of the response format. Note that you need to create an instance of the model that ADHERES to this schema. Don't output the schema itself.

```json
{response_schema}
```

### Note

It is important that you output both plain text reasoning AND the json-formatted code block adhering to the schema.

"""


def _parse_model_from_response(response: str, model: Type[BaseModel]) -> BaseModel:
    """Extract the json from the response string."""
    # Use regex to find the JSON block in the response
    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            return model.model_validate_json(json_str)
        except ValidationError as e:
            raise ValueError(
                f"Failed to parse JSON response: {e}.\n"
                f"Response: {response}\n"
                f"JSON string: {json_str}"
            )
    else:
        raise ValueError("No JSON block found in the response.")


async def reason_and_output_model(
    model_client: ChatCompletionClient,
    messages: list[LLMMessage],
    cancellation_token: CancellationToken,
    response_model: Type[BaseModel],
    retries: int = 3,
) -> BaseModel:
    reason_and_format_message = REASON_AND_FORMAT_PROMPT.format(
        response_schema=json.dumps(response_model.model_json_schema(), indent=2)
    )
    create_messages = messages + [
        UserMessage(content=reason_and_format_message, source="user"),
    ]

    errors = []
    for _ in range(retries):
        try:
            response = await model_client.create(
                messages=create_messages,
                cancellation_token=cancellation_token,
            )
            assert isinstance(response.content, str)
            return _parse_model_from_response(
                response.content,
                response_model,
            )
        except ValueError as e:
            errors.append(e)
            continue
    raise ValueError(
        f"Failed to get a valid response after multiple attempts:\n{errors}"
    )
