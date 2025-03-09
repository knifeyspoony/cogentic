import os

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from cogentic import CogenticGroupChat


class SimpleAdditionAssistant(AssistantAgent):
    def __init__(self, name: str, model_client: ChatCompletionClient):
        add_two_ints_tool = FunctionTool(
            name="add_two_ints",
            description="Add two integers.",
            func=self.add_two_ints,
        )
        super().__init__(
            name=name,
            model_client=model_client,
            description="Simple assistant that can add two integers.",
            tools=[add_two_ints_tool],
        )

    def add_two_ints(self, a: int, b: int) -> int:
        return a + b


class SimpleMultiplicationAssistant(AssistantAgent):
    def __init__(self, name: str, model_client: ChatCompletionClient):
        multiply_two_ints_tool = FunctionTool(
            name="multiply_two_ints",
            description="Multiply two integers.",
            func=self.multiply_two_ints,
        )
        super().__init__(
            name=name,
            model_client=model_client,
            description="Simple assistant that can multiply two integers.",
            tools=[multiply_two_ints_tool],
        )

    def multiply_two_ints(self, a: int, b: int) -> int:
        return a * b


async def main():
    # Initialize the Azure OpenAI model client
    orchestration_model_client = AzureOpenAIChatCompletionClient(
        model="gpt-4o",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["OPENAI_API_VERSION"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    assistant_model_client = AzureOpenAIChatCompletionClient(
        model="gpt-4o-mini",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["OPENAI_API_VERSION"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )

    addition_assistant = SimpleAdditionAssistant(
        name="SimpleAdditionAssistant", model_client=assistant_model_client
    )
    multiplication_assistant = SimpleMultiplicationAssistant(
        name="SimpleMultiplicationAssistant", model_client=assistant_model_client
    )

    group_chat = CogenticGroupChat(
        participants=[addition_assistant, multiplication_assistant],
        model_client=orchestration_model_client,
    )

    # Run the group chat with a task
    task = "What is the answer to (33 + 22) * (2 + 3)?"
    response = await group_chat.run(task=task)
    print(response.stop_reason)


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
