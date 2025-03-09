from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_core.tools import FunctionTool


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
