import logging
from typing import Callable, List

from autogen_agentchat import EVENT_LOGGER_NAME, TRACE_LOGGER_NAME
from autogen_agentchat.base import ChatAgent, TerminationCondition
from autogen_agentchat.teams._group_chat._base_group_chat import BaseGroupChat
from autogen_core import Component, ComponentModel
from autogen_core.models import ChatCompletionClient
from cogentic.orchestration.orchestrator import CogenticOrchestrator
from cogentic.orchestration.prompts.prompts import (
    FINAL_ANSWER_PROMPT as RIGOROUS_FINAL_ANSWER_PROMPT,
)
from pydantic import BaseModel
from typing_extensions import Self

trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
event_logger = logging.getLogger(EVENT_LOGGER_NAME)


class CogenticGroupChatConfig(BaseModel):
    """The declarative configuration for a CogenticGroupChat."""

    participants: List[ComponentModel]
    model_client: ComponentModel
    termination_condition: ComponentModel | None = None
    max_turns: int | None = None
    max_stalls: int
    final_answer_prompt: str


class CogenticGroupChat(BaseGroupChat, Component[CogenticGroupChatConfig]):
    """A team that runs a group chat with participants managed by the CogenticOrchestrator."""

    component_config_schema = CogenticGroupChatConfig
    component_provider_override = "autogen_agentchat.teams.CogenticGroupChat"

    def __init__(
        self,
        participants: List[ChatAgent],
        model_client: ChatCompletionClient,
        *,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = 20,
        max_stalls: int = 3,
        final_answer_prompt: str = RIGOROUS_FINAL_ANSWER_PROMPT,
    ):
        super().__init__(
            participants,
            group_chat_manager_class=CogenticOrchestrator,
            termination_condition=termination_condition,
            max_turns=max_turns,
        )

        # Validate the participants.
        if len(participants) == 0:
            raise ValueError(
                "At least one participant is required for CogenticGroupChat."
            )
        self._model_client = model_client
        self._max_stalls = max_stalls
        self._final_answer_prompt = final_answer_prompt

    def _create_group_chat_manager_factory(
        self,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_descriptions: List[str],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
    ) -> Callable[[], CogenticOrchestrator]:
        return lambda: CogenticOrchestrator(
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_descriptions,
            max_turns,
            self._model_client,
            self._max_stalls,
            self._final_answer_prompt,
            termination_condition,
        )

    def _to_config(self) -> CogenticGroupChatConfig:
        participants = [
            participant.dump_component() for participant in self._participants
        ]
        termination_condition = (
            self._termination_condition.dump_component()
            if self._termination_condition
            else None
        )
        return CogenticGroupChatConfig(
            participants=participants,
            model_client=self._model_client.dump_component(),
            termination_condition=termination_condition,
            max_turns=self._max_turns,
            max_stalls=self._max_stalls,
            final_answer_prompt=self._final_answer_prompt,
        )

    @classmethod
    def _from_config(cls, config: CogenticGroupChatConfig) -> Self:
        participants = [
            ChatAgent.load_component(participant) for participant in config.participants
        ]
        model_client = ChatCompletionClient.load_component(config.model_client)
        termination_condition = (
            TerminationCondition.load_component(config.termination_condition)
            if config.termination_condition
            else None
        )
        return cls(
            participants,
            model_client,
            termination_condition=termination_condition,
            max_turns=config.max_turns,
            max_stalls=config.max_stalls,
            final_answer_prompt=config.final_answer_prompt,
        )
