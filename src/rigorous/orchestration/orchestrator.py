import logging
import re
from typing import Any, List, Mapping

from autogen_agentchat import TRACE_LOGGER_NAME
from autogen_agentchat.base import Response, TerminationCondition
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
    HandoffMessage,
    MultiModalMessage,
    StopMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams._group_chat._base_group_chat_manager import (
    BaseGroupChatManager,
)
from autogen_agentchat.teams._group_chat._events import (
    GroupChatAgentResponse,
    GroupChatMessage,
    GroupChatRequestPublish,
    GroupChatReset,
    GroupChatStart,
    GroupChatTermination,
)
from autogen_agentchat.utils import content_to_str, remove_images
from autogen_core import (
    AgentId,
    CancellationToken,
    DefaultTopicId,
    MessageContext,
    event,
    rpc,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    UserMessage,
)

from rigorous.orchestration.model_output import reason_and_output_model
from rigorous.orchestration.prompts import (
    create_final_answer_prompt,
    create_hypothesis_prompt,
    create_initial_plan_prompt,
    create_initial_question_prompt,
    create_progress_ledger_prompt,
    create_update_facts_on_completion_prompt,
    create_update_facts_on_stall_prompt,
    create_update_plan_on_completion_prompt,
    create_update_plan_on_stall_prompt,
)
from rigorous.orchestration.state import (
    RigorousFactSheet,
    RigorousFinalAnswer,
    RigorousHypothesis,
    RigorousPlan,
    RigorousProgressLedger,
    RigorousState,
)

trace_logger = logging.getLogger(TRACE_LOGGER_NAME)


class RigorousOrchestrator(BaseGroupChatManager):
    """The RigorousOrchestrator manages a group chat with hypothesis validation."""

    def __init__(
        self,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_descriptions: List[str],
        max_turns: int | None,
        model_client: ChatCompletionClient,
        max_stalls: int,
        final_answer_prompt: str,
        termination_condition: TerminationCondition | None,
    ):
        super().__init__(
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_descriptions,
            termination_condition,
            max_turns,
        )
        self._model_client = model_client
        self._max_stalls = max_stalls
        self._final_answer_prompt = final_answer_prompt
        self._name = "RigorousOrchestrator"
        self._max_json_retries = 10
        self._question = ""
        self._plan: RigorousPlan | None = None
        self._fact_sheet: RigorousFactSheet | None = None
        self._current_hypothesis: RigorousHypothesis | None = None
        self._n_rounds = 0
        self._n_stalls = 0

        # Produce a team description. Each agent sould appear on a single line.
        self._team_description = ""
        for topic_type, description in zip(
            self._participant_topic_types, self._participant_descriptions, strict=True
        ):
            self._team_description += (
                re.sub(r"\s+", " ", f"{topic_type}: {description}").strip() + "\n"
            )
        self._team_description = self._team_description.strip()

    async def _log_message(self, log_message: str) -> None:
        trace_logger.debug(log_message)

    @rpc
    async def handle_start(self, message: GroupChatStart, ctx: MessageContext) -> None:  # type: ignore
        """Handle the start of a task."""

        # Check if the conversation has already terminated.
        if (
            self._termination_condition is not None
            and self._termination_condition.terminated
        ):
            early_stop_message = StopMessage(
                content="The group chat has already terminated.", source=self._name
            )
            await self.publish_message(
                GroupChatTermination(message=early_stop_message),
                topic_id=DefaultTopicId(type=self._output_topic_type),
            )
            # Stop the group chat.
            return
        assert message is not None and message.messages is not None

        # Validate the group state given all the messages.
        await self.validate_group_state(message.messages)

        # Log the message.
        await self.publish_message(
            message, topic_id=DefaultTopicId(type=self._output_topic_type)
        )

        # Outer Loop for first time
        # Create the initial task ledger
        #################################
        # Combine all message contents for task
        self._question = " ".join(
            [content_to_str(msg.content) for msg in message.messages]
        )
        planning_conversation: List[LLMMessage] = []

        # Fact sheet
        planning_conversation.append(
            UserMessage(
                content=create_initial_question_prompt(self._question),
                source=self._name,
            )
        )
        self._fact_sheet = await reason_and_output_model(
            self._model_client,
            self._get_compatible_context(planning_conversation),
            ctx.cancellation_token,
            response_model=RigorousFactSheet,
        )
        assert isinstance(self._fact_sheet, RigorousFactSheet)

        # self._model_client.create(
        #     self._get_compatible_context(planning_conversation),
        #     cancellation_token=ctx.cancellation_token,
        #     extra_create_args={"response_format": RigorousFactSheet},
        # )

        # assert isinstance(response.content, str)
        # self._fact_sheet = RigorousFactSheet.model_validate_json(response.content)
        # planning_conversation.append(
        #     AssistantMessage(
        #         content=self._fact_sheet.model_dump_markdown(),
        #         source=self._name,
        #     )
        # )

        # Plan
        planning_conversation.append(
            UserMessage(
                content=create_initial_plan_prompt(self._team_description),
                source=self._name,
            )
        )

        self._plan = await reason_and_output_model(
            self._model_client,
            self._get_compatible_context(planning_conversation),
            ctx.cancellation_token,
            response_model=RigorousPlan,
        )
        assert isinstance(self._plan, RigorousPlan)

        # response = await self._model_client.create(
        #     self._get_compatible_context(planning_conversation),
        #     cancellation_token=ctx.cancellation_token,
        #     extra_create_args={"response_format": RigorousPlan},
        # )

        # assert isinstance(response.content, str)

        # self._plan = RigorousPlan.model_validate_json(response.content)

        # Kick things off
        self._n_stalls = 0
        await self._reenter_outer_loop(ctx.cancellation_token)

    @event
    async def handle_agent_response(
        self, message: GroupChatAgentResponse, ctx: MessageContext
    ) -> None:  # type: ignore
        delta: List[AgentEvent | ChatMessage] = []
        if message.agent_response.inner_messages is not None:
            for inner_message in message.agent_response.inner_messages:
                delta.append(inner_message)
        self._message_thread.append(message.agent_response.chat_message)
        delta.append(message.agent_response.chat_message)

        if self._termination_condition is not None:
            stop_message = await self._termination_condition(delta)
            if stop_message is not None:
                await self.publish_message(
                    GroupChatTermination(message=stop_message),
                    topic_id=DefaultTopicId(type=self._output_topic_type),
                )
                # Stop the group chat and reset the termination conditions and turn count.
                await self._termination_condition.reset()
                return
        await self._orchestrate_step(ctx.cancellation_token)

    async def validate_group_state(self, messages: List[ChatMessage] | None) -> None:
        pass

    async def save_state(self) -> Mapping[str, Any]:
        state = RigorousState(
            message_thread=list(self._message_thread),
            current_turn=self._current_turn,
            task=self._question,
            fact_sheet=self._fact_sheet,
            plan=self._plan,
            current_hypothesis=self._current_hypothesis,
            n_rounds=self._n_rounds,
            n_stalls=self._n_stalls,
        )
        return state.model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        orchestrator_state = RigorousState.model_validate(state)
        self._message_thread = orchestrator_state.message_thread
        self._current_turn = orchestrator_state.current_turn
        self._question = orchestrator_state.question
        self._fact_sheet = orchestrator_state.fact_sheet
        self._plan = orchestrator_state.plan
        self._current_hypothesis = orchestrator_state.current_hypothesis
        self._n_rounds = orchestrator_state.n_rounds
        self._n_stalls = orchestrator_state.n_stalls

    async def select_speaker(self, thread: List[AgentEvent | ChatMessage]) -> str:
        """Not used in this orchestrator, we select next speaker in _orchestrate_step."""
        return ""

    async def reset(self) -> None:
        """Reset the group chat manager."""
        self._message_thread.clear()
        if self._termination_condition is not None:
            await self._termination_condition.reset()
        self._n_rounds = 0
        self._n_stalls = 0
        self._question = ""
        self._fact_sheet = None
        self._plan = None
        self._current_hypothesis = None

    async def _reenter_outer_loop(self, cancellation_token: CancellationToken) -> None:
        """Re-enter Outer loop of the orchestrator after creating task ledger."""
        # Reset the agents
        for participant_topic_type in self._participant_topic_types:
            await self._runtime.send_message(
                GroupChatReset(),
                recipient=AgentId(type=participant_topic_type, key=self.id.key),
                cancellation_token=cancellation_token,
            )
        # Reset partially the group chat manager
        self._message_thread.clear()

        # Choose the next unverified hypothesis
        self._current_hypothesis = next(
            (h for h in self._plan.hypotheses if h.state == "unverified"), None
        )

        if not self._current_hypothesis:
            await self._prepare_final_answer(
                "No remaining hypotheses to verify. This may indicate failure if we cannot come to a conclusion based on verified hypotheses (this is ok, failure is a real and frequent possibility).",
                cancellation_token,
            )
            return

        # Prepare the ledger
        ledger_message = TextMessage(
            content=create_hypothesis_prompt(
                self._team_description,
                self._fact_sheet,
                self._current_hypothesis,
            ),
            source=self._name,
        )

        # Save my copy
        self._message_thread.append(ledger_message)

        # Log it
        await self.publish_message(
            GroupChatMessage(message=ledger_message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )

        # Broadcast
        await self.publish_message(
            GroupChatAgentResponse(
                agent_response=Response(chat_message=ledger_message)
            ),
            topic_id=DefaultTopicId(type=self._group_topic_type),
        )

        # Restart the inner loop
        await self._orchestrate_step(cancellation_token=cancellation_token)

    async def _orchestrate_step(self, cancellation_token: CancellationToken) -> None:
        """Implements the inner loop of the orchestrator and selects next speaker."""
        # Check if we reached the maximum number of rounds
        if self._max_turns is not None and self._n_rounds > self._max_turns:
            await self._prepare_final_answer("Max rounds reached.", cancellation_token)
            return
        self._n_rounds += 1

        # Update the progress ledger
        context = self._thread_to_context()

        progress_ledger_prompt = create_progress_ledger_prompt(
            self._question, self._team_description, self._participant_topic_types
        )
        context.append(UserMessage(content=progress_ledger_prompt, source=self._name))
        ledger_type = RigorousProgressLedger.with_speakers(
            self._participant_topic_types
        )
        assert self._max_json_retries > 0
        key_error: bool = False
        progress_ledger: RigorousProgressLedger | None = None
        for _ in range(self._max_json_retries):
            try:
                progress_ledger = await reason_and_output_model(
                    self._model_client,
                    self._get_compatible_context(context),
                    cancellation_token=cancellation_token,
                    response_model=ledger_type,
                )
                assert isinstance(progress_ledger, ledger_type)
                key_error = False
                break
            except Exception:
                key_error = True
                await self._log_message(
                    "Invalid ledger format encountered, retrying..."
                )
                continue

            # response = await self._model_client.create(
            #     self._get_compatible_context(context),
            #     extra_create_args={"response_format": ledger_type},
            #     cancellation_token=cancellation_token,
            # )
            # ledger_str = response.content
            # try:
            #     assert isinstance(ledger_str, str)
            #     progress_ledger = ledger_type.model_validate_json(ledger_str)
            #     key_error = False
            #     break

            # except (ValidationError, TypeError):
            #     key_error = True
            #     await self._log_message(
            #         "Invalid ledger format encountered, retrying..."
            #     )
            #     continue
        if key_error or progress_ledger is None:
            raise ValueError(
                "Failed to parse ledger information after multiple retries."
            )
        await self._log_message(f"Progress Ledger: {progress_ledger}")

        # Check for task completion
        if progress_ledger.is_request_satisfied.answer:
            await self._log_message("Task completed, preparing final answer...")
            await self._prepare_final_answer(
                progress_ledger.is_request_satisfied.reason, cancellation_token
            )
            return

        # Check if we're done with the current hypothesis
        if progress_ledger.is_current_hypothesis_work_complete.answer:
            await self._log_message("Current hypothesis work complete.")
            # Update the ledger

            # Move on to the next unverified hypothesis
            self._current_hypothesis = next(
                (h for h in self._plan.hypotheses if h.state == "unverified"), None
            )
            if self._current_hypothesis is None:
                await self._log_message(
                    "All hypotheses verified. Preparing final answer."
                )
                await self._prepare_final_answer(
                    "All hypotheses verified.", cancellation_token
                )

        # Check for stalling
        if not progress_ledger.is_progress_being_made.answer:
            self._n_stalls += 1
        elif progress_ledger.is_in_loop.answer:
            self._n_stalls += 1
        else:
            self._n_stalls = max(0, self._n_stalls - 1)

        # Too much stalling
        if self._n_stalls >= self._max_stalls:
            await self._log_message(
                "Stall count exceeded, re-planning with the outer loop..."
            )
            await self._update_task_ledger(cancellation_token, stalled=True)
            await self._reenter_outer_loop(cancellation_token)
            return

        # Broadcast the next step
        message = TextMessage(
            content=progress_ledger.instruction_or_question.answer,
            source=self._name,
        )
        self._message_thread.append(message)  # My copy

        # Log it
        await self._log_message(f"Next Speaker: {progress_ledger.next_speaker.answer}")
        await self.publish_message(
            GroupChatMessage(message=message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )

        # Broadcast it
        await self.publish_message(  # Broadcast
            GroupChatAgentResponse(agent_response=Response(chat_message=message)),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

        # Request that the step be completed
        await self.publish_message(
            GroupChatRequestPublish(),
            topic_id=DefaultTopicId(type=progress_ledger.next_speaker.answer),
            cancellation_token=cancellation_token,
        )

    async def _update_task_ledger(
        self, cancellation_token: CancellationToken, stalled=False
    ) -> None:
        """Update the task ledger (outer loop) with the latest facts and plan."""
        context = self._thread_to_context()

        # Update the facts based on whether we're stalled or not
        if stalled:
            update_facts_prompt = create_update_facts_on_stall_prompt(
                self._current_hypothesis, self._fact_sheet
            )
        else:
            update_facts_prompt = create_update_facts_on_completion_prompt(
                self._current_hypothesis, self._fact_sheet
            )
        context.append(UserMessage(content=update_facts_prompt, source=self._name))

        self._fact_sheet = await reason_and_output_model(
            self._model_client,
            self._get_compatible_context(context),
            cancellation_token=cancellation_token,
            response_model=RigorousFactSheet,
        )
        assert isinstance(self._fact_sheet, RigorousFactSheet)

        # response = await self._model_client.create(
        #     self._get_compatible_context(context),
        #     cancellation_token=cancellation_token,
        #     extra_create_args={"response_format": RigorousFactSheet},
        # )

        # assert isinstance(response.content, str)
        # self._fact_sheet = RigorousFactSheet.model_validate_json(response.content)

        context.append(
            AssistantMessage(
                content=self._fact_sheet.model_dump_markdown(), source=self._name
            )
        )

        # Update the plan based on whether we're stalled or not
        if stalled:
            update_plan_prompt = create_update_plan_on_stall_prompt(
                question=self._question,
                current_hypothesis=self._current_hypothesis,
                team_description=self._team_description,
                fact_sheet=self._fact_sheet,
                plan=self._plan,
            )
        else:
            update_plan_prompt = create_update_plan_on_completion_prompt(
                question=self._question,
                current_hypothesis=self._current_hypothesis,
                team_description=self._team_description,
                fact_sheet=self._fact_sheet,
                plan=self._plan,
            )
        context.append(UserMessage(content=update_plan_prompt, source=self._name))

        self._plan = await reason_and_output_model(
            self._model_client,
            self._get_compatible_context(context),
            cancellation_token=cancellation_token,
            response_model=RigorousPlan,
        )
        assert isinstance(self._plan, RigorousPlan)

        # response = await self._model_client.create(
        #     self._get_compatible_context(context),
        #     cancellation_token=cancellation_token,
        #     extra_create_args={"response_format": RigorousPlan},
        # )

        # assert isinstance(response.content, str)
        # self._plan = RigorousPlan.model_validate_json(response.content)

    async def _prepare_final_answer(
        self, reason: str, cancellation_token: CancellationToken
    ) -> None:
        """Prepare the final answer for the task."""
        context = self._thread_to_context()

        # Get the final answer
        final_answer_prompt = create_final_answer_prompt(self._question)
        context.append(UserMessage(content=final_answer_prompt, source=self._name))

        final_answer = await reason_and_output_model(
            self._model_client,
            self._get_compatible_context(context),
            cancellation_token=cancellation_token,
            response_model=RigorousFinalAnswer,
        )
        assert isinstance(final_answer, RigorousFinalAnswer)

        # response = await self._model_client.create(
        #     self._get_compatible_context(context),
        #     cancellation_token=cancellation_token,
        #     extra_create_args={"response_format": RigorousFinalAnswer},
        # )
        # assert isinstance(response.content, str)
        # final_answer = RigorousFinalAnswer.model_validate_json(response.content)

        message = TextMessage(
            content=final_answer.model_dump_markdown(), source=self._name
        )

        self._message_thread.append(message)  # My copy

        # Log it
        await self.publish_message(
            GroupChatMessage(message=message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )

        # Broadcast
        await self.publish_message(
            GroupChatAgentResponse(agent_response=Response(chat_message=message)),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

        # Signal termination
        await self.publish_message(
            GroupChatTermination(
                message=StopMessage(content=reason, source=self._name)
            ),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )
        if self._termination_condition is not None:
            await self._termination_condition.reset()

    def _thread_to_context(self) -> List[LLMMessage]:
        """Convert the message thread to a context for the model."""
        context: List[LLMMessage] = []
        for m in self._message_thread:
            if isinstance(m, ToolCallRequestEvent | ToolCallExecutionEvent):
                # Ignore tool call messages.
                continue
            elif isinstance(m, StopMessage | HandoffMessage):
                context.append(UserMessage(content=m.content, source=m.source))
            elif m.source == self._name:
                assert isinstance(m, TextMessage | ToolCallSummaryMessage)
                context.append(AssistantMessage(content=m.content, source=m.source))
            else:
                assert isinstance(
                    m, (TextMessage, MultiModalMessage, ToolCallSummaryMessage)
                )
                context.append(UserMessage(content=m.content, source=m.source))
        return context

    def _get_compatible_context(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        """Ensure that the messages are compatible with the underlying client, by removing images if needed."""
        if self._model_client.model_info["vision"]:
            return messages
        else:
            return remove_images(messages)
