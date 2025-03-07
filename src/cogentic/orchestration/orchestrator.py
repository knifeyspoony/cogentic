import logging
import re
from typing import Any, List, Mapping

from autogen_agentchat import TRACE_LOGGER_NAME
from autogen_agentchat.base import Response
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

from cogentic.orchestration.model_output import reason_and_output_model
from cogentic.orchestration.models import (
    CogenticFactSheet,
    CogenticFinalAnswer,
    CogenticNextSpeaker,
    CogenticPlan,
    CogenticProgressLedger,
    CogenticState,
)
from cogentic.orchestration.prompts import (
    create_final_answer_prompt,
    create_hypothesis_prompt,
    create_initial_fact_sheet_prompt,
    create_initial_plan_prompt,
    create_next_speaker_prompt,
    create_progress_ledger_prompt,
    create_update_plan_on_completion_prompt,
    create_update_plan_on_stall_prompt,
)


class CogenticOrchestrator(BaseGroupChatManager):
    """The CogenticOrchestrator manages a group chat with hypothesis validation."""

    def __init__(
        self,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_descriptions: List[str],
        model_client: ChatCompletionClient,
        max_turns_total: int | None,
        max_turns_per_hypothesis: int | None,
        max_turns_per_test: int | None,
        max_stalls: int,
        final_answer_prompt: str,
    ):
        super().__init__(
            group_topic_type=group_topic_type,
            output_topic_type=output_topic_type,
            participant_topic_types=participant_topic_types,
            participant_descriptions=participant_descriptions,
            termination_condition=None,
            max_turns=max_turns_total,
        )
        self._model_client = model_client
        self._max_stalls = max_stalls
        self._max_turns_total = max_turns_total
        self._max_turns_per_hypothesis = max_turns_per_hypothesis
        self._max_turns_per_test = max_turns_per_test
        self._final_answer_prompt = final_answer_prompt
        self._name = "CogenticOrchestrator"
        self._max_json_retries = 10
        self._question = ""
        self._plan: CogenticPlan | None = None
        self._fact_sheet: CogenticFactSheet | None = None
        self._current_ledger: CogenticProgressLedger | None = None
        self._total_turns: int = 0
        self._current_hypothesis_turns: int = 0
        self._current_test_turns: int = 0
        self._current_stall_count: int = 0
        self.logger = logging.getLogger(TRACE_LOGGER_NAME)

        # Create a markdown table of our team members with Name and Description
        self._team_description = "| Name | Description |\n"
        self._team_description += "| ---- | ----------- |\n"
        for topic_type, description in zip(
            self._participant_topic_types, self._participant_descriptions
        ):
            self._team_description += (
                re.sub(r"\s+", " ", f"| {topic_type} | {description} |").strip() + "\n"
            )
        self._team_description = self._team_description.strip()

    async def _publish_to_output(
        self,
        message: Any,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        """Log a message to our output topic"""

        await self.publish_message(
            GroupChatMessage(
                message=message,
            ),
            topic_id=DefaultTopicId(type=self._output_topic_type),
            cancellation_token=cancellation_token,
        )

    async def _publish_to_group(
        self,
        message: Any,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        """Log a message to our group topic"""

        await self.publish_message(
            GroupChatAgentResponse(
                agent_response=Response(chat_message=message),
            ),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

    async def _request_speaker(
        self, target_topic_id: str, cancellation_token: CancellationToken
    ) -> None:
        """Request a speaker to respond to the group chat.

        Args:
            target (str): The target speaker's topic ID.

        """
        await self.publish_message(
            GroupChatRequestPublish(),
            topic_id=DefaultTopicId(type=target_topic_id),
            cancellation_token=cancellation_token,
        )

    async def _terminate_chat(
        self, message: str, cancellation_token: CancellationToken
    ) -> None:
        """Terminate the chat.

        Args:
            message (str): The termination message.

        """
        await self.publish_message(
            message=GroupChatTermination(
                message=StopMessage(content=message, source=self._name)
            ),
            topic_id=DefaultTopicId(type=self._output_topic_type),
            cancellation_token=cancellation_token,
        )

    async def _start_chat(
        self, messages: list[ChatMessage], cancellation_token: CancellationToken
    ) -> None:
        """Start the chat."""
        await self.publish_message(
            GroupChatStart(messages=messages),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

    async def _reset_state(self, cancellation_token: CancellationToken) -> None:
        """Reset the chat state."""
        self._current_stall_count = 0
        self._current_hypothesis_turns = 0
        self._current_test_turns = 0
        for participant_topic_type in self._participant_topic_types:
            await self._runtime.send_message(
                GroupChatReset(),
                recipient=AgentId(type=participant_topic_type, key=self.id.key),
                cancellation_token=cancellation_token,
            )

    @rpc
    async def handle_start(self, message: GroupChatStart, ctx: MessageContext) -> None:  # type: ignore
        """
        Handle the start of a group chat.

        We initialize the group chat manager and set up the initial state.

        - Create a fact sheet of facts presented in the task
        - Create an initial plan containing hypotheses to be verified
        - Finish by selecting a hypothesis to process

        """
        assert message is not None and message.messages is not None

        # Start chat
        await self._start_chat(message.messages, ctx.cancellation_token)

        # Initialize the question by combining the initial messages
        self._question = "\n".join(
            [content_to_str(msg.content) for msg in message.messages]
        )

        # The planning conversation only exists to create a formal plan and fact sheet.
        # It is not broadcast to the group chat.
        planning_conversation: List[LLMMessage] = []

        # Ask the model to create a fact sheet of facts presented in the task/question
        planning_conversation.append(
            UserMessage(
                content=create_initial_fact_sheet_prompt(self._question),
                source=self._name,
            )
        )
        self._fact_sheet = await reason_and_output_model(
            self._model_client,
            self._get_compatible_context(planning_conversation),
            ctx.cancellation_token,
            response_model=CogenticFactSheet,
            retries=self._max_json_retries,
        )
        assert isinstance(self._fact_sheet, CogenticFactSheet)

        # Add the fact sheet to the planning conversation
        planning_conversation.append(
            AssistantMessage(
                content=self._fact_sheet.model_dump_markdown(
                    title="Initial Fact Sheet"
                ),
                source=self._name,
            )
        )

        # Now, based on the question and the known facts, ask the model to create a plan
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
            response_model=CogenticPlan,
            retries=self._max_json_retries,
        )
        assert isinstance(self._plan, CogenticPlan)
        # We save the plan internally, it isn't broadcast.

        # Enter hypothesis processing

        await self._process_next_hypothesis(ctx.cancellation_token)

    async def _process_next_hypothesis(
        self, cancellation_token: CancellationToken
    ) -> None:
        """Process the next hypothesis in the plan.

        - Reset the agents
        - Choose the next unverified hypothesis. If none, prepare the final answer.

        """
        # Reset state
        await self._reset_state(cancellation_token)
        assert self._plan and self._fact_sheet

        # Validate that we have team members available
        if self._plan.benched_team_members:
            available_team_members = [
                member
                for member in self._participant_topic_types
                if member not in self._plan.benched_team_members
            ]
            if not available_team_members:
                await self._prepare_final_answer(
                    "All team members have been benched due to errors.",
                    cancellation_token,
                )
                return

        # Choose the next unverified hypothesis. If our current hypothesis has no remaining tests
        if (
            self._plan.current_hypothesis
            and self._plan.current_hypothesis.all_tests_finished
        ):
            if self._plan.current_hypothesis.all_tests_completed:
                self._plan.current_hypothesis.state = "verified"
            else:
                self._plan.current_hypothesis.state = "unverifiable"

        if not self._plan.current_hypothesis:
            await self._prepare_final_answer(
                "No remaining hypotheses to verify. This may indicate failure if we cannot come to a conclusion based on verified hypotheses (this is ok, failure is a real and frequent possibility).",
                cancellation_token,
            )
            return

        await self._initialize_orchestrator_message_thread()
        await self._hypothesis_loop(
            cancellation_token=cancellation_token, first_iteration=True
        )

    async def _initialize_orchestrator_message_thread(self):
        """Initialize the orchestrator message thread with the current hypothesis."""
        assert self._plan and self._fact_sheet and self._plan.current_hypothesis

        # Clear the orchestrator message thread
        self._message_thread.clear()

        # Clear the ledger
        self._current_ledger = None

        # Introduce the current hypothesis
        hypothesis_message = TextMessage(
            content=create_hypothesis_prompt(
                self._question,
                self._team_description,
                self._fact_sheet,
                self._plan.current_hypothesis,
            ),
            source=self._name,
        )
        self._message_thread.append(hypothesis_message)

        # Publish to the output and group
        await self._publish_to_output(message=hypothesis_message)
        await self._publish_to_group(message=hypothesis_message)

    def _update_facts_in_thread(self):
        """Update the facts in the message thread."""
        assert self._fact_sheet and self._plan and self._plan.current_hypothesis
        # We will replace the first message in the thread
        hypothesis_message = TextMessage(
            content=create_hypothesis_prompt(
                self._question,
                self._team_description,
                self._fact_sheet,
                self._plan.current_hypothesis,
            ),
            source=self._name,
        )
        self._message_thread[0] = hypothesis_message

    async def _check_max_turns(self, cancellation_token: CancellationToken) -> None:
        """Check if we have reached the maximum number of turns for the orchestrator."""
        if self._max_turns is not None and self._total_turns > self._max_turns:
            await self._prepare_final_answer(
                f"Maximum turn count reached ({self._max_turns}). Can we come to a conclusion?",
                cancellation_token,
            )
            return

    async def _hypothesis_loop(
        self, cancellation_token: CancellationToken, first_iteration=False
    ) -> None:
        """
        This is the work loop for the current hypothesis
        After each iteration, we update facts and current hypothesis data.
        """
        # Check if we have reached the maximum number of turns for the orchestrator.
        await self._check_max_turns(cancellation_token)
        self._total_turns += 1
        self._current_hypothesis_turns += 1
        self._current_test_turns += 1

        # If this not our first iteration, we want to update the facts and plan
        if not first_iteration:
            self._current_ledger = await self._update_progress_ledger(
                cancellation_token=cancellation_token
            )

            # Update the fact sheet, we might use it in the final answer prompt
            assert self._plan and self._fact_sheet and self._plan.current_hypothesis

            # Check if we have new facts to add to the fact sheet
            if self._current_ledger.new_facts:
                self._fact_sheet.facts.extend(self._current_ledger.new_facts)
                self.logger.info(
                    f"New facts added to the fact sheet: {self._current_ledger.new_facts}"
                )
                # To avoid confusion, we'll replace the fact sheet in the message thread
                self._update_facts_in_thread()

            # Check if we're done with the current hypothesis. If so, on to the next.
            if self._current_ledger.current_test.state != "incomplete":
                self._current_test_turns = 0
                self.logger.info("Current test work complete.")

                # Update the test in the current hypothesis
                self._plan.current_hypothesis.update_test(
                    self._current_ledger.current_test
                )

                # Check for task completion
                if self._current_ledger.is_request_satisfied.answer:
                    self.logger.info("Task completed, preparing final answer...")
                    await self._prepare_final_answer(
                        self._current_ledger.is_request_satisfied.reason,
                        cancellation_token,
                    )
                    return
                else:
                    if self._plan.current_hypothesis.all_tests_finished:
                        await self._update_plan_on_hypothesis_completion(
                            cancellation_token=cancellation_token
                        )

                    await self._process_next_hypothesis(cancellation_token)
                    return

            # Check for stalling
            if not self._current_ledger.is_progress_being_made.answer:
                self._current_stall_count += 1
            elif self._current_ledger.is_in_loop.answer:
                self._current_stall_count += 1
            else:
                # Decrement stall count if we're making progress
                self._current_stall_count = max(0, self._current_stall_count - 1)

            # Re-plan on full stall or max hypothesis turns reached
            if self._needs_replan():
                self.logger.warning(
                    "Stalled or hypothesis turn count exceeded, time to update the plan."
                )
                await self._update_plan_on_stall(cancellation_token)
                await self._process_next_hypothesis(cancellation_token)
                return

        next_speaker = await self._select_next_speaker(cancellation_token)

        if not next_speaker.next_speaker.answer:
            await self._prepare_final_answer(
                f"We can't make any further progress with the current team composition. Reason: {next_speaker.next_speaker.reason}",
                cancellation_token,
            )
            return

        # Publish the next speaker message to the output topic and the group
        message = TextMessage(
            content=next_speaker.instruction_or_question.answer,
            source=self._name,
        )
        self._message_thread.append(message)

        await self._publish_to_output(
            message=message, cancellation_token=cancellation_token
        )
        await self._publish_to_group(
            message=message, cancellation_token=cancellation_token
        )

        # Ask the next speaker to respond
        await self._request_speaker(
            target_topic_id=next_speaker.next_speaker.answer,
            cancellation_token=cancellation_token,
        )

    async def _update_plan_on_hypothesis_completion(
        self, cancellation_token: CancellationToken
    ) -> None:
        """Update the plan on hypothesis completion."""
        assert self._plan and self._plan.current_hypothesis and self._fact_sheet
        update_plan_prompt = create_update_plan_on_completion_prompt(
            question=self._question,
            current_hypothesis=self._plan.current_hypothesis,
            team_description=self._team_description,
            fact_sheet=self._fact_sheet,
            plan=self._plan,
        )
        self._plan = await reason_and_output_model(
            self._model_client,
            self._get_compatible_context(
                [UserMessage(content=update_plan_prompt, source=self._name)]
            ),
            cancellation_token=cancellation_token,
            response_model=CogenticPlan,
            retries=self._max_json_retries,
        )

    def _needs_replan(self) -> bool:
        """Check if we need to replan based on the current state."""
        stalled = self._current_stall_count >= self._max_stalls
        hypothesis_turns_exceeded = self._max_turns_per_hypothesis is not None and (
            self._current_hypothesis_turns >= self._max_turns_per_hypothesis
        )
        test_turns_exceeded = self._max_turns_per_test is not None and (
            self._current_test_turns >= self._max_turns_per_test
        )
        return stalled or hypothesis_turns_exceeded or test_turns_exceeded

    async def _update_progress_ledger(
        self, cancellation_token: CancellationToken
    ) -> CogenticProgressLedger:
        """Create the progress ledger based on the current state of the group chat.

        Returns:
            CogenticProgressLedger: The progress ledger containing updated facts and hypothesis info.
        """
        assert (
            self._plan
            and self._plan.current_hypothesis
            and self._plan.current_hypothesis.current_test
        )

        context = self._thread_to_context()

        progress_ledger_prompt = create_progress_ledger_prompt(
            self._plan.current_hypothesis.current_test, self._current_ledger
        )
        context.append(UserMessage(content=progress_ledger_prompt, source=self._name))
        assert self._max_json_retries > 0
        progress_ledger: CogenticProgressLedger | None = None
        progress_ledger = await reason_and_output_model(
            self._model_client,
            self._get_compatible_context(context),
            cancellation_token=cancellation_token,
            response_model=CogenticProgressLedger,
            retries=self._max_json_retries,
        )
        assert isinstance(progress_ledger, CogenticProgressLedger)
        self.logger.debug(f"Progress Ledger: {progress_ledger}")
        return progress_ledger

    async def _select_next_speaker(
        self, cancellation_token: CancellationToken
    ) -> CogenticNextSpeaker:
        """Select the next speaker"""
        assert self._plan
        context = self._thread_to_context()

        # Create the next speaker prompt
        next_speaker_type = CogenticNextSpeaker.with_choices(
            choices=[
                topic_type
                for topic_type in self._participant_topic_types
                if topic_type not in self._plan.benched_team_members
            ]
        )
        next_speaker_prompt = create_next_speaker_prompt(
            names=self._participant_topic_types,
        )
        context.append(UserMessage(content=next_speaker_prompt, source=self._name))
        # Get the next speaker
        next_speaker = await reason_and_output_model(
            self._model_client,
            self._get_compatible_context(context),
            cancellation_token=cancellation_token,
            response_model=next_speaker_type,
            retries=self._max_json_retries,
        )
        assert isinstance(next_speaker, CogenticNextSpeaker)
        self.logger.debug(f"Next Speaker: {next_speaker}")

        return next_speaker

    async def _update_plan_on_stall(
        self, cancellation_token: CancellationToken
    ) -> None:
        """Update the facts and plan based on the current state."""

        assert self._plan and self._plan.current_hypothesis and self._fact_sheet

        update_plan_prompt = create_update_plan_on_stall_prompt(
            question=self._question,
            current_hypothesis=self._plan.current_hypothesis,
            team_description=self._team_description,
            fact_sheet=self._fact_sheet,
            plan=self._plan,
        )

        self._plan = await reason_and_output_model(
            self._model_client,
            self._get_compatible_context(
                [UserMessage(content=update_plan_prompt, source=self._name)]
            ),
            cancellation_token=cancellation_token,
            response_model=CogenticPlan,
            retries=self._max_json_retries,
        )
        assert isinstance(self._plan, CogenticPlan)

    async def _prepare_final_answer(
        self, reason: str, cancellation_token: CancellationToken
    ) -> None:
        """Prepare the final answer for the task."""

        assert self._fact_sheet and self._plan

        # Get the final answer
        final_answer_prompt = create_final_answer_prompt(
            question=self._question,
            finish_reason=reason,
            fact_sheet=self._fact_sheet,
            plan=self._plan,
        )
        messages: list[LLMMessage] = [
            UserMessage(content=final_answer_prompt, source=self._name)
        ]

        final_answer = await reason_and_output_model(
            self._model_client,
            messages=messages,
            cancellation_token=cancellation_token,
            response_model=CogenticFinalAnswer,
            retries=self._max_json_retries,
        )
        assert isinstance(final_answer, CogenticFinalAnswer)

        message = TextMessage(
            content=final_answer.model_dump_markdown(), source=self._name
        )
        self._message_thread.append(message)

        # Publish the response message
        await self._publish_to_output(
            message=message,
            cancellation_token=cancellation_token,
        )
        await self._publish_to_group(
            message=message,
            cancellation_token=cancellation_token,
        )

        # Terminate
        await self._terminate_chat(
            message=reason,
            cancellation_token=cancellation_token,
        )

    @event
    async def handle_agent_response(
        self, message: GroupChatAgentResponse, ctx: MessageContext
    ) -> None:
        """Handle the response from an agent in our group chat."""
        # Add this message to our ongoing hypothesis work thread
        self._message_thread.append(message.agent_response.chat_message)
        # Continue the work loop
        await self._hypothesis_loop(ctx.cancellation_token)

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

    async def validate_group_state(self, messages: List[ChatMessage] | None) -> None:
        pass

    async def save_state(self) -> Mapping[str, Any]:
        state = CogenticState(
            message_thread=list(self._message_thread),
            current_turn=self._current_turn,
            question=self._question,
            fact_sheet=self._fact_sheet,
            plan=self._plan,
            n_rounds=self._total_turns,
            n_stalls=self._current_stall_count,
        )
        return state.model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        orchestrator_state = CogenticState.model_validate(state)
        self._message_thread = orchestrator_state.message_thread
        self._current_turn = orchestrator_state.current_turn
        self._question = orchestrator_state.question
        self._fact_sheet = orchestrator_state.fact_sheet
        self._plan = orchestrator_state.plan
        self._current_ledger = orchestrator_state.current_ledger
        self._total_turns = orchestrator_state.n_rounds
        self._current_stall_count = orchestrator_state.n_stalls

    async def select_speaker(self, thread: List[AgentEvent | ChatMessage]) -> str:
        """Not used in this orchestrator, we select next speaker in _orchestrate_step."""
        return ""

    async def reset(self) -> None:
        """Reset the group chat manager."""
        self._message_thread.clear()
        self._total_turns = 0
        self._current_stall_count = 0
        self._question = ""
        self._fact_sheet = None
        self._plan = None
        self._current_ledger = None
        self._current_hypothesis_turns = 0
        self._current_test_turns = 0
