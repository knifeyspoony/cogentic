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
    SystemMessage,
    UserMessage,
)

from cogentic.orchestration.model_output import reason_and_output_model
from cogentic.orchestration.models.action import CogenticAction
from cogentic.orchestration.models.evidence import CogenticInitialEvidence
from cogentic.orchestration.models.hypothesis import CogenticInitialHypotheses
from cogentic.orchestration.models.ledger import CogenticProgressLedger
from cogentic.orchestration.models.orchestration import (
    CogenticFinalAnswer,
    CogenticHypothesisUpdate,
    CogenticNextStep,
    CogenticPlanUpdate,
)
from cogentic.orchestration.models.plan import CogenticPlan
from cogentic.orchestration.models.state import CogenticState
from cogentic.orchestration.prompts import (
    create_current_state_prompt,
    create_final_answer_prompt,
    create_initial_evidence_prompt,
    create_initial_hypotheses_prompt,
    create_next_step_prompt,
    create_persona_prompt,
    create_progress_ledger_prompt,
    create_summarize_result_prompt,
    create_update_hypothesis_on_completion_prompt,
    create_update_hypothesis_on_stall_prompt,
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
        json_model_client: ChatCompletionClient | None,
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
        self._ledger: CogenticProgressLedger | None = None
        self._active_step: CogenticNextStep | None = None
        self._total_turns: int = 0
        self._current_hypothesis_turns: int = 0
        self._current_test_turns: int = 0
        self._current_stall_count: int = 0
        self._summarized_thread: List[AgentEvent | ChatMessage] = []
        self.logger = logging.getLogger(TRACE_LOGGER_NAME)
        if json_model_client is None:
            self._json_model_client = json_model_client or model_client
        else:
            self._json_model_client = json_model_client

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

        - Gather initial evidence from the question itself
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

        # The planning conversation only exists to create a formal plan based on the question.
        # It is not broadcast to the group chat.
        planning_conversation: List[LLMMessage] = []

        self._plan = CogenticPlan()

        # Add our persona
        planning_conversation.append(SystemMessage(content=create_persona_prompt()))

        # Collect initial evidence from the question.
        planning_conversation.append(
            UserMessage(
                content=create_initial_evidence_prompt(self._question),
                source=self._name,
            )
        )
        initial_evidence = await reason_and_output_model(
            self._model_client,
            self._json_model_client,
            self._get_compatible_context(planning_conversation),
            ctx.cancellation_token,
            response_model=CogenticInitialEvidence,
            retries=self._max_json_retries,
        )
        self._plan.evidence.extend(initial_evidence.evidence)

        # Add the fact sheet to the planning conversation
        planning_conversation.append(
            AssistantMessage(
                content=initial_evidence.model_dump_markdown(title="Initial Evidence"),
                source=self._name,
            )
        )

        # Now, based on the question and the known facts, ask the model to create a plan
        planning_conversation.append(
            UserMessage(
                content=create_initial_hypotheses_prompt(self._team_description),
                source=self._name,
            )
        )
        initial_hypotheses = await reason_and_output_model(
            self._model_client,
            self._json_model_client,
            self._get_compatible_context(planning_conversation),
            ctx.cancellation_token,
            response_model=CogenticInitialHypotheses,
            retries=self._max_json_retries,
        )
        self._plan.hypotheses = initial_hypotheses.hypotheses

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
        assert self._plan

        if not self._plan.current_hypothesis:
            raise ValueError("No current hypothesis to process. This is unexpected.")

        # Clear the ledger
        self._ledger = None

        # Clear the orchestrator message thread
        self._message_thread.clear()
        self._summarized_thread.clear()

        # Add our persona to our thread
        persona_message = TextMessage(content=create_persona_prompt(), source="system")
        self._message_thread.append(persona_message)

        # Create the initial message for the group chat
        current_state_message = TextMessage(
            content=create_current_state_prompt(
                question=self._question,
                team_description=self._team_description,
                plan=self._plan,
            ),
            source=self._name,
        )
        # First message in any work thread is the state
        self._message_thread.append(current_state_message)
        self._summarized_thread.append(current_state_message)

        # Publish to the output and group
        await self._publish_to_output(message=current_state_message)
        await self._publish_to_group(message=current_state_message)

        # Start the hypothesis loop
        await self._hypothesis_loop(
            cancellation_token=cancellation_token, first_iteration=True
        )

    async def _hypothesis_loop(
        self, cancellation_token: CancellationToken, first_iteration=False
    ) -> None:
        """
        This is the work loop for the current hypothesis
        After each iteration, we update facts and current hypothesis data.

        Args:
            first_iteration (bool): Whether this is the first iteration of the loop (don't create ledger).
        """
        # Check if we have reached the maximum number of turns for the orchestrator.
        if self._max_turns is not None and self._total_turns > self._max_turns:
            await self._create_final_answer(
                f"Maximum turn count reached ({self._max_turns}). Can we come to a conclusion?",
                cancellation_token,
            )
            return

        self._total_turns += 1
        self._current_hypothesis_turns += 1
        self._current_test_turns += 1

        # We exit immediately on the first iteration - no need to create a progress ledger.
        if first_iteration:
            await self._execute_next_step(
                cancellation_token=cancellation_token,
            )
            return

        # Request an update to the ledger
        self._ledger = await self._update_progress_ledger(
            cancellation_token=cancellation_token
        )

        assert (
            self._plan
            and self._plan.current_hypothesis
            and self._plan.current_hypothesis.current_test
        )

        current_test = self._plan.current_hypothesis.current_test

        # Updates from the ledger
        current_test.state = self._ledger.test_state.answer
        self._plan.evidence.extend(self._ledger.new_test_evidence)
        self._plan.issues.extend(self._ledger.new_issues)

        # Check if we're totally done:
        if self._ledger.original_question_answered.answer:
            self.logger.info("Original question answered, preparing final answer...")
            await self._create_final_answer(
                self._ledger.original_question_answered.reason,
                cancellation_token,
            )
            return

        # Check if we're done the test/hypothesis
        if self._ledger.test_state.answer != "incomplete":
            self._current_test_turns = 0
            self.logger.info("Current test work complete.")

            # Replan on completed hypothesis
            if self._plan.current_hypothesis.all_tests_finished:
                return await self._replan(
                    cancellation_token=cancellation_token, stalled=False
                )

        # Check for stalling
        stalling = False
        if not self._ledger.forward_progress.answer:
            stalling = True
            self._current_stall_count += 1
        if self._ledger.stuck_in_loop.answer:
            stalling = True
            self._current_stall_count += 1

        if not stalling:
            # Decrement stall count if we're making progress
            self._current_stall_count = max(0, self._current_stall_count - 1)

        # Re-plan on full stall or max hypothesis turns reached
        if self._needs_replan():
            self.logger.warning(
                "Stalled or hypothesis turn count exceeded, time to update the plan."
            )
            return await self._replan(cancellation_token, stalled=True)

        # Keep going!
        await self._execute_next_step(
            cancellation_token=cancellation_token,
            next_step=self._ledger.next_step,
        )
        return

    async def _execute_next_step(
        self,
        cancellation_token: CancellationToken,
        next_step: CogenticNextStep | None = None,
    ) -> None:
        """Continue working by executing the next step.

        Args:
            next_step (CogenticNextStep | None): The next step to execute. If None, a new step will be created.
        """
        if not next_step:
            next_step = await self._create_next_step(cancellation_token)
        # Save this so we can summarize the results later
        self._active_step = next_step

        # Create the next step message and send it out to the group
        next_step_message = TextMessage(
            content=next_step.instruction_or_question.answer,
            source=self._name,
        )
        # Add it to our own internal conversation as well as our agents
        self._message_thread.append(next_step_message)
        self._summarized_thread.append(next_step_message)

        await self._publish_to_output(
            message=next_step_message, cancellation_token=cancellation_token
        )
        await self._publish_to_group(
            message=next_step_message, cancellation_token=cancellation_token
        )

        # Ask the next speaker to respond
        await self.publish_message(
            GroupChatRequestPublish(),
            topic_id=DefaultTopicId(type=next_step.next_speaker.answer),
            cancellation_token=cancellation_token,
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
        """Update the progress ledger based on the current state of the group chat.

        Returns:
            CogenticProgressLedger: An updated progress ledger for the current state of the group chat.
        """

        # Get the active conversation (this doesn't contain previous ledger updates, just instructions/results)
        context = self._thread_to_context(self._summarized_thread)
        ledger_type = CogenticProgressLedger.with_speakers(
            choices=self._participant_topic_types
        )
        progress_ledger_prompt = create_progress_ledger_prompt()
        context.append(UserMessage(content=progress_ledger_prompt, source=self._name))
        progress_ledger = await reason_and_output_model(
            self._model_client,
            self._json_model_client,
            self._get_compatible_context(context),
            cancellation_token=cancellation_token,
            response_model=ledger_type,
            retries=self._max_json_retries,
        )
        self.logger.debug(f"Progress Ledger: {progress_ledger}")
        return progress_ledger

    async def _create_next_step(
        self, cancellation_token: CancellationToken
    ) -> CogenticNextStep:
        """Select the next step"""
        assert self._plan

        # Next step selection is always done as part of a conversation
        context = self._thread_to_context(self._summarized_thread)

        # Create the next step prompt
        next_step_type = CogenticNextStep.with_speaker_choices(
            choices=self._participant_topic_types
        )
        next_step_prompt = create_next_step_prompt(
            names=self._participant_topic_types,
        )
        context.append(UserMessage(content=next_step_prompt, source=self._name))
        # Get the next step
        next_step = await reason_and_output_model(
            self._model_client,
            self._json_model_client,
            self._get_compatible_context(context),
            cancellation_token=cancellation_token,
            response_model=next_step_type,
            retries=self._max_json_retries,
        )
        self.logger.debug(f"Next Step: {next_step}")

        return next_step

    async def _replan(
        self, cancellation_token: CancellationToken, stalled=False
    ) -> None:
        """Update our plan according to the current state of the group chat.
        Args:
            stalled (bool): Whether we are doing this because we're stalled.
        """
        assert self._plan and self._plan.current_hypothesis

        persona = create_persona_prompt()
        current_state = create_current_state_prompt(
            question=self._question,
            team_description=self._team_description,
            plan=self._plan,
        )
        if stalled:
            update_hypothesis_prompt = create_update_hypothesis_on_stall_prompt()
            plan_update_prompt = create_update_plan_on_stall_prompt()
        else:
            update_hypothesis_prompt = create_update_hypothesis_on_completion_prompt()
            plan_update_prompt = create_update_plan_on_completion_prompt()

        messages = [
            SystemMessage(content=persona),
            UserMessage(content=current_state, source=self._name),
            UserMessage(content=update_hypothesis_prompt, source=self._name),
        ]

        hypothesis_update = await reason_and_output_model(
            self._model_client,
            self._json_model_client,
            self._get_compatible_context(messages),
            cancellation_token=cancellation_token,
            response_model=CogenticHypothesisUpdate,
            retries=self._max_json_retries,
        )
        # Add new tests to the hypothesis.
        self._plan.current_hypothesis.tests.extend(hypothesis_update.new_tests)
        # Update the state of the hypothesis.
        # NOTE: if this sets the hypothesis to anything but unverified it will change future results of self._plan.current_hypothesis!
        self._plan.current_hypothesis.state = hypothesis_update.hypothesis_state.answer

        # We won't update the plan if we aren't stalled and there's work left to do
        if not stalled and self._plan.current_hypothesis:
            return await self._process_next_hypothesis(cancellation_token)

        # Otherwise, we need to update the plan
        messages = [
            SystemMessage(content=persona),
            UserMessage(content=current_state, source=self._name),
            UserMessage(content=plan_update_prompt, source=self._name),
        ]

        # Get the plan update
        plan_update = await reason_and_output_model(
            self._model_client,
            self._json_model_client,
            self._get_compatible_context(messages),
            cancellation_token=cancellation_token,
            response_model=CogenticPlanUpdate,
            retries=self._max_json_retries,
        )
        # Update the plan state
        self._plan.state = plan_update.plan_state.answer
        if self._plan.state != "in_progress":
            # If the plan is complete, we need to prepare the final answer
            self.logger.info("Plan complete, preparing final answer...")
            await self._create_final_answer(
                "No work remaining. This could mean success or failure depending on the results",
                cancellation_token,
            )
            return
        else:
            # Add new hypotheses to the plan
            self._plan.hypotheses.extend(plan_update.new_hypotheses)
            self.logger.info("Plan still in progress, selecting next hypothesis...")
            return await self._process_next_hypothesis(cancellation_token)

    async def _create_final_answer(
        self, reason: str, cancellation_token: CancellationToken
    ) -> None:
        """Prepare the final answer for the task."""

        assert self._plan

        # Create the final answer prompt
        persona = create_persona_prompt()
        final_answer_prompt = create_final_answer_prompt(
            question=self._question,
            finish_reason=reason,
            plan=self._plan,
        )
        messages = [
            SystemMessage(content=persona),
            UserMessage(content=final_answer_prompt, source=self._name),
        ]

        final_answer = await reason_and_output_model(
            self._model_client,
            self._json_model_client,
            messages=self._get_compatible_context(messages),
            cancellation_token=cancellation_token,
            response_model=CogenticFinalAnswer,
            retries=self._max_json_retries,
        )

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
        # Add this message to our ongoing work thread
        self._message_thread.append(message.agent_response.chat_message)
        # Summarize what happened for our plan history
        await self._summarize_action(
            message.agent_response.chat_message, ctx.cancellation_token
        )
        # Continue the work loop
        await self._hypothesis_loop(ctx.cancellation_token)

    async def _summarize_action(
        self, message: ChatMessage, cancellation_token: CancellationToken
    ):
        """Summarize the action we just took

        Args:
            message (GroupChatAgentResponse): The response from the agent.
            cancellation_token (CancellationToken): The cancellation token for the operation.

        """
        if not isinstance(message.content, str):
            return

        assert self._active_step
        assert self._plan
        assert self._plan.current_hypothesis
        assert self._plan.current_hypothesis.current_test

        # Get the action summary
        action_summary_prompt = create_summarize_result_prompt(
            response=message.content,
        )
        action_summary_response = await self._model_client.create(
            messages=[
                UserMessage(content=action_summary_prompt, source=self._name),
            ],
            cancellation_token=cancellation_token,
        )
        assert isinstance(action_summary_response.content, str)
        action_summary = action_summary_response.content
        # Create the action
        action = CogenticAction(
            goal=self._active_step.goal.answer,
            team_member_name=self._active_step.next_speaker.answer,
            test_name=self._plan.current_hypothesis.current_test.name,
            outcome=action_summary,
        )
        # Add the action to the plan
        self._plan.actions.append(action)
        # Store the result in our summarized thread:
        self._summarized_thread.append(
            TextMessage(
                content=action_summary,
                source=self._active_step.next_speaker.answer,
            )
        )

    def _thread_to_context(
        self, thread: list[AgentEvent | ChatMessage]
    ) -> List[LLMMessage]:
        """Convert the message thread to a context for the model."""
        context: List[LLMMessage] = []
        for m in thread:
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
            plan=self._plan,
            total_turns=self._total_turns,
            stalls=self._current_stall_count,
        )
        return state.model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        orchestrator_state = CogenticState.model_validate(state)
        self._message_thread = orchestrator_state.message_thread
        self._current_turn = orchestrator_state.current_turn
        self._question = orchestrator_state.question
        self._plan = orchestrator_state.plan
        self._total_turns = orchestrator_state.total_turns
        self._current_stall_count = orchestrator_state.stalls

    async def select_speaker(self, thread: List[AgentEvent | ChatMessage]) -> str:
        """Not used in this orchestrator, we select next speaker in _orchestrate_step."""
        return ""

    async def reset(self) -> None:
        """Reset the group chat manager."""
        self._message_thread.clear()
        self._total_turns = 0
        self._current_stall_count = 0
        self._question = ""
        self._plan = None
        self._ledger = None
        self._current_hypothesis_turns = 0
        self._current_test_turns = 0
