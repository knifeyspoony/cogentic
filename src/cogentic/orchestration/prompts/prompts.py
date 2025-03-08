# import json
from pathlib import Path

from cogentic.orchestration.models.plan import CogenticPlan

PROMPTS_DIR = Path(__file__).parent

PERSONA_PROMPT_PATH = PROMPTS_DIR / "persona.md"
PERSONA_PROMPT = PERSONA_PROMPT_PATH.read_text()


def create_persona_prompt() -> str:
    return PERSONA_PROMPT


INITIAL_EVIDENCE_PROMPT_PATH = PROMPTS_DIR / "initial_evidence.md"
INITIAL_EVIDENCE_PROMPT = INITIAL_EVIDENCE_PROMPT_PATH.read_text()


def create_initial_evidence_prompt(
    question: str,
) -> str:
    return INITIAL_EVIDENCE_PROMPT.format(
        question=question,
    )


INITIAL_HYPOTHESES_PROMPT_PATH = PROMPTS_DIR / "initial_hypotheses.md"
INITIAL_HYPOTHESES_PROMPT = INITIAL_HYPOTHESES_PROMPT_PATH.read_text()


def create_initial_hypotheses_prompt(team_description: str) -> str:
    return INITIAL_HYPOTHESES_PROMPT.format(
        team_description=team_description,
    )


CURRENT_STATE_PROMPT_PATH = PROMPTS_DIR / "current_state.md"
CURRENT_STATE_PROMPT = CURRENT_STATE_PROMPT_PATH.read_text()


def create_current_state_prompt(
    question: str,
    team_description: str,
    plan: CogenticPlan,
) -> str:
    current_test = "No tests have work remaining."
    current_hypothesis = "No hypotheses have work remaining."
    if plan.current_hypothesis:
        if plan.current_hypothesis.current_test:
            current_test = plan.current_hypothesis.current_test.model_dump_markdown()
        current_hypothesis = plan.current_hypothesis.model_dump_markdown()
    return CURRENT_STATE_PROMPT.format(
        question=question,
        current_hypothesis=current_hypothesis,
        current_test=current_test,
        team_description=team_description,
        evidence=plan.model_dump_field_as_markdown("evidence"),
        issues=plan.model_dump_field_as_markdown("issues"),
    )


CREATE_PROGRESS_LEDGER_PROMPT_PATH = PROMPTS_DIR / "create_progress_ledger.md"
CREATE_PROGRESS_LEDGER_PROMPT = CREATE_PROGRESS_LEDGER_PROMPT_PATH.read_text()


def create_progress_ledger_prompt() -> str:
    return CREATE_PROGRESS_LEDGER_PROMPT


FINAL_ANSWER_PROMPT_PATH = PROMPTS_DIR / "final_answer.md"
FINAL_ANSWER_PROMPT = FINAL_ANSWER_PROMPT_PATH.read_text()


def create_final_answer_prompt(
    question: str,
    finish_reason: str,
    plan: CogenticPlan,
) -> str:
    return FINAL_ANSWER_PROMPT.format(
        question=question,
        finish_reason=finish_reason,
        plan=plan.model_dump_markdown(),
    )


UPDATE_HYPOTHESIS_ON_STALL_PROMPT_PATH = PROMPTS_DIR / "update_hypothesis_on_stall.md"
UPDATE_HYPOTHESIS_ON_STALL_PROMPT = UPDATE_HYPOTHESIS_ON_STALL_PROMPT_PATH.read_text()


def create_update_hypothesis_on_stall_prompt() -> str:
    return UPDATE_HYPOTHESIS_ON_STALL_PROMPT


UPDATE_HYPOTHESIS_ON_COMPLETION_PROMPT_PATH = (
    PROMPTS_DIR / "update_hypothesis_on_completion.md"
)
UPDATE_HYPOTHESIS_ON_COMPLETION_PROMPT = (
    UPDATE_HYPOTHESIS_ON_COMPLETION_PROMPT_PATH.read_text()
)


def create_update_hypothesis_on_completion_prompt() -> str:
    return UPDATE_HYPOTHESIS_ON_COMPLETION_PROMPT


UPDATE_PLAN_ON_COMPLETION_PROMPT_PATH = PROMPTS_DIR / "update_plan_on_completion.md"
UPDATE_PLAN_ON_COMPLETION_PROMPT = UPDATE_PLAN_ON_COMPLETION_PROMPT_PATH.read_text()


def create_update_plan_on_completion_prompt() -> str:
    """Update the plan when we have completed work on a hypothesis."""
    return UPDATE_PLAN_ON_COMPLETION_PROMPT


UPDATE_PLAN_ON_STALL_PROMPT_PATH = PROMPTS_DIR / "update_plan_on_stall.md"
UPDATE_PLAN_ON_STALL_PROMPT = UPDATE_PLAN_ON_STALL_PROMPT_PATH.read_text()


def create_update_plan_on_stall_prompt() -> str:
    """Update the plan when we have stalled on a hypothesis."""
    return UPDATE_PLAN_ON_STALL_PROMPT


NEXT_STEP_PROMPT_PATH = PROMPTS_DIR / "next_step.md"
NEXT_STEP_PROMPT = NEXT_STEP_PROMPT_PATH.read_text()


def create_next_step_prompt(
    names: list[str],
) -> str:
    return NEXT_STEP_PROMPT.format(
        names=", ".join(names),
    )


SUMMARIZE_RESULT_PROMPT_PATH = PROMPTS_DIR / "summarize_result.md"
SUMMARIZE_RESULT_PROMPT = SUMMARIZE_RESULT_PROMPT_PATH.read_text()


def create_summarize_result_prompt(
    response: str,
) -> str:
    response = response or "No response provided."

    return SUMMARIZE_RESULT_PROMPT.format(
        response=response,
    )
