# import json
from pathlib import Path

from cogentic.orchestration.models import (
    CogenticFactSheet,
    CogenticHypothesis,
    CogenticPlan,
)

PROMPTS_DIR = Path(__file__).parent

INITIAL_QUESTION_PROMPT_PATH = PROMPTS_DIR / "initial_question.md"
INITIAL_QUESTION_PROMPT = INITIAL_QUESTION_PROMPT_PATH.read_text()


def create_initial_question_prompt(question: str) -> str:
    return INITIAL_QUESTION_PROMPT.format(question=question)


INITIAL_PLAN_PROMPT_PATH = PROMPTS_DIR / "initial_plan.md"
INITIAL_PLAN_PROMPT = INITIAL_PLAN_PROMPT_PATH.read_text()


def create_initial_plan_prompt(team_description: str) -> str:
    return INITIAL_PLAN_PROMPT.format(
        team_description=team_description,
    )


HYPOTHESIS_PROMPT_PATH = PROMPTS_DIR / "hypothesis.md"
HYPOTHESIS_PROMPT = HYPOTHESIS_PROMPT_PATH.read_text()


def create_hypothesis_prompt(
    team_description: str,
    fact_sheet: CogenticFactSheet,
    current_hypothesis: CogenticHypothesis,
) -> str:
    return HYPOTHESIS_PROMPT.format(
        current_hypothesis=current_hypothesis.model_dump_markdown(),
        team_description=team_description,
        fact_sheet=fact_sheet.model_dump_markdown(),
    )


PROGRESS_LEDGER_PROMPT_PATH = PROMPTS_DIR / "progress_ledger.md"
PROGRESS_LEDGER_PROMPT = PROGRESS_LEDGER_PROMPT_PATH.read_text()


def create_progress_ledger_prompt(
    question: str, team_description: str, names: list[str]
) -> str:
    return PROGRESS_LEDGER_PROMPT.format(
        question=question,
        team_description=team_description,
        names="\n".join(names),
    )


FINAL_ANSWER_PROMPT_PATH = PROMPTS_DIR / "final_answer.md"
FINAL_ANSWER_PROMPT = FINAL_ANSWER_PROMPT_PATH.read_text()


def create_final_answer_prompt(question: str) -> str:
    return FINAL_ANSWER_PROMPT.format(question=question)


UPDATE_FACTS_ON_STALL_PROMPT_PATH = PROMPTS_DIR / "update_facts_on_stall.md"
UPDATE_FACTS_ON_STALL_PROMPT = UPDATE_FACTS_ON_STALL_PROMPT_PATH.read_text()


def create_update_facts_on_stall_prompt(
    current_hypothesis: CogenticHypothesis, fact_sheet: CogenticFactSheet
) -> str:
    return UPDATE_FACTS_ON_STALL_PROMPT.format(
        current_hypothesis=current_hypothesis.model_dump_markdown(),
        fact_sheet=fact_sheet.model_dump_markdown(),
    )


UPDATE_FACTS_ON_COMPLETION_PROMPT_PATH = PROMPTS_DIR / "update_facts_on_completion.md"
UPDATE_FACTS_ON_COMPLETION_PROMPT = UPDATE_FACTS_ON_COMPLETION_PROMPT_PATH.read_text()


def create_update_facts_on_completion_prompt(
    current_hypothesis: CogenticHypothesis, fact_sheet: CogenticFactSheet
) -> str:
    return UPDATE_FACTS_ON_COMPLETION_PROMPT.format(
        current_hypothesis=current_hypothesis.model_dump_markdown(),
        fact_sheet=fact_sheet.model_dump_markdown(),
    )


UPDATE_PLAN_ON_STALL_PROMPT_PATH = PROMPTS_DIR / "update_plan_on_stall.md"
UPDATE_PLAN_ON_STALL_PROMPT = UPDATE_PLAN_ON_STALL_PROMPT_PATH.read_text()


def create_update_plan_on_stall_prompt(
    question: str,
    current_hypothesis: CogenticHypothesis,
    team_description: str,
    fact_sheet: CogenticFactSheet,
    plan: CogenticPlan,
) -> str:
    return UPDATE_PLAN_ON_STALL_PROMPT.format(
        question=question,
        current_hypothesis=current_hypothesis.model_dump_markdown(),
        team_description=team_description,
        fact_sheet=fact_sheet.model_dump_markdown(),
        plan=plan.model_dump_markdown(),
    )


UPDATE_PLAN_ON_COMPLETION_PROMPT_PATH = PROMPTS_DIR / "update_plan_on_completion.md"
UPDATE_PLAN_ON_COMPLETION_PROMPT = UPDATE_PLAN_ON_COMPLETION_PROMPT_PATH.read_text()


def create_update_plan_on_completion_prompt(
    question: str,
    current_hypothesis: CogenticHypothesis,
    team_description: str,
    fact_sheet: CogenticFactSheet,
    plan: CogenticPlan,
) -> str:
    return UPDATE_PLAN_ON_COMPLETION_PROMPT.format(
        question=question,
        current_hypothesis=current_hypothesis.model_dump_markdown(),
        team_description=team_description,
        fact_sheet=fact_sheet.model_dump_markdown(),
        plan=plan.model_dump_markdown(),
    )
