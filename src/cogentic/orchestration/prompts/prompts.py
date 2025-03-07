# import json
from pathlib import Path

from cogentic.orchestration.models import (
    CogenticFactSheet,
    CogenticHypothesis,
    CogenticPlan,
    CogenticProgressLedger,
    CogenticTest,
)

PROMPTS_DIR = Path(__file__).parent

PERSONA_PROMPT_PATH = PROMPTS_DIR / "persona.md"
PERSONA_PROMPT = PERSONA_PROMPT_PATH.read_text()

INITIAL_FACT_SHEET_PROMPT_PATH = PROMPTS_DIR / "initial_fact_sheet.md"
INITIAL_FACT_SHEET_PROMPT = INITIAL_FACT_SHEET_PROMPT_PATH.read_text()


def create_initial_fact_sheet_prompt(
    question: str,
) -> str:
    return INITIAL_FACT_SHEET_PROMPT.format(
        question=question,
    )


INITIAL_PLAN_PROMPT_PATH = PROMPTS_DIR / "initial_plan.md"
INITIAL_PLAN_PROMPT = INITIAL_PLAN_PROMPT_PATH.read_text()


def create_initial_plan_prompt(team_description: str) -> str:
    return INITIAL_PLAN_PROMPT.format(
        team_description=team_description,
    )


HYPOTHESIS_PROMPT_PATH = PROMPTS_DIR / "hypothesis.md"
HYPOTHESIS_PROMPT = HYPOTHESIS_PROMPT_PATH.read_text()


def create_hypothesis_prompt(
    question: str,
    team_description: str,
    fact_sheet: CogenticFactSheet,
    current_hypothesis: CogenticHypothesis,
) -> str:
    if current_hypothesis.current_test:
        current_test = current_hypothesis.current_test.model_dump_markdown()
    else:
        current_test = "All tests have been completed."
    return (
        PERSONA_PROMPT
        + "\n\n"
        + HYPOTHESIS_PROMPT.format(
            question=question,
            current_hypothesis=current_hypothesis.model_dump_markdown(),
            current_test=current_test,
            team_description=team_description,
            fact_sheet=fact_sheet.model_dump_markdown(),
        )
    )


PROGRESS_LEDGER_PROMPT_PATH = PROMPTS_DIR / "progress_ledger.md"
PROGRESS_LEDGER_PROMPT = PROGRESS_LEDGER_PROMPT_PATH.read_text()


def create_progress_ledger_prompt(
    current_test: CogenticTest,
    progress_ledger: CogenticProgressLedger | None = None,
) -> str:
    previous_ledger = (
        progress_ledger.model_dump_markdown()
        if progress_ledger
        else "No existing progress ledger. Create one from scratch."
    )
    return PROGRESS_LEDGER_PROMPT.format(
        current_test=current_test.model_dump_markdown(),
        previous_ledger=previous_ledger,
    )


FINAL_ANSWER_PROMPT_PATH = PROMPTS_DIR / "final_answer.md"
FINAL_ANSWER_PROMPT = FINAL_ANSWER_PROMPT_PATH.read_text()


def create_final_answer_prompt(
    question: str, finish_reason: str, fact_sheet: CogenticFactSheet, plan: CogenticPlan
) -> str:
    return FINAL_ANSWER_PROMPT.format(
        question=question,
        finish_reason=finish_reason,
        fact_sheet=fact_sheet.model_dump_markdown(),
        plan=plan.model_dump_markdown(),
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
    assert current_hypothesis.current_test is not None, "Current test must be set."
    hypothesis_prompt = create_hypothesis_prompt(
        question=question,
        team_description=team_description,
        fact_sheet=fact_sheet,
        current_hypothesis=current_hypothesis,
    )
    return (
        hypothesis_prompt
        + "\n\n"
        + UPDATE_PLAN_ON_STALL_PROMPT.format(
            plan=plan.model_dump_markdown(),
        )
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
    hypothesis_prompt = create_hypothesis_prompt(
        question=question,
        team_description=team_description,
        fact_sheet=fact_sheet,
        current_hypothesis=current_hypothesis,
    )
    return (
        hypothesis_prompt
        + "\n\n"
        + UPDATE_PLAN_ON_COMPLETION_PROMPT.format(
            plan=plan.model_dump_markdown(),
        )
    )


NEXT_SPEAKER_PROMPT_PATH = PROMPTS_DIR / "next_speaker.md"
NEXT_SPEAKER_PROMPT = NEXT_SPEAKER_PROMPT_PATH.read_text()


def create_next_speaker_prompt(
    names: list[str],
) -> str:
    return NEXT_SPEAKER_PROMPT.format(
        names=", ".join(names),
    )
