# import json
from pathlib import Path

from rigorous.orchestration.state import (  # RigorousProgressLedger,; RigorousReasonedBooleanAnswer,; RigorousReasonedChoiceAnswer,; RigorousReasonedStringAnswer,; RigorousTest,; RigorousEvidence,
    RigorousFactSheet,
    RigorousHypothesis,
    RigorousPlan,
)

PROMPTS_DIR = Path(__file__).parent

INITIAL_QUESTION_PROMPT_PATH = PROMPTS_DIR / "initial_question.md"
INITIAL_QUESTION_PROMPT = INITIAL_QUESTION_PROMPT_PATH.read_text()


def create_initial_question_prompt(question: str) -> str:
    return INITIAL_QUESTION_PROMPT.format(question=question)


# EXAMPLE_PLAN = RigorousPlan(
#     hypotheses=[
#         RigorousHypothesis(
#             hypothesis="The sky is blue.",
#             state="verified",
#             tests=[
#                 RigorousTest(
#                     description="Perform a wikipedia search for the topic 'sky' and check if the sky is blue.",
#                     completed=True,
#                     result="The daytime sky appears blue. The night sky is black, so we will be sure to be explicit that we are referencing the daytime sky.",
#                     supporting_evidence=[
#                         RigorousEvidence(
#                             source="https://en.wikipedia.org/wiki/Sky",
#                             content="The daytime sky appears blue because air molecules scatter shorter wavelengths of sunlight more than longer ones (redder light).",
#                         )
#                     ],
#                 )
#             ],
#         ),
#         RigorousHypothesis(
#             hypothesis="The sky is blue due to a known scientific phenomenon.",
#             state="unverified",
#             tests=[
#                 RigorousTest(
#                     description="Perform a wikipedia search for the topic 'sky' and look for the scientific phenomenon that causes the sky to be blue.",
#                     result=None,
#                     completed=False,
#                     supporting_evidence=None,
#                 )
#             ],
#         ),
#     ]
# )

INITIAL_PLAN_PROMPT_PATH = PROMPTS_DIR / "initial_plan.md"
INITIAL_PLAN_PROMPT = INITIAL_PLAN_PROMPT_PATH.read_text()


def create_initial_plan_prompt(team_description: str) -> str:
    return INITIAL_PLAN_PROMPT.format(
        team_description=team_description,
        # example_plan=EXAMPLE_PLAN.model_dump_markdown(),
    )


HYPOTHESIS_PROMPT_PATH = PROMPTS_DIR / "hypothesis.md"
HYPOTHESIS_PROMPT = HYPOTHESIS_PROMPT_PATH.read_text()


def create_hypothesis_prompt(
    team_description: str,
    fact_sheet: RigorousFactSheet,
    current_hypothesis: RigorousHypothesis,
) -> str:
    return HYPOTHESIS_PROMPT.format(
        current_hypothesis=current_hypothesis.model_dump_markdown(),
        team_description=team_description,
        fact_sheet=fact_sheet.model_dump_markdown(),
    )


PROGRESS_LEDGER_PROMPT_PATH = PROMPTS_DIR / "progress_ledger.md"
PROGRESS_LEDGER_PROMPT = PROGRESS_LEDGER_PROMPT_PATH.read_text()

# ledger_type = RigorousProgressLedger.with_speakers(
#     names=["CoderAssistant", "WebSearchAssistant"]
# )
# PROGRESS_LEDGER_EXAMPLE = ledger_type(
#     is_request_satisfied=RigorousReasonedBooleanAnswer(
#         reason="The question is not yet fully answered.", answer=False
#     ),
#     is_current_hypothesis_work_complete=RigorousReasonedBooleanAnswer(
#         reason="The current hypothesis work is not yet complete.", answer=False
#     ),
#     is_in_loop=RigorousReasonedBooleanAnswer(
#         reason="The conversation is not stuck in a loop.", answer=False
#     ),
#     is_progress_being_made=RigorousReasonedBooleanAnswer(
#         reason="The conversation is making progress.", answer=True
#     ),
#     next_speaker={
#         "reason": "The next speaker should be the WebSearchAssistant, so they can search wikipedia for the topic 'sky' and look for the scientific phenomenon that causes the sky to be blue.",
#         "answer": "WebSearchAssistant",
#     },
#     instruction_or_question=RigorousReasonedStringAnswer(
#         reason="We want to determine if there is a scientific phenomenon that causes the sky to be blue.",
#         answer="Perform a wikipedia search for the topic 'sky' and look for the scientific phenomenon that causes the sky to be blue.",
#     ),
# )


def create_progress_ledger_prompt(
    question: str, team_description: str, names: list[str]
) -> str:
    return PROGRESS_LEDGER_PROMPT.format(
        question=question,
        team_description=team_description,
        names="\n".join(names),
        # ledger_format=PROGRESS_LEDGER_EXAMPLE.model_dump_markdown(),
    )


FINAL_ANSWER_PROMPT_PATH = PROMPTS_DIR / "final_answer.md"
FINAL_ANSWER_PROMPT = FINAL_ANSWER_PROMPT_PATH.read_text()


def create_final_answer_prompt(question: str) -> str:
    return FINAL_ANSWER_PROMPT.format(question=question)


UPDATE_FACTS_ON_STALL_PROMPT_PATH = PROMPTS_DIR / "update_facts_on_stall.md"
UPDATE_FACTS_ON_STALL_PROMPT = UPDATE_FACTS_ON_STALL_PROMPT_PATH.read_text()


def create_update_facts_on_stall_prompt(
    current_hypothesis: RigorousHypothesis, fact_sheet: RigorousFactSheet
) -> str:
    return UPDATE_FACTS_ON_STALL_PROMPT.format(
        current_hypothesis=current_hypothesis.model_dump_markdown(),
        fact_sheet=fact_sheet.model_dump_markdown(),
    )


UPDATE_FACTS_ON_COMPLETION_PROMPT_PATH = PROMPTS_DIR / "update_facts_on_completion.md"
UPDATE_FACTS_ON_COMPLETION_PROMPT = UPDATE_FACTS_ON_COMPLETION_PROMPT_PATH.read_text()


def create_update_facts_on_completion_prompt(
    current_hypothesis: RigorousHypothesis, fact_sheet: RigorousFactSheet
) -> str:
    return UPDATE_FACTS_ON_COMPLETION_PROMPT.format(
        current_hypothesis=current_hypothesis.model_dump_markdown(),
        fact_sheet=fact_sheet.model_dump_markdown(),
    )


UPDATE_PLAN_ON_STALL_PROMPT_PATH = PROMPTS_DIR / "update_plan_on_stall.md"
UPDATE_PLAN_ON_STALL_PROMPT = UPDATE_PLAN_ON_STALL_PROMPT_PATH.read_text()


def create_update_plan_on_stall_prompt(
    question: str,
    current_hypothesis: RigorousHypothesis,
    team_description: str,
    fact_sheet: RigorousFactSheet,
    plan: RigorousPlan,
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
    current_hypothesis: RigorousHypothesis,
    team_description: str,
    fact_sheet: RigorousFactSheet,
    plan: RigorousPlan,
) -> str:
    return UPDATE_PLAN_ON_COMPLETION_PROMPT.format(
        question=question,
        current_hypothesis=current_hypothesis.model_dump_markdown(),
        team_description=team_description,
        fact_sheet=fact_sheet.model_dump_markdown(),
        plan=plan.model_dump_markdown(),
    )
