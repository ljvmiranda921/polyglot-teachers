import logging
import sys
from typing import Any
import random

from langcodes import standardize_tag
from langcodes import Language as LangCodeLanguage
from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric
from lighteval.metrics.normalizations import LogProbCharNorm  # fmt: skip
from lighteval.metrics.normalizations import LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import MCFFormulation
from lighteval.utils.language import Language

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


# ==== Global-MMU-Lite ====

GLOBAL_MMLU_LITE = [
    LightevalTaskConfig(
        name=f"global_mmlu_lite:{standardize_tag(language.value)}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": [
                    line["option_a"],
                    line["option_b"],
                    line["option_c"],
                    line["option_d"],
                ],
                "gold_idx": LETTER_INDICES.index(line["answer"]),
            },
            formulation=MCFFormulation(),
        ),
        suite=("lighteval",),
        hf_repo="CohereForAI/Global-MMLU-Lite",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(
            MCFFormulation(),
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for language in [
        Language.ARABIC,
        Language.GERMAN,
        Language.SPANISH,
        Language.INDONESIAN,
        Language.JAPANESE,
    ]
]


# from last night LJ to morning LJ: IGNORE EVERYTHING ABOVE. I THINK THIS IS JUST MCQ. what the others did is that an LM made a response, and then it's judged.
# FOR US IT"S JUST MCQ!! SO SIMPLE! ignore everything above!!

# ==== M-RewardBench ====


def get_mrewardbench_eval_instances(line: dict) -> dict[str, Any]:
    """Returns an eval instance of M-RewardBench containing the question, choices, and gold index."""
    # Reference: https://github.com/Cohere-Labs-Community/m-rewardbench/blob/main/scripts/generative.py#L218
    PROMPT_TEMPLATE = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "  # noqa
        "The question provided is in {src_lang}. "  # noqa
        "You should choose the assistant that follows the user's instructions and answers the user's question better. "  # noqa
        "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. "  # noqa
        "Also, make sure that the assistant responses are in {tgt_lang}. "  # noqa
        "Begin your evaluation by comparing the two responses and provide a short explanation. "  # noqa
        "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. "  # noqa
        "Do not allow the length of the responses to influence your evaluation. "  # noqa
        "Do not favor certain names of the assistants. "  # noqa
        "Be as objective as possible. "  # noqa
        "After providing your explanation, output your final verdict by strictly following this format: "  # noqa
        '"A" if assistant A is better, "B" if assistant B is better.'  # noqa, removed tie option as , and \"[[C]]\ " for a tie
        "Don't put any quotation marks around your final verdict. "  # noqa
        "Here is the user question: {question} "  # noqa
        "Here is assistant A's response: {assistant_a_response} "  # noqa
        "Here is assistant B's response: {assistant_b_response} "  # noqa
    )

    chosen_response = line["chosen"]
    rejected_response = line["rejected"]

    PROMPT_TEMPLATE.format(
        src_lang=line["source_language"],
        tgt_lang=line["target_language"],
        question=line["prompt"],
        assistant_a_response=chosen_response,
        assistant_b_response=rejected_response,
    )

    return {"question": question, "choices": choices, "gold_idx": gold_idx}


def iso2_to_extended_format(iso2_code: str) -> str:
    """Convert ISO 639-1 (2-letter) language code to extended format with script."""
    lang = LangCodeLanguage.get(iso2_code)
    maximized = lang.maximize()
    alpha3 = lang.to_alpha3()
    script = maximized.script
    return f"{alpha3}_{script}" if script else alpha3


M_REWARDBENCH = [
    LightevalTaskConfig(
        name=f"mrewardbench:{standardize_tag(language.value)}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                # TODO: follow this https://github.com/filbench/lighteval/blob/main/filbench/sib200.py
            },
            formulation=MCFFormulation(),
        ),
        suite=("lighteval",),
        hf_repo="CohereLabsCommunity/multilingual-reward-bench",
        hf_subset=iso2_to_extended_format(standardize_tag(language.value)),
        evaluation_splits=("test",),
        few_shots_split="test",
    )
    for language in [
        Language.ARABIC,
        Language.CZECH,
        Language.GERMAN,
        Language.SPANISH,
        Language.INDONESIAN,
        Language.JAPANESE,
    ]
]


TASKS_TABLE: list[LightevalTaskConfig] = GLOBAL_MMLU_LITE
