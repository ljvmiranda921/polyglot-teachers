import logging
import sys

from langcodes import standardize_tag
from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric
from lighteval.metrics.metrics_sample import JudgeLLM
from lighteval.metrics.normalizations import LogProbCharNorm  # fmt: skip
from lighteval.metrics.normalizations import LogProbPMINorm, LogProbTokenNorm
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import MCFFormulation
from lighteval.utils.language import Language

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


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


# IGNORE EVERYTHING ABOVE. I THINK THIS IS JUST MCQ. what the others did is that an LM made a response, and then it's judged.
# FOR US IT"S JUST MCQ!! SO SIMPLE! don't do the things above!!

M_REWARDBENCH = [
    LightevalTaskConfig(
        name=f"mrewardbench:{standardize_tag(language.value)}",
        prompt_function=get_mcq_prompt_function(
            language,
            lambda line: {
                # TODO: follow this https://github.com/filbench/lighteval/blob/main/filbench/sib200.py
            },
        ),
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
