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


def get_judge_prompt_mrewardbench(question, answer, gold, **kwargs):
    MREWARDBENCH_TEMPLATE = ""
    content = MREWARDBENCH_TEMPLATE.format(
        question=question,
        target=gold,
        predicted_answer=answer,
    )
    return [{"role": "user", "content": content}]


def process_judge_response_mrewardbench(response: str) -> float:
    if response == "A":
        return 1.0
    elif response == "B":
        return 0.0
    else:
        logging.warning(f"Unknown response from judge: {response}")
        return 0.0


class JudgeMRewardBench(JudgeLLM):
    def __init__(self):
        # TODO: update this. use as reference: https://github.com/huggingface/lighteval/blob/main/src/lighteval/metrics/metrics_sample.py#L1011
        super().__init__(
            judge_model_name="gpt-4o-2024-08-06",
            template=get_judge_prompt_mrewardbench,
            process_judge_response=process_judge_response_mrewardbench,
            judge_backend="openai",
            short_judge_name="gpt4o",
        )

    def compute(
        self, responses: list[ModelResponse], docs: list[Doc], **kwargs
    ) -> list:
        # TODO
        pass


# the judge is then passed on to the SampleMetric: https://github.com/huggingface/lighteval/blob/99ef5b98d422cf3620eebec9db13285493d35542/src/lighteval/metrics/metrics.py#L553C1-L562C6
# which is then passed on to the metrics in the LightEvaltask config: https://github.com/huggingface/lighteval/blob/99ef5b98d422cf3620eebec9db13285493d35542/src/lighteval/tasks/tasks/mt_bench/main.py#L82
# but it's kinda weird because you see something like this: https://github.com/huggingface/lighteval/blob/99ef5b98d422cf3620eebec9db13285493d35542/src/lighteval/tasks/tasks/mt_bench/main.py#L38

M_REWARDBENCH = []


TASKS_TABLE: list[LightevalTaskConfig] = GLOBAL_MMLU_LITE
