from functools import partial

from langcodes import standardize_tag
from lighteval.metrics.dynamic_metrics import loglikelihood_acc_metric
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm  # fmt: skip
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.tasks import MMLU_SUBSETS
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, HybridFormulation, MCFFormulation # fmt: skip
from lighteval.utils.language import Language


GLOBAL_MMLU_LITE = [
    LightevalTaskConfig(
        name=f"global_mmlu_{sensitivity_label.lower()}_{language.value}_{formulation.name.lower()}:{subset}",
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
            formulation=formulation,
        ),
        suite=("community",),
        hf_repo="CohereForAI/Global-MMLU-Lite",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split="dev",
        hf_filter=partial(
            lambda subset, sensitivity_label, x: x["subject"].lower() == subset
            and sensitivity_label
            in x["cultural_sensitivity_label"].replace("-", "UNK"),
            subset,
            sensitivity_label,
        ),
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in MMLU_SUBSETS
    for language in [
        Language.ARABIC,
        Language.GERMAN,
        Language.SPANISH,
        Language.INDONESIAN,
        Language.JAPANESE,
    ]
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
    for sensitivity_label in ["CA", "CS"]
]

TASKS_TABLE: list[LightevalTaskConfig] = [
    GLOBAL_MMLU_LITE,
]
