from pydantic import BaseModel
from typing import Optional, Literal, Any


class ModelInfo(BaseModel):
    name: str
    model_family: str
    url: Optional[str] = None

    # Model characteristics
    parameter_size: float | Literal["Unknown"]
    context_length: Optional[int] = None

    # Multilingual capability
    pct_multi_pretraining: Optional[float] = None
    num_languages_supported: Optional[int] = None

    # Training approach
    instruction_tuned: Optional[bool] = None
    training_approach: Optional[list[str]] = None  # e.g., ["SFT", "RLHF"], ["DPO"]

    # Cost & efficiency
    cost_in_mtok: Optional[float] = None  # Cost per million input tokens (USD)
    cost_out_mtok: Optional[float] = None  # Cost per million output tokens (USD)

    # Performance
    benchmark_scores: Optional[dict[str, Any]] = None

    # Metadata
    release_date: Optional[str] = None  # YYYY-MM format
    license: Optional[str] = None  # e.g., "apache-2.0", "proprietary"


MODEL_INFORMATION: list[ModelInfo] = [
    ModelInfo(
        name="CohereLabs/aya-expanse-32b",
        model_family="Aya",
        parameter_size=32.0,
        context_length=128_000,
        num_languages_supported=23,
        benchmark_scores={
            "global_mmlu_lite:es": {"acc": 0.5625, "acc_stderr": 0.02483498169496957},
            "global_mmlu_lite:ja": {"acc": 0.635, "acc_stderr": 0.0241016539745881},
            "global_mmlu_lite:de": {"acc": 0.6425, "acc_stderr": 0.02399319817984352},
        },
    ),
    ModelInfo(
        name="cohere-command-a",
        model_family="Command",
        parameter_size=111.0,
        context_length=256_000,
    ),
    ModelInfo(
        name="google/gemma-3-27b-it",
        model_family="Gemma",
        parameter_size=27.0,
        context_length=128_000,
        num_languages_supported=140,
        benchmark_scores={
            "global_mmlu_lite:ja": {"acc": 0.5725, "acc_stderr": 0.024766769210836766},
            "global_mmlu_lite:es": {"acc": 0.5825, "acc_stderr": 0.024688218756390913},
            "global_mmlu_lite:de": {"acc": 0.525, "acc_stderr": 0.02499999999999999},
        },
    ),
    ModelInfo(
        name="google/gemma-3-12b-it",
        model_family="Gemma",
        parameter_size=12.0,
        context_length=128_000,
        num_languages_supported=140,
        benchmark_scores={
            "global_mmlu_lite:es": {"acc": 0.5675, "acc_stderr": 0.024802162065186352},
            "global_mmlu_lite:de": {"acc": 0.58, "acc_stderr": 0.02470883072485368},
            "global_mmlu_lite:ja": {"acc": 0.4075, "acc_stderr": 0.024599231297971983},
        },
    ),
    ModelInfo(
        name="google/gemma-3-4b-it",
        model_family="Gemma",
        parameter_size=4.0,
        context_length=128_000,
        num_languages_supported=140,
        benchmark_scores={
            "global_mmlu_lite:de": {"acc": 0.4775, "acc_stderr": 0.025005951672504308},
            "global_mmlu_lite:es": {"acc": 0.3725, "acc_stderr": 0.024203800008203095},
            "global_mmlu_lite:ja": {"acc": 0.3925, "acc_stderr": 0.024445927747963326},
        },
    ),
    ModelInfo(
        name="gpt-4o-mini-2024-07-18",
        model_family="GPT-4o",
        parameter_size="Unknown",
        context_length=128_000,
    ),
    ModelInfo(
        name="ibm-granite/granite-4.0-1b",
        model_family="Granite",
        parameter_size=1.0,
        context_length=128_000,
        benchmark_scores={
            "global_mmlu_lite:es": {"acc": 0.48, "acc_stderr": 0.02501127565268187},
            "global_mmlu_lite:de": {"acc": 0.4475, "acc_stderr": 0.0248929411943076},
            "global_mmlu_lite:ja": {"acc": 0.4175, "acc_stderr": 0.024688218756390913},
        },
    ),
    ModelInfo(
        name="ibm-granite/granite-4.0-micro",
        model_family="Granite",
        parameter_size=3.0,
        context_length=128_000,
        num_languages_supported=12,
        benchmark_scores={
            "global_mmlu_lite:es": {"acc": 0.59, "acc_stderr": 0.02462246259333947},
            "global_mmlu_lite:de": {"acc": 0.54, "acc_stderr": 0.02495107995613509},
            "global_mmlu_lite:ja": {"acc": 0.4725, "acc_stderr": 0.024993420186752734},
        },
    ),
    ModelInfo(
        name="meta-llama/Llama-3.1-70B-Instruct",
        model_family="Llama",
        parameter_size=70.0,
        context_length=8_000,
        benchmark_scores={
            "mgsm_custom:de": {"extractive_match": 0.3, "extractive_match_stderr": 0.014491376746189437},
        },
    ),
    ModelInfo(
        name="meta-llama/Llama-3.1-8B-Instruct",
        model_family="Llama",
        parameter_size=8.0,
        context_length=8_000,
        benchmark_scores={
            "global_mmlu_lite:de": {"acc": 0.42, "acc_stderr": 0.02470883072485368},
            "global_mmlu_lite:es": {"acc": 0.505, "acc_stderr": 0.02503005711936146},
            "global_mmlu_lite:ja": {"acc": 0.5, "acc_stderr": 0.02503130871608794},
        },
    ),
    ModelInfo(
        name="mistralai/Mistral-Small-24B-Instruct-2501",
        model_family="Mistral",
        parameter_size=24.0,
        context_length=128_000,
        benchmark_scores={
            "global_mmlu_lite:de": {"acc": 0.57, "acc_stderr": 0.024769873817356935},
        },
    ),
]
