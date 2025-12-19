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
    ),
    ModelInfo(
        name="google/gemma-3-12b-it",
        model_family="Gemma",
        parameter_size=12.0,
        context_length=128_000,
        num_languages_supported=140,
    ),
    ModelInfo(
        name="google/gemma-3-4b-it",
        model_family="Gemma",
        parameter_size=4.0,
        context_length=128_000,
        num_languages_supported=140,
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
            "global_mmlu_lite:de": {"acc": 0.4475, "acc_stderr": 0.0248929411943076},
            "global_mmlu_lite:es": {"acc": 0.48, "acc_stderr": 0.02501127565268187},
            "global_mmlu_lite:ja": {"acc": 0.4175, "acc_stderr": 0.024688218756390913},
            "mgsm_custom:de": {
                "extractive_match": 0.096,
                "extractive_match_stderr": 0.018668961419477183,
            },
            "mgsm_custom:es": {
                "extractive_match": 0.12,
                "extractive_match_stderr": 0.020593600596839953,
            },
            "mgsm_custom:ja": {
                "extractive_match": 0.084,
                "extractive_match_stderr": 0.017578738526776334,
            },
            "mrewardbench_mcf:de": {
                "weighted_acc": 0.5646485197511596,
                "weighted_acc_stderr": 0.00035439776714259007,
            },
            "mrewardbench_mcf:es": {
                "weighted_acc": 0.5503686376364357,
                "weighted_acc_stderr": 0.0003460494108649813,
            },
            "mrewardbench_mcf:ja": {
                "weighted_acc": 0.5503426057854023,
                "weighted_acc_stderr": 0.0003531286733025996,
            },
        },
    ),
    ModelInfo(
        name="ibm-granite/granite-4.0-micro",
        model_family="Granite",
        parameter_size=3.0,
        context_length=128_000,
        num_languages_supported=12,
        benchmark_scores={
            "global_mmlu_lite:de": {"acc": 0.54, "acc_stderr": 0.02495107995613509},
            "global_mmlu_lite:es": {"acc": 0.59, "acc_stderr": 0.02462246259333947},
            "global_mmlu_lite:ja": {"acc": 0.4725, "acc_stderr": 0.024993420186752734},
            "mgsm_custom:de": {
                "extractive_match": 0.22,
                "extractive_match_stderr": 0.026251792824605845,
            },
            "mgsm_custom:es": {
                "extractive_match": 0.204,
                "extractive_match_stderr": 0.02553712157454815,
            },
            "mgsm_custom:ja": {
                "extractive_match": 0.152,
                "extractive_match_stderr": 0.022752024491765464,
            },
            "mrewardbench_mcf:de": {
                "weighted_acc": 0.5519676014201315,
                "weighted_acc_stderr": 0.0003516868492426823,
            },
            "mrewardbench_mcf:es": {
                "weighted_acc": 0.562949385111396,
                "weighted_acc_stderr": 0.00034317182166555284,
            },
            "mrewardbench_mcf:ja": {
                "weighted_acc": 0.514154157802626,
                "weighted_acc_stderr": 0.00036734533395975215,
            },
        },
    ),
    ModelInfo(
        name="meta-llama/Llama-3.1-70B-Instruct",
        model_family="Llama",
        parameter_size=70.0,
        context_length=8_000,
        benchmark_scores={
            "mgsm_custom:de": {
                "extractive_match": 0.3,
                "extractive_match_stderr": 0.029040893477575862,
            },
            "mgsm_custom:es": {
                "extractive_match": 0.344,
                "extractive_match_stderr": 0.03010450339231639,
            },
            "mgsm_custom:ja": {
                "extractive_match": 0.292,
                "extractive_match_stderr": 0.028814320402205648,
            },
            "mrewardbench_mcf:de": {
                "weighted_acc": 0.6061121625696093,
                "weighted_acc_stderr": 0.0003261066443640376,
            },
            "mrewardbench_mcf:es": {
                "weighted_acc": 0.5723811219944781,
                "weighted_acc_stderr": 0.0003535743105742438,
            },
            "mrewardbench_mcf:ja": {
                "weighted_acc": 0.5451370923181692,
                "weighted_acc_stderr": 0.00035959642172163846,
            },
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
            "mgsm_custom:de": {
                "extractive_match": 0.16,
                "extractive_match_stderr": 0.023232714782060643,
            },
            "mgsm_custom:es": {
                "extractive_match": 0.136,
                "extractive_match_stderr": 0.021723342617052062,
            },
            "mgsm_custom:ja": {
                "extractive_match": 0.124,
                "extractive_match_stderr": 0.02088638225867326,
            },
            "mrewardbench_mcf:de": {
                "weighted_acc": 0.4879259246080536,
                "weighted_acc_stderr": 0.0003615563111154185,
            },
            "mrewardbench_mcf:es": {
                "weighted_acc": 0.518372969820565,
                "weighted_acc_stderr": 0.0003662965753565583,
            },
            "mrewardbench_mcf:ja": {
                "weighted_acc": 0.5119041310960775,
                "weighted_acc_stderr": 0.00036416225530845267,
            },
        },
    ),
    ModelInfo(
        name="mistralai/Mistral-Small-24B-Instruct-2501",
        model_family="Mistral",
        parameter_size=24.0,
        context_length=128_000,
        benchmark_scores={
            "global_mmlu_lite:de": {"acc": 0.57, "acc_stderr": 0.02478478796128207},
            "global_mmlu_lite:es": {"acc": 0.595, "acc_stderr": 0.02457534065727368},
            "global_mmlu_lite:ja": {"acc": 0.45, "acc_stderr": 0.024905837706844926},
        },
    ),
]
