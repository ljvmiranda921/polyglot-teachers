from pydantic import BaseModel
from typing import Optional, Literal, Any


class ModelInfo(BaseModel):
    name: str
    model_family: str
    beautiful_name: Optional[str] = None  # Short display name for plots
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


class LanguageInfo(BaseModel):
    name: str
    iso_639_1: str

    # Resourcesness
    joshi_etal_resource_level: Optional[int] = None  # 1-5 scale
    ethnologue_num_speakers: Optional[int] = None
    pct_commoncrawl: Optional[float] = None  # Percentage of CommonCrawl data
    native_speakers_in_m: Optional[float] = None


LANGUAGE_INFORMATION: list[LanguageInfo] = [
    LanguageInfo(
        name="Arabic",
        iso_639_1="ar",
        joshi_etal_resource_level=5,
        pct_commoncrawl=0.65,
        native_speakers_in_m=380,
    ),
    LanguageInfo(
        name="Czech",
        iso_639_1="cs",
        joshi_etal_resource_level=4,
        pct_commoncrawl=0.99,
        native_speakers_in_m=10.7,
    ),
    LanguageInfo(
        name="German",
        iso_639_1="de",
        joshi_etal_resource_level=5,
        pct_commoncrawl=6.01,
        native_speakers_in_m=95,
    ),
    LanguageInfo(
        name="Spanish",
        iso_639_1="es",
        joshi_etal_resource_level=5,
        pct_commoncrawl=4.37,
        native_speakers_in_m=500,
    ),
    LanguageInfo(
        name="Indonesian",
        iso_639_1="id",
        joshi_etal_resource_level=3,
        pct_commoncrawl=0.95,
        native_speakers_in_m=43,
    ),
    LanguageInfo(
        name="Japanese",
        iso_639_1="ja",
        joshi_etal_resource_level=5,
        pct_commoncrawl=5.20,
        native_speakers_in_m=120,
    ),
]


MODEL_INFORMATION: list[ModelInfo] = [
    ModelInfo(
        name="CohereLabs/aya-expanse-32b",
        model_family="Aya",
        beautiful_name="Aya Expanse 32B",
        parameter_size=32.0,
        context_length=128_000,
        num_languages_supported=23,
    ),
    ModelInfo(
        name="cohere-command-a",
        model_family="Command",
        beautiful_name="Command A",
        parameter_size=111.0,
        context_length=256_000,
    ),
    ModelInfo(
        name="google/gemma-3-27b-it",
        model_family="Gemma",
        beautiful_name="Gemma 3 27B",
        parameter_size=27.0,
        context_length=128_000,
        num_languages_supported=140,
    ),
    ModelInfo(
        name="google/gemma-3-12b-it",
        model_family="Gemma",
        beautiful_name="Gemma 3 12B",
        parameter_size=12.0,
        context_length=128_000,
        num_languages_supported=140,
    ),
    ModelInfo(
        name="google/gemma-3-4b-it",
        model_family="Gemma",
        beautiful_name="Gemma 3 4B",
        parameter_size=4.0,
        context_length=128_000,
        num_languages_supported=140,
    ),
    ModelInfo(
        name="gpt-4o-mini-2024-07-18",
        model_family="GPT-4o",
        beautiful_name="GPT-4o mini",
        parameter_size="Unknown",
        context_length=128_000,
    ),
    ModelInfo(
        name="ibm-granite/granite-4.0-1b",
        model_family="Granite",
        beautiful_name="Granite 4.0 1B",
        parameter_size=1.0,
        context_length=128_000,
    ),
    ModelInfo(
        name="ibm-granite/granite-4.0-micro",
        model_family="Granite",
        beautiful_name="Granite 4.0 Micro",
        parameter_size=3.0,
        context_length=128_000,
        num_languages_supported=12,
    ),
    ModelInfo(
        name="meta-llama/Llama-3.1-70B-Instruct",
        model_family="Llama",
        beautiful_name="Llama 3.1 70B",
        parameter_size=70.0,
        context_length=8_000,
    ),
    ModelInfo(
        name="meta-llama/Llama-3.1-8B-Instruct",
        model_family="Llama",
        beautiful_name="Llama 3.1 8B",
        parameter_size=8.0,
        context_length=8_000,
    ),
    ModelInfo(
        name="mistralai/Mistral-Small-24B-Instruct-2501",
        model_family="Mistral",
        beautiful_name="Mistral Small 24B",
        parameter_size=24.0,
        context_length=128_000,
    ),
]
