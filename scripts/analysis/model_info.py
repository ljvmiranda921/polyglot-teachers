from pydantic import BaseModel
from typing import Optional


class ModelInfo(BaseModel):
    name: str
    model_family: str
    url: Optional[str] = None

    # Model characteristics
    parameter_size: float
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
        parameter_size=32.0,  # TODO: Fill in
        context_length=128_000,
        num_languages_supported=23,
    ),
    ModelInfo(
        name="cohere-command-a",
        model_family="Command",
        parameter_size=111.0,  # TODO: Fill in
    ),
    ModelInfo(
        name="google/gemma-3-12b-it",
        model_family="Gemma",
        parameter_size=0.0,  # TODO: Fill in
    ),
    ModelInfo(
        name="google/gemma-3-27b-it",
        model_family="Gemma",
        parameter_size=0.0,  # TODO: Fill in
    ),
    ModelInfo(
        name="google/gemma-3-4b-it",
        model_family="Gemma",
        parameter_size=0.0,  # TODO: Fill in
    ),
    ModelInfo(
        name="gpt-4o-mini-2024-07-18",
        model_family="GPT-4o",
        parameter_size=0.0,  # TODO: Fill in
    ),
    ModelInfo(
        name="ibm-granite/granite-4.0-1b",
        model_family="Granite",
        parameter_size=0.0,  # TODO: Fill in
    ),
    ModelInfo(
        name="ibm-granite/granite-4.0-micro",
        model_family="Granite",
        parameter_size=0.0,  # TODO: Fill in
    ),
    ModelInfo(
        name="meta-llama/Llama-3.1-70B-Instruct",
        model_family="Llama",
        parameter_size=0.0,  # TODO: Fill in
    ),
    ModelInfo(
        name="meta-llama/Llama-3.1-8B-Instruct",
        model_family="Llama",
        parameter_size=0.0,  # TODO: Fill in
    ),
]
