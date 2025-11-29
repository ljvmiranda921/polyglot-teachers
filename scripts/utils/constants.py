from pydantic import BaseModel
from typing import Optional


class ModelInfo(BaseModel):
    name: str
    model_family: str

    # Model characteristics
    parameter_size: float
    context_length: Optional[int] = None

    # Multilingual capability
    pct_multi_pretraining: Optional[float] = (
        None  # % of multilingual data in pretraining
    )
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


MODEL_INFORMATION: list[ModelInfo] = []
