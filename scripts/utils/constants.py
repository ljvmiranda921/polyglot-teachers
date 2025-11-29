# Model information

from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    model_family: str
    parameter_size: float


MODEL_INFORMATION: list[ModelInfo] = [
    # fmt: off
    # Cohere models
    ModelInfo(name="CohereLabs/aya-expanse-32b", model_family="cohere", parameter_size=32.0),
    ModelInfo(name="cohere-command-a", model_family="cohere", parameter_size=104.0),
    # Google Gemma models
    ModelInfo(name="google/gemma-3-4b-it", model_family="gemma", parameter_size=4.0),
    ModelInfo(name="google/gemma-3-12b-it", model_family="gemma", parameter_size=12.0),
    ModelInfo(name="google/gemma-3-27b-it", model_family="gemma", parameter_size=27.0),
    # OpenAI models
    ModelInfo(name="gpt-4o-mini-2024-07-18", model_family="gpt", parameter_size=8.0),
    # IBM Granite models
    ModelInfo(name="ibm-granite/granite-4.0-micro", model_family="granite", parameter_size=0.5),
    ModelInfo(name="ibm-granite/granite-4.0-1b", model_family="granite", parameter_size=1.0),
    # Meta Llama models
    ModelInfo(name="meta-llama/Llama-3.1-8B-Instruct", model_family="llama", parameter_size=8.0),
    ModelInfo(name="meta-llama/Llama-3.1-70B-Instruct", model_family="llama", parameter_size=70.0),
    # fmt: on
]
