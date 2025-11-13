from typing import Callable

import pandas as pd
from bespokelabs import curator
from datasets import Dataset
from pydantic import BaseModel, Field

from scripts.utils.prompts import GENERATE_TPL, RESPOND_TPL, TRANSLATE_TPL


class SFTExample(BaseModel):
    prompt: str = Field(description="The input prompt for the LLM.")
    response: str = Field(description="The expected response from the LLM.")


class SFTExampleT(BaseModel):
    prompt: str = Field(description="The input prompt translated into the target language (not English).")  # fmt: skip
    response: str = Field(description="The expected response from the LLM.")


class SFTSynthesizerResponseOnly(curator.LLM):
    def prompt(self, input: dict) -> str:
        return input["synth_prompt"]

    def parse(self, input: dict, response: str) -> dict:
        return {"id": input["id"], "prompt": input["prompt"], "response": response}  # fmt: skip


class SFTSynthesizerPromptResponse(curator.LLM):
    response_format = SFTExample

    def prompt(self, input: dict) -> str:
        return input["synth_prompt"]

    def parse(self, input: dict, response: str) -> dict:
        return {"id": input["id"], "prompt": response.prompt, "response": response.response}  # fmt: skip


class SFTSynthesizerPromptResponseT(curator.LLM):
    response_format = SFTExampleT

    def prompt(self, input: dict) -> str:
        return input["synth_prompt"]

    def parse(self, input: dict, response: str) -> dict:
        return {"id": input["id"], "prompt": response.prompt, "response": response.response}  # fmt: skip


def format_generate(dataset: Dataset, lang_name: str) -> Dataset:
    """Sample prompt, response pairs from a seed dataset and use it as in-context examples to generate new data."""
    assert "prompt" in dataset.column_names, "The column 'prompt' is missing from input dataset!"  # fmt: skip
    assert "response" in dataset.column_names, "The column 'response' is missing from input dataset!"  # fmt: skip

    # Subsample in-context examples
    def sample_in_context_examples(example):
        in_context_examples = dataset.shuffle().select(range(3))
        examples_str = "\n\n".join(
            [
                f"Prompt: {ex['prompt']}\nResponse: {ex['response']}"
                for ex in in_context_examples
            ]
        )
        return examples_str

    dataset = dataset.map(
        lambda x: {
            "synth_prompt": GENERATE_TPL.format(
                lang_name=lang_name,
                examples=sample_in_context_examples(x),
            )
        }
    )
    return dataset


def format_translate(dataset: Dataset, lang_name: str) -> Dataset:
    """Given a prompt in English, translate it to the target language and generate a response in the same language."""
    assert "prompt" in dataset.column_names, "The column 'prompt' is missing from input dataset!"  # fmt: skip

    dataset = dataset.map(
        lambda x: {
            "synth_prompt": TRANSLATE_TPL.format(
                prompt=x["prompt"], lang_name=lang_name
            )
        }
    )
    return dataset


def format_respond(dataset: Dataset, lang_name: str) -> Dataset:
    """Given a prompt in English, generate a response in the target language."""
    assert "prompt" in dataset.column_names, "The column 'prompt' is missing from input dataset!"  # fmt: skip

    dataset = dataset.map(
        lambda x: {
            "synth_prompt": RESPOND_TPL.format(prompt=x["prompt"], lang_name=lang_name)
        }
    )
    return dataset


def get_strategy(name: str) -> tuple[Callable[[Dataset, str], Dataset], curator.LLM]:
    """Returns the formatting function and appropriate distiller class based on the synthesis strategy."""
    formatters = {
        "generate": format_generate,
        "translate": format_translate,
        "respond": format_respond,
    }
    formatter_fn = formatters.get(name)
    distillers = {
        "generate": SFTSynthesizerPromptResponse,
        "respond": SFTSynthesizerResponseOnly,
        "translate": SFTSynthesizerPromptResponseT,
    }
    distiller_fn = distillers.get(name)
    return formatter_fn, distiller_fn
