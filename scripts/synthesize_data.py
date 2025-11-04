import argparse
import logging
import os
import sys
import json
from pathlib import Path

import pandas as pd
import tiktoken
from bespokelabs.curator.types.curator_response import CuratorResponse
from datasets import Dataset, load_dataset
from langcodes import Language

from scripts.utils.llm_inference import get_strategy
from scripts.utils.prompts import SYSTEM_PROMPT

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    description = "Generate synthetic data given a dataset, strategy (generate, translate, refine), and target language."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_dataset", type=str, required=True, help="Seed HuggingFace dataset for data synthesis.")
    parser.add_argument("--output_dataset", type=str, required=True, help="Name of the HuggingFace dataset to store the outputs.")
    parser.add_argument("--target_lang", type=str, required=True, help="The two-letter code (ISO 639-2) of the target language.")
    parser.add_argument("--strategy", choices=["generate", "translate", "respond"], required=True, help="The synthesis strategy to use.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18", help="The model to use for model generation. If it's a GPT-4 API-based model, will use batch inference. Be sure to check the values in the --cache_dir option.")
    parser.add_argument("--has_prefilter", action="store_true", help="If set, assumes that the input dataset has a 'strategy' and 'language' fields to pre-filter instances based on the chosen strategy and language.")
    parser.add_argument("--limit", default=None, help="If set, then will only run the synthesis strategy on the first N instances.")
    parser.add_argument("--shuffle", default=None, help="If set, will shuffle the dataset using the seed provided before synthesizing. If --limit is set, then THIS command will be run first before shuffling.")
    parser.add_argument("--batch_mode", action="store_true", help="If set, will use batch inference for LLM calls.")
    parser.add_argument("--backend", default=None, help="The backend to use for LLM inference. See: https://docs.bespokelabs.ai/bespoke-curator/how-to-guides")
    parser.add_argument("--no_cache", action="store_true", help="If set, will not use any caching for LLM calls.")
    parser.add_argument("--append", action="store_true", help="If set, will append to existing output dataset instead of overwriting.")
    parser.add_argument("--backend_params", type=str, default=None, help="If set, will pass these additional parameters (in JSON format) to the backend LLM inference calls.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    # Curator setup
    if args.no_cache:
        logging.info("Disabling Curator caching as per --no_cache flag.")
        os.environ["CURATOR_DISABLE_CACHE"] = "1"

    backend_params = json.loads(args.backend_params) if args.backend_params else None

    # Prepare dataset for synthesis
    dataset = load_dataset(args.input_dataset, split="train")
    if args.has_prefilter:
        dataset = dataset.filter(lambda ex: args.strategy in (ex.get("strategy")))
        if args.strategy != "translate":
            dataset = dataset.filter(lambda ex: args.target_lang == ex.get("language"))
    if args.shuffle:
        logging.info(f"Shuffling the dataset using seed {args.shuffle}")
        dataset = dataset.shuffle(seed=args.shuffle)
    if args.limit:
        logging.info(f"Getting the first {args.limit} instances")
        dataset = dataset.select(range(min(int(args.limit), len(dataset))))

    # Prepare data synthesis prompts
    logging.info(f"Using '{args.strategy}' synthesis strategy")
    logging.info(f"No. of instances: {len(dataset)}")
    format_fn, distiller_fn = get_strategy(name=args.strategy)

    lang_name = Language.make(args.target_lang).display_name()
    if "Unknown language" in lang_name:
        raise ValueError(f"Unknown language: {args.target_lang}. Please input a two-letter ISO 693-2 code.")  # fmt: skip

    input_dataset: Dataset = format_fn(dataset, lang_name=lang_name)
    system_prompt = SYSTEM_PROMPT.format(lang_name=lang_name)
    if backend_params and "max_model_length" in backend_params:
        max_model_len = int(backend_params.get("max_model_length"))
        input_dataset = filter_by_token_length(
            input_dataset,
            max_model_len,
            system_prompt=system_prompt,
            prompt_key="synth_prompt",
        )

    # Perform data synthesis
    distiller = distiller_fn(
        model_name=args.model,
        batch=args.batch_mode,
        system_prompt=system_prompt,
        backend=args.backend,
        backend_params=backend_params,
    )
    curator_response: CuratorResponse = distiller(input_dataset)
    logging.info(f"Data synthesis cost: {curator_response.cost_info.total_cost} USD")

    # Merge input dataset and the synthesized outputs, and format outputs for post-training
    output_dataset = prepare_output_dataset(
        curator_response.dataset,
        input_dataset=input_dataset,
        strategy=args.strategy,
        model=args.model,
    )
    output_dataset = output_dataset.filter(lambda ex: ex["response"] is not None)

    # Upload output to HuggingFace
    logging.info(f"Uploading output dataset to HuggingFace: {args.output_dataset}")
    upload_to_huggingface(
        dataset=output_dataset,
        dataset_name=args.output_dataset,
        append=args.append,
    )


def filter_by_token_length(
    dataset: Dataset,
    max_model_length: int,
    *,
    system_prompt: str,
    prompt_key: str = "synth_prompt",
    buffer: int = 500,
) -> Dataset:
    encoding = tiktoken.get_encoding("cl100k_base")
    system_tokens = len(encoding.encode(system_prompt))

    def is_within_length(example):
        prompt_tokens = len(encoding.encode(example[prompt_key]))
        total_tokens = prompt_tokens + system_tokens + buffer
        return total_tokens <= max_model_length

    original_len = len(dataset)
    filtered_dataset = dataset.filter(is_within_length)
    filtered_len = len(filtered_dataset)
    logging.info(
        f"Filtered {original_len - filtered_len} prompts exceeding max_model_length. "
        f"Remaining: {filtered_len}"
    )

    return filtered_dataset


def prepare_output_dataset(
    synth_dataset: Dataset,
    *,
    input_dataset: Dataset,
    strategy: str,
    model: str,
) -> Dataset:

    # Merge input dataset and synthesized dataset to keep some metadata
    input_df = input_dataset.to_pandas().drop(columns=["prompt", "response"])
    input_df["strategy"] = strategy  # Keep track of the synthesis strategy used
    input_df["model"] = model  # Keep track of the model used for synthesis
    synth_df = synth_dataset.to_pandas()
    output_df = pd.merge(input_df, synth_df, on="id", how="left")

    ds = Dataset.from_pandas(output_df)
    final_ds = ds.map(to_conversation_format)
    return final_ds


def upload_to_huggingface(dataset: Dataset, dataset_name: str, append: bool = False):
    """Upload the dataset to HuggingFace hub. If append is True, will append to existing dataset."""
    try:
        if append:
            logging.info(
                f"Appending to existing dataset on HuggingFace: {dataset_name}"
            )
            existing_dataset = load_dataset(dataset_name, split="train")
            combined_dataset = Dataset.from_pandas(
                pd.concat(
                    [existing_dataset.to_pandas(), dataset.to_pandas()],
                    ignore_index=True,
                )
            )
            combined_dataset.push_to_hub(dataset_name, private=False)
        else:
            logging.info(f"Pushing new dataset to HuggingFace: {dataset_name}")
            dataset.push_to_hub(dataset_name, private=False)
    except Exception:
        logging.exception(
            "Failed to push dataset to HuggingFace hub, saving locally as parquet."
        )
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        safe_name = dataset_name.replace("/", "___")
        output_path = data_dir / f"{safe_name}.parquet"
        df = dataset.to_pandas()
        df.to_parquet(output_path, index=False)
        logging.info(f"Saved output dataset to {output_path}")


def to_conversation_format(example):
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }


if __name__ == "__main__":
    main()
