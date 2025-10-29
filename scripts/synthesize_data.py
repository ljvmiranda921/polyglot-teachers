import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
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
    parser.add_argument("--input_dataset", type=str, help="Seed HuggingFace dataset for data synthesis.")
    parser.add_argument("--output_dataset", type=str, help="Name of the HuggingFace dataset to store the outputs.")
    parser.add_argument("--target_lang", type=str, help="The two-letter code (ISO 639-2) of the target language.")
    parser.add_argument("--strategy", choices=["generate", "translate", "refine"], help="The synthesis strategy to use.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18", help="The model to use for model generation. If it's a GPT-4 API-based model, will use batch inference. Be sure to check the values in the --cache_dir option.")
    parser.add_argument("--limit", default=None, help="If set, then will only run the synthesis strategy on the first N instances.")
    parser.add_argument("--shuffle", default=None, help="If set, will shuffle the dataset using the seed provided before synthesizing. If --limit is set, then that command will be run first before shuffling.")
    parser.add_argument("--dry_run", action="store_true", help="If set, will only prepare the dataset and call a single instance to show what a response will look like.")
    parser.add_argument("--batch_mode", action="store_true", help="If set, will use batch inference for LLM calls.")
    parser.add_argument("--backend", type=str, default="openai", help="The backend to use for LLM inference.")
    parser.add_argument("--has_prefilter", action="store_true", help="If set, assumes that the input dataset has a 'strategy' field to pre-filter instances based on the chosen strategy.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    # Prepare dataset for synthesis
    dataset = load_dataset(args.input_dataset)
    if args.limit:
        logging.info(f"Getting the first {args.limit} instances")
        dataset = dataset.select(range(int(args.limit)))
    if args.shuffle:
        logging.info(f"Shuffling the dataset using seed {args.shuffle}")
        dataset = dataset.shuffle(seed=args.shuffle)

    # Prepare data synthesis prompts
    logging.info(f"Using '{args.strategy}' synthesis strategy")
    if args.has_prefilter:
        dataset = dataset.filter(lambda ex: args.strategy in (ex.get("strategy")))
    breakpoint()
    format_fn, distiller_fn = get_strategy(name=args.strategy)

    lang_name = Language.make(args.target_lang).display_name()
    if "Unknown language" in lang_name:
        raise ValueError(f"Unknown language: {args.target_lang}. Please input a two-letter ISO 693-2 code.")  # fmt: skip

    input_dataset = format_fn(dataset, lang_name=lang_name)
    system_prompt = SYSTEM_PROMPT.format(lang_name=lang_name)
    breakpoint()

    # Perform data synthesis
    distiller = distiller_fn(
        model_name=args.model,
        batch=args.batch_mode,
        system_prompt=system_prompt,
        backend=args.backend,
    )
    output_dataset = distiller(input_dataset)

    # Format dataset for post-training

    # Upload output to HuggingFace


if __name__ == "__main__":
    main()
