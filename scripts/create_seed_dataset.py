import uuid
import logging
import sys
import argparse
import hashlib

import pandas as pd
from datasets import Dataset, load_dataset


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

LANG_MAPPING = {
    "Spanish": "es",
    "German": "de",
    "Indonesian": "id",
    "Czech": "cs",
    "Japanese": "ja",
}


def get_data_processors():
    """Registry of dataset processors. See implementation of each processor below."""
    return {
        "allenai/WildChat-4.8M": _process_wildchat,
        "openai/gsm8k": _process_gsm8k,
        "Magpie-Align/Magpie-Pro-300K-Filtered": _process_magpie_pro_300k,
    }


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Create a seed dataset from a series of datasets.")
    parser.add_argument("--output_dataset", type=str, required=True, help="HuggingFace dataset path to save the seed dataset to.")
    parser.add_argument("--exclude", nargs="+", type=str, default=[], help="List of dataset names to exclude from the seed dataset.")
    parser.add_argument("--include", nargs="+", type=str, default=[], help="List of dataset names to exclusively include in the seed dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    all_dfs = []
    for dataset_name, processor in get_data_processors().items():
        if args.include:
            if dataset_name not in args.include:
                continue
        elif dataset_name in args.exclude:
            logging.info(f"Skipping excluded dataset: {dataset_name}")
            continue

        logging.info(f"Processing dataset: {dataset_name}")
        df = processor()
        all_dfs.append(df)
        breakpoint()


def _process_wildchat() -> pd.DataFrame:
    """Process the allenai/WildChat-4.8M dataset that contains multilingual prompt-response pairs."""
    num_instances = 200_000
    wildchat_4_8m = load_dataset("allenai/WildChat-4.8M", split="train", streaming=True)
    sampled = wildchat_4_8m.shuffle(seed=42).take(num_instances)

    sampled_df = pd.DataFrame(list(sampled))
    filtered_df = sampled_df[(sampled_df["language"].isin(LANG_MAPPING.keys()))]

    # Transform to desired format
    wildchat_df = pd.DataFrame(
        {
            "id": [uuid.uuid4().hex for _ in range(len(filtered_df))],
            "source": "allenai/WildChat-4.8M",
            "conversation": filtered_df["conversation"].values,
            "language": filtered_df["language"].map(LANG_MAPPING).values,
            "strategy": [["generate", "respond"] for _ in range(len(filtered_df))],
            "source_id": filtered_df["conversation_hash"].values,
        }
    )

    wildchat_df["prompt"] = wildchat_df.conversation.apply(lambda x: x[0]["content"])
    wildchat_df["response"] = wildchat_df.conversation.apply(lambda x: x[1]["content"])
    wildchat_df = wildchat_df.drop(columns=["conversation"])  # No longer needed
    return wildchat_df


def _process_gsm8k() -> pd.DataFrame:
    """Process the openai/gsm8k dataset for math word problems."""
    gsm8k_df = load_dataset("openai/gsm8k", "main", split="train").to_pandas()
    gsm8k_df["source_id"] = gsm8k_df["question"].apply(
        lambda x: hashlib.md5(x.encode()).hexdigest()
    )
    gsm8k_df["id"] = [uuid.uuid4().hex for _ in range(len(gsm8k_df))]
    gsm8k_df = gsm8k_df.rename(columns={"question": "prompt", "answer": "response"})
    gsm8k_df["source"] = "openai/gsm8k"
    gsm8k_df["language"] = "en"
    gsm8k_df["strategy"] = [["translate"] for _ in range(len(gsm8k_df))]
    return gsm8k_df


def _process_magpie_pro_300k() -> pd.DataFrame:
    """Process the Magpie-Align/Magpie-Pro-300K-Filtered dataset for general chat text."""
    pass


if __name__ == "__main__":
    main()
