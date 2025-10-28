import uuid
import logging
import sys
import argparse

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


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Create a seed dataset from a series of datasets.")
    parser.add_argument("--output_dataset", type=str, required=True, help="HuggingFace dataset path to save the seed dataset to.")
    parser.add_argument("--exclude", nargs="+", type=list, default=[], help="List of dataset names to exclude from the seed dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    all_dfs = []
    for dataset_name, processor in DATA_PROCESSORS.items():
        if dataset_name in args.exclude:
            logging.info(f"Skipping excluded dataset: {dataset_name}")
            continue
        logging.info(f"Processing dataset: {dataset_name}")
        df = processor()
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
    return wildchat_df


DATA_PROCESSORS = {"allenai/WildChat-4.8M": _process_wildchat}

if __name__ == "__main__":
    main()
