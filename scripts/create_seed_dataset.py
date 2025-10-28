import uuid
import argparse

import pandas as pd
from datasets import Dataset, load_dataset

LANGUAGES = {
    "es": "Spanish",
    "de": "German",
    "id": "Indonesian",
    "cs": "Czech",
    "ja": "Japanese",
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


if __name__ == "__main__":
    main()
