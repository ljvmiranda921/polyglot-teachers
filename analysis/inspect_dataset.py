import argparse

import pandas as pd
from datasets import load_dataset


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Inspect and get dataset statistics.")
    parser.add_argument("--input_dataset", help="HuggingFace dataset ID.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df = load_dataset(args.input_dataset, split="train").to_pandas()

    df_report = df.groupby(["model", "strategy"]).size().unstack(fill_value=0)

    breakpoint()


if __name__ == "__main__":
    main()
