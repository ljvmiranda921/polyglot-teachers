import pandas as pd
import argparse
from pathlib import Path
import logging
import sys


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    parser = argparse.ArgumentParser(description="Compute base model effect")
    # fmt: off
    parser.add_argument("-b", "--base_model_result", nargs="+", type=str, help="Base model result in format <base_model>::<path/to/results.jsonl>")
    parser.add_argument("-o", "--output_path", type=Path, default="results/base_model_effect.csv", help="Path to save the results in CSV format.")
    parser.add_argument("-l", "--languages", nargs="+", type=str, default=["ar", "id", "de"], help="Language code to include in computation.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    base_model_results: list[tuple[str, Path]] = [parse_base_model_input(str_input) for str_input in args.base_model_result]  # fmt: skip
    breakpoint()
    languages = args.languages
    logging.info(f"Using languages: {languages}")

    results = []
    for base_model, fp in base_model_results:
        df = pd.read_json(fp, lines=True)
        df = df[df["target_lang"].isin(languages)].reset_index(drop=True)
        base_df = df.groupby("teacher_model").agg({"pg_score": "mean", "pgr": "mean"})
        base_df["base_model"] = base_model
        base_df = base_df.sort_values(by="pg_score", ascending=False)
        print(f"========== Results for {base_model} ==========")
        print(base_df.to_markdown())
        results.append(base_df.reset_index(drop=True))

    results_df = pd.concat(results)
    breakpoint()


def parse_base_model_input(s: str) -> tuple[str, Path]:
    """Parse a string input <base_model>::<path/to/results.jsonl>"""
    base_model, path = s.split("::")
    if not Path(path).exists():
        raise ValueError(f"Cannot find file or input: {path}")
    return base_model, Path(path)


if __name__ == "__main__":
    main()
