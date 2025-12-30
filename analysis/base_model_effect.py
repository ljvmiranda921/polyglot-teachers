import pandas as pd
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(help="Compute base model effect")
    # fmt: off
    parser.add_argument("-b", "--base_model_result", nargs="+", type=str, help="Base model result in format <base_model>::<path/to/results.jsonl>")
    parser.add_argument("-o", "--output_path", type=Path, default="results/base_model_effect.csv", help="Path to save the results in CSV format.")
    parser.add_argument("-l", "--languages", nargs="+", type=str, help="Language code to include in computation.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    breakpoint()


def parse_base_model_input(s: str) -> tuple[str, Path]:
    """Parse a string input <base_model>::<path/to/results.jsonl>"""
    pass
