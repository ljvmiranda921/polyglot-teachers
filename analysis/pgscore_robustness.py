import argparse
import pandas as pd
from pathlib import Path
import logging
import sys

from scipy.stats import spearmanr


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

ALPHA_VALUES = [0, 0.25, 0.50, 0.75, 1.00]


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Compute weight robustness of PG-Score")
    parser.add_argument("-i", "--input_path", type=Path, help="Path to the JSONL file results containing PG-Scores with 'pgr' and 'result' fields.")
    parser.add_argument("--intrinsic_col", type=str, default="z_score", help="Field name of the intrinsic metrics.")
    parser.add_argument("--extrinsic_col", type=str, default="pgr", help="Field name of the extrinsic metrics.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    df = pd.read_json(args.input_path, lines=True)
    spearman_rho, spearman_p = spearmanr(df[args.intrinsic_col], df[args.extrinsic_col])
    logging.info(f"Rank correlation between intr-extr: {spearman_rho} (p={spearman_p})")

    for alpha in ALPHA_VALUES:
        df[f"pg_score_{alpha}"] = df.apply(
            compute_pgscore,
            axis=1,
            **{
                "alpha": alpha,
                "intrinsic": x[args.intrinsic_col],
                "extrinsic": x[args.extrinsic_col],
            },
        )

    breakpoint()


def compute_pgscore(alpha: float, intrinsic: float, extrinsic: float) -> float:
    breakpoint()
    return ((alpha * intrinsic) + (1 - alpha) * extrinsic) / 2


if __name__ == "__main__":
    main()
