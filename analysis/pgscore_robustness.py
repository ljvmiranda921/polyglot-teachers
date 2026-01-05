import argparse
import pandas as pd
from pathlib import Path
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import itertools

from scipy.stats import spearmanr
from analysis.utils.plot_theme import PLOT_PARAMS, COLORS, OUTPUT_DIR, FONT_SIZES

plt.rcParams.update(PLOT_PARAMS)

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
        col_name = alpha
        df[col_name] = df.apply(
            lambda row: compute_pgscore(
                alpha=alpha,
                intrinsic=row[args.intrinsic_col],
                extrinsic=row[args.extrinsic_col],
            ),
            axis=1,
        )

    pairs = list(itertools.product(ALPHA_VALUES, ALPHA_VALUES))
    pairs_spearman_rho = []
    for set1, set2 in pairs:
        rho, p = spearmanr(df[set1], df[set2])
        print(set1, set2, f"{rho} (p={p})")
        pairs_spearman_rho.append((set1, set2, rho, p))

    fig, ax = plt.subplots(figsize=(5, 6))
    cmap = LinearSegmentedColormap.from_list(
        "cambridge_diverging",
        [COLORS["cherry"], COLORS["white"], COLORS["green"]],
    )
    rankings_df = pd.DataFrame(0, index=ALPHA_VALUES, columns=ALPHA_VALUES)
    for set1, set2, rho, p in pairs_spearman_rho:
        rankings_df.loc[set1, set2] = rho
    mask = np.triu(np.ones_like(rankings_df, dtype=bool), k=1)
    heatmap = sns.heatmap(
        rankings_df,
        annot=True,
        fmt=".2f",
        mask=mask,
        cmap=cmap,
        center=0,
        cbar_kws={
            "label": "Spearman rank $\\rho$",
            "orientation": "horizontal",
            "shrink": 0.8,
            "pad": 0.1,
        },
        ax=ax,
    )

    plt.savefig(OUTPUT_DIR / "pgscore_robustness_heatmap.pdf", bbox_inches="tight")


def compute_pgscore(alpha: float, intrinsic: float, extrinsic: float) -> float:
    return (alpha * intrinsic) + (1 - alpha) * extrinsic


if __name__ == "__main__":
    main()
