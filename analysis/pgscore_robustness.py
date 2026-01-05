import argparse
import pandas as pd
from pathlib import Path
import logging
import sys
import matplotlib.pyplot as plt
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

    col_name_tracker = []
    for alpha in ALPHA_VALUES:
        col_name = f"pg_score_{alpha}"
        col_name_tracker.append(col_name)
        df[col_name] = df.apply(
            lambda row: compute_pgscore(
                alpha=alpha,
                intrinsic=row[args.intrinsic_col],
                extrinsic=row[args.extrinsic_col],
            ),
            axis=1,
        )

    pairs = list(itertools.product(col_name_tracker, col_name_tracker))
    pairs_spearman_rho = []
    for set1, set2 in pairs:
        rho, p = spearmanr(df[set1], df[set2])
        print(set1, set2, f"{rho} (p={p})")
        pairs_spearman_rho.append((set1, set2, rho, p))

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.matshow(
        [
            [x[2] for x in pairs_spearman_rho[i : i + len(col_name_tracker)]]
            for i in range(0, len(pairs_spearman_rho), len(col_name_tracker))
        ],
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(col_name_tracker)))
    ax.set_yticks(range(len(col_name_tracker)))
    ax.set_xticklabels([ALPHA_VALUES])
    ax.set_yticklabels([ALPHA_VALUES])
    plt.savefig(OUTPUT_DIR / "pgscore_robustness_heatmap.pdf", bbox_inches="tight")


def compute_pgscore(alpha: float, intrinsic: float, extrinsic: float) -> float:
    return (alpha * intrinsic) + (1 - alpha) * extrinsic


if __name__ == "__main__":
    main()
