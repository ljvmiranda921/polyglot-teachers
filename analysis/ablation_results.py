import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils.plot_theme import COLORS, OUTPUT_DIR, PLOT_PARAMS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

plt.rcParams.update(PLOT_PARAMS)


def get_args():
    parser = argparse.ArgumentParser(description="Draw ablation results for Tagalog.")
    # fmt: off
    parser.add_argument("--input_dir", type=Path, help="Directory containing ablation result JSON files.", required=True)
    parser.add_argument("--figsize", type=lambda s: tuple(map(int, s.split(","))), default=(12, 7), help="Figure size as WIDTH,HEIGHT in inches. Default: 6,8")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    logging.info(f"Loading ablation results from {args.input_dir}")

    data = []
    for json_file in sorted(args.input_dir.glob("*.json")):
        import json

        with open(json_file, "r") as f:
            result = json.load(f)
            exp_name = json_file.stem
            filbench_score = result.get("filbench_score", 0)
            data.append({"experiment": exp_name, "filbench_score": filbench_score})

    df = pd.DataFrame(data)
    logging.info(f"Loaded {len(df)} experiments")

    fig, ax = plt.subplots(figsize=args.figsize)

    x_positions = np.arange(len(df))
    bars = ax.bar(
        x_positions, df["filbench_score"], color=COLORS["cambridge_blue"], width=0.7
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(df["experiment"], rotation=45, ha="right")

    ax.set_ylabel("FILBench Score")
    ax.set_xlabel("Experiment")

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "tgl_ablation_filbench_scores.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
