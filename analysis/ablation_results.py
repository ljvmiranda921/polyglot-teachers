import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils.plot_theme import COLORS, OUTPUT_DIR, PLOT_PARAMS, FONT_SIZES

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


def add_arc_annotation(
    ax, df, start_idx, end_idx, annotation_text, y_offset=10, text_y_offset=20, rad=-0.5
):
    from matplotlib.patches import FancyArrowPatch

    arc_arrow = FancyArrowPatch(
        (start_idx, df["filbench_score"].iloc[start_idx] + y_offset),
        (end_idx, df["filbench_score"].iloc[end_idx] + y_offset),
        arrowstyle="->",
        connectionstyle=f"arc3,rad={rad}",
        color="black",
        linewidth=1.5,
        mutation_scale=20,
    )
    ax.add_patch(arc_arrow)

    arc_mid_x = (start_idx + end_idx) / 2.0
    arc_mid_y = df["filbench_score"].iloc[start_idx : end_idx + 1].max() + text_y_offset
    ax.text(
        arc_mid_x,
        arc_mid_y,
        annotation_text,
        ha="center",
        va="bottom",
        fontdict={"size": FONT_SIZES.get("small")},
        multialignment="left",
    )


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
    colors = [
        COLORS["slate_2"] if i <= 1 else COLORS["cambridge_blue"]
        for i in range(len(df))
    ]
    hatches = ["///" if i <= 1 else "" for i in range(len(df))]

    bars = ax.bar(
        x_positions,
        df["filbench_score"],
        color=colors,
        width=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    for bar, score in zip(bars, df["filbench_score"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{score:.1f}",
            ha="center",
            va="bottom",
            fontdict={"size": FONT_SIZES.get("medium")},
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)

    ax.set_ylim([0, 90])

    ax.text(
        0,
        -0.02,
        "None",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontdict={"size": FONT_SIZES.get("medium")},
    )

    ax.text(
        1,
        -0.02,
        "GPT-4o",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontdict={"size": FONT_SIZES.get("medium")},
    )

    ax.text(
        2,
        -0.02,
        "Aya Exp",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontdict={"size": FONT_SIZES.get("medium")},
    )

    gemma_27b_start = 3
    gemma_27b_end = 6
    gemma_27b_mid = (gemma_27b_start + gemma_27b_end) / 2.0
    num_dashes = 8

    ax.text(
        gemma_27b_mid,
        -0.02,
        r"$\vert$" + "-" * num_dashes + " Gemma 3 27B " + "-" * num_dashes + r"$\vert$",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontdict={"size": FONT_SIZES.get("medium")},
    )

    # fmt: off
    add_arc_annotation(ax, df, 0, 1, "Use synthetic\npipeline", y_offset=10, text_y_offset=20)
    add_arc_annotation(ax, df, 1, 2, "Better\nteacher", y_offset=10, text_y_offset=20)
    add_arc_annotation(ax, df, 2, 3, "Match\nfamily", y_offset=10, text_y_offset=20)
    # fmt: on

    ax.set_xlabel("Teacher Model", fontsize=FONT_SIZES.get("large"), labelpad=25)
    ax.set_ylabel(r"\textsc{FilBench Score}", fontsize=FONT_SIZES.get("large"))

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "tgl_ablation_filbench_scores.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
