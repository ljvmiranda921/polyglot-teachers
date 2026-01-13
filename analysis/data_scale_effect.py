import argparse
from pathlib import Path
import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd

from analysis.utils.metadata import LANGUAGE_INFORMATION
from analysis.utils.plot_theme import COLORS, OUTPUT_DIR, PLOT_PARAMS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

plt.rcParams.update(PLOT_PARAMS)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot data scale effect analysis.")
    parser.add_argument("-i", "--input_path", type=Path, default="results/pg_scores_data_scaling.jsonl", help="Path to input JSONL file.")
    parser.add_argument("-o", "--output_path", type=Path, default=OUTPUT_DIR / "data_scale_effect.pdf", help="Path to output plot file.")
    parser.add_argument("--figsize", type=lambda s: tuple(map(int, s.split(","))), default=(14, 6), help="Figure size as WIDTH,HEIGHT in inches. Default: 14,6")
    parser.add_argument("--benchmark_only", action="store_true", help="Plot only benchmark performance (not PG-Score).")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    # Load data
    df = pd.read_json(args.input_path, lines=True)

    # Merge with language metadata to get full language names
    df_langs = pd.DataFrame([lang.model_dump() for lang in LANGUAGE_INFORMATION])
    df_plot = df.merge(
        df_langs[["iso_639_1", "name"]],
        left_on="target_lang",
        right_on="iso_639_1",
        how="left",
    )

    # Sort by scale for proper line plotting
    df_plot = df_plot.sort_values("scale")

    # Define colors, markers, and line styles for each language
    language_styles = {
        "Arabic": {
            "color": COLORS.get("warm_crest"),
            "marker": "o",
            "linestyle": "-",
        },
        "German": {
            "color": COLORS.get("warm_blue"),
            "marker": "s",
            "linestyle": "--",
        },
        "Indonesian": {
            "color": COLORS.get("warm_purple"),
            "marker": "^",
            "linestyle": "-.",
        },
    }

    if args.benchmark_only:
        # Single plot: Benchmark performance only
        fig, ax = plt.subplots(1, 1, figsize=args.figsize)

        for lang_name, group in df_plot.groupby("name"):
            style = language_styles.get(lang_name, {})
            ax.plot(
                group["scale"],
                group["result"],
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                label=lang_name,
                color=style.get("color", COLORS.get("slate_3")),
                linewidth=2,
                markersize=8,
                alpha=0.8,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Num. samples, log")
        ax.set_ylabel("Avg. Multilingual Performance")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        # Add legend below the plot
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
            frameon=False,
        )
    else:
        # Two plots: PG-Score and Benchmark performance
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=args.figsize)

        # Plot 1: PG Score vs Data Scale
        for lang_name, group in df_plot.groupby("name"):
            style = language_styles.get(lang_name, {})
            ax1.plot(
                group["scale"],
                group["pg_score"],
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                label=lang_name,
                color=style.get("color", COLORS.get("slate_3")),
                linewidth=2,
                markersize=8,
                alpha=0.8,
            )

        ax1.set_xscale("log")
        ax1.set_xlabel("Num. samples, log")
        ax1.set_ylabel("PG-Score")
        ax1.grid(True, which="both", linestyle="--", alpha=0.3)

        # Plot 2: Result vs Data Scale (with error bars)
        for lang_name, group in df_plot.groupby("name"):
            style = language_styles.get(lang_name, {})
            ax2.errorbar(
                group["scale"],
                group["result"],
                yerr=group["result_stderr"],
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                label=lang_name,
                color=style.get("color", COLORS.get("slate_3")),
                linewidth=2,
                markersize=8,
                alpha=0.8,
                capsize=5,
                capthick=2,
            )

        ax2.set_xscale("log")
        ax2.set_xlabel("Num. samples, log")
        ax2.set_ylabel("Avg. Multilingual Performance")
        ax2.grid(True, which="both", linestyle="--", alpha=0.3)

        # Add single legend below the plots
        handles, labels = ax1.get_legend_handles_labels()
        fig = plt.gcf()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
            frameon=False,
        )

    # Save figure
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output_path, bbox_inches="tight")
    print(f"Saved figure to {args.output_path}")


if __name__ == "__main__":
    main()
