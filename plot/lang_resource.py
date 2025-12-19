import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot.utils.plot_theme import PLOT_PARAMS, COLORS, OUTPUT_DIR, FONT_SIZES
from plot.utils.metadata import MODEL_INFORMATION, LANGUAGE_INFORMATION

plt.rcParams.update(PLOT_PARAMS)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot language resource vs PG-Score")
    parser.add_argument("--input_path", type=Path, required=True, help="Results JSONL file to plot. Must contain the fields `teacher_model`, `target_lang`, and `pg_score`.")
    parser.add_argument("--output_path", type=Path, default=OUTPUT_DIR / "lang_resource.pdf", help="Path to save the outputs.")
    parser.add_argument("--figsize", type=lambda s: tuple(map(int, s.split(","))), default=(12, 7), help="Figure size as WIDTH,HEIGHT in inches. Default: 12,7")
    parser.add_argument("--resource_by", type=str, choices=["pct_commoncrawl", "native_speakers_in_m", "joshi_etal_resource_level"], default="pct_commoncrawl", help="Language resource metric to use for x-axis.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df = pd.read_json(args.input_path, lines=True)

    df_models = pd.DataFrame([m.model_dump() for m in MODEL_INFORMATION])
    df_models["teacher_model"] = df_models["name"].str.split("/").str[-1]
    df_langs = pd.DataFrame([lang.model_dump() for lang in LANGUAGE_INFORMATION])
    df_plot = df.merge(
        df_models[["teacher_model", "parameter_size", "beautiful_name"]],
        on="teacher_model",
        how="left",
    ).merge(
        df_langs,
        left_on="target_lang",
        right_on="iso_639_1",
        how="left",
    )
    df_plot = df_plot[df_plot["parameter_size"] != "Unknown"].copy()
    df_plot["model_size"] = df_plot["parameter_size"].astype(float)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=args.figsize)

    # Aggregate statistics across models for each resource level
    agg_stats = df_plot.groupby(args.resource_by).agg(
        mean_pg_score=("pg_score", "mean"),
        q25_pg_score=("pg_score", lambda x: x.quantile(0.25)),
        q75_pg_score=("pg_score", lambda x: x.quantile(0.75)),
    ).reset_index()

    # Sort by resource level for proper line plotting
    agg_stats = agg_stats.sort_values(args.resource_by)

    # Plot gray shaded area (IQR: 25th-75th percentile)
    ax.fill_between(
        agg_stats[args.resource_by],
        agg_stats["q25_pg_score"],
        agg_stats["q75_pg_score"],
        color=COLORS.get("slate_2"),
        alpha=0.3,
        label="Model range (IQR)",
    )

    # Plot red average line
    ax.plot(
        agg_stats[args.resource_by],
        agg_stats["mean_pg_score"],
        marker="o",
        markersize=10,
        linewidth=3,
        color=COLORS.get("dark_crest"),
        label="Average across models",
        zorder=10,
    )

    # Set labels
    x_label = {
        "pct_commoncrawl": "CommonCrawl Coverage (\%)",
        "native_speakers_in_m": "Native Speakers (millions)",
        "joshi_etal_resource_level": "Joshi et al. Resource Level",
    }.get(args.resource_by, args.resource_by)

    ax.set_xlabel(x_label)
    ax.set_ylabel("PG-Score")

    # Aesthetics
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="best", framealpha=0.9, ncol=2)

    # Save figure
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output_path)
    print(f"Saved figure to {args.output_path}")


if __name__ == "__main__":
    main()
