import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot.utils.plot_theme import PLOT_PARAMS, COLORS, OUTPUT_DIR, FONT_SIZES
from plot.utils.metadata import MODEL_INFORMATION, LANGUAGE_INFORMATION

plt.rcParams.update(PLOT_PARAMS)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot language resource correlation with PG-scores")
    parser.add_argument("--input_path", type=Path, required=True, help="Results JSONL file to plot. Must contain the fields `teacher_model`, `target_lang`, and `pg_score`.")
    parser.add_argument("--output_path", type=Path, default=OUTPUT_DIR / "language_correl.pdf", help="Path to save the outputs.")
    parser.add_argument("--figsize", type=lambda s: tuple(map(int, s.split(","))), default=(8, 6), help="Figure size as WIDTH,HEIGHT in inches. Default: 14,7")
    parser.add_argument("--property", type=str, choices=["pct_commoncrawl", "native_speakers_in_m", "joshi_etal_resource_level"], default="pct_commoncrawl", help="Language property to use on x-axis.")
    parser.add_argument("--use_average", action="store_true", help="Use average PG-score per language instead of showing all models separately.")
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

    # Filter out rows with missing language properties
    df_plot = df_plot.dropna(subset=[args.property]).copy()

    # Create bins for the language property
    if args.property == "pct_commoncrawl":
        bins = [0, 1, 5, 10]
        labels = ["<1%", "1-5%", ">5%"]
        xlabel = "Percentage in CommonCrawl"
    elif args.property == "native_speakers_in_m":
        bins = [0, 50, 150, 600]
        labels = ["<50M", "50-150M", ">150M"]
        xlabel = "Native Speakers (millions)"
    else:  # joshi_etal_resource_level
        # For resource level, use discrete values
        df_plot["resource_bin"] = df_plot[args.property].astype(str)
        labels = sorted(df_plot["resource_bin"].unique())
        xlabel = "Resource Level (Joshi et al.)"

    # Create bins if not using discrete resource levels
    if args.property != "joshi_etal_resource_level":
        df_plot["resource_bin"] = pd.cut(
            df_plot[args.property],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

    # Aggregate if using average
    if args.use_average:
        df_plot = df_plot.groupby(["resource_bin"], as_index=False)["pg_score"].mean()
        box_data = [[row["pg_score"]] for _, row in df_plot.iterrows()]
        positions = range(len(df_plot))
        tick_labels = df_plot["resource_bin"].tolist()
    else:
        # Prepare data for box plot (group by resource bin)
        box_data = []
        tick_labels = []
        for label in labels:
            subset = df_plot[df_plot["resource_bin"] == label]
            if len(subset) > 0:
                box_data.append(subset["pg_score"].tolist())
                tick_labels.append(label)
        positions = range(len(box_data))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=args.figsize)

    # Create box plot
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        notch=False,
        boxprops=dict(
            facecolor=COLORS.get("warm_blue"), color=COLORS.get("dark_blue"), alpha=0.7
        ),
        whiskerprops=dict(color=COLORS.get("dark_blue"), linewidth=1.5),
        capprops=dict(color=COLORS.get("dark_blue"), linewidth=1.5),
        medianprops=dict(color=COLORS.get("dark_crest"), linewidth=2.5),
        flierprops=dict(
            marker="o",
            markerfacecolor=COLORS.get("slate_3"),
            markersize=6,
            alpha=0.5,
            markeredgecolor="none",
        ),
    )

    # Add scatter points for individual observations if not averaging
    if not args.use_average:
        import numpy as np

        for i, (label, data) in enumerate(zip(tick_labels, box_data)):
            # Add jitter to x-coordinates for better visualization
            x = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(
                x,
                data,
                alpha=0.3,
                s=40,
                color=COLORS.get("dark_blue"),
                zorder=3,
            )

    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PG-Score")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Save figure
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output_path)
    print(f"Saved figure to {args.output_path}")


if __name__ == "__main__":
    main()
