import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from plot.utils.metadata import LANGUAGE_INFORMATION, MODEL_INFORMATION
from plot.utils.plot_theme import COLORS, FONT_SIZES, OUTPUT_DIR, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot language resource correlation with PG-scores")
    parser.add_argument("--input_path", type=Path, required=True, help="Results JSONL file to plot. Must contain the fields `teacher_model`, `target_lang`, and `pg_score`.")
    parser.add_argument("--output_path", type=Path, default=OUTPUT_DIR / "language_correl.pdf", help="Path to save the outputs.")
    parser.add_argument("--figsize", type=lambda s: tuple(map(int, s.split(","))), default=(6, 8), help="Figure size as WIDTH,HEIGHT in inches. Default: 14,7")
    parser.add_argument("--property", type=str, choices=["pct_commoncrawl", "native_speakers_in_m", "joshi_etal_resource_level"], default="pct_commoncrawl", help="Language property to use on x-axis.")
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

    df_plot = df_plot.dropna(subset=[args.property]).copy()
    if args.property == "pct_commoncrawl":
        bins = [0, 1, 2, 5, 10]
        labels = [r"$<$1\%", "1--2\%", "2--5\%", r"$>$5\%"]
        xlabel = "Percentage of a Language\n in CommonCrawl"
    elif args.property == "native_speakers_in_m":
        bins = [0, 50, 150, 600]
        labels = [r"$<$50M", "50--150M", r"$>$150M"]
        xlabel = "Native Speakers (millions)"
    else:
        df_plot["resource_bin"] = df_plot[args.property].astype(str)
        labels = sorted(df_plot["resource_bin"].unique())
        xlabel = "Resource Level (Joshi et al., 2020)"

    if args.property != "joshi_etal_resource_level":
        df_plot["resource_bin"] = pd.cut(
            df_plot[args.property],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

        box_data = []
        tick_labels = []
        for label in labels:
            subset = df_plot[df_plot["resource_bin"] == label]
            if len(subset) > 0:
                box_data.append(subset["pg_score"].tolist())
                tick_labels.append(label)
        positions = range(len(box_data))

    fig, ax = plt.subplots(1, 1, figsize=args.figsize)
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        notch=False,
        # fmt: off
        boxprops=dict(facecolor=COLORS.get("warm_blue"), color=COLORS.get("dark_blue"), alpha=0.7),
        whiskerprops=dict(color=COLORS.get("dark_blue"), linewidth=1.5),
        capprops=dict(color=COLORS.get("dark_blue"), linewidth=1.5),
        medianprops=dict(color=COLORS.get("dark_crest"), linewidth=2.5),
        flierprops=dict(marker="o", markerfacecolor=COLORS.get("slate_3"), markersize=6, alpha=0.5, markeredgecolor="none"),
        # fmt: on
    )

    for i, (label, data) in enumerate(zip(tick_labels, box_data)):
        ax.scatter(i, data, alpha=0.3, s=40, color=COLORS.get("dark_blue"), zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PG-Score")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    df_corr = df_plot.groupby("target_lang", as_index=False).agg({args.property: "first", "pg_score": "mean"})  # fmt: skip
    rho, p_value = spearmanr(df_corr[args.property], df_corr["pg_score"])

    if p_value < 0.001:
        p_text = r"$p < 0.001$"
    elif p_value < 0.01:
        p_text = r"$p < 0.01$"
    elif p_value < 0.05:
        p_text = r"$p < 0.05$"
    else:
        p_text = f"$p = {p_value:.3f}$"

    ax.text(
        0.05,
        0.95,
        f"$\\rho$ = {rho:.3f}, {p_text}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=FONT_SIZES.get("large"),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="none"),  # fmt: skip
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output_path, bbox_inches="tight")
    print(f"Saved figure to {args.output_path}")


if __name__ == "__main__":
    main()
