import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot.utils.plot_theme import PLOT_PARAMS, COLORS, OUTPUT_DIR
from plot.utils.metadata import MODEL_INFORMATION, LANGUAGE_INFORMATION

plt.rcParams.update(PLOT_PARAMS)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot model scale")
    parser.add_argument("--input_path", type=Path, required=True, help="Results JSONL file to plot. Must contain the fields `teacher_model`, `target_lang`, and `pg_score`.")
    parser.add_argument("--output_path", type=Path, default=OUTPUT_DIR / "model_scale.pdf", help="Path to save the outputs.")
    parser.add_argument("--figsize", type=lambda s: tuple(map(int, s.split(","))), default=(8, 6), help="Figure size as WIDTH,HEIGHT in inches. Default: 6,8")
    parser.add_argument("--average", action="store_true", help="Plot average pg_score per model instead of individual language points.")
    parser.add_argument("--size_by", type=str, choices=["joshi_etal_resource_level", "pct_commoncrawl", "native_speakers_in_m"], default="pct_commoncrawl", help="Vary marker size based on language metadata (only for non-average mode).")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df = pd.read_json(args.input_path, lines=True)

    df_models = pd.DataFrame([m.model_dump() for m in MODEL_INFORMATION])
    df_models["teacher_model"] = df_models["name"].str.split("/").str[-1]

    df_langs = pd.DataFrame([lang.model_dump() for lang in LANGUAGE_INFORMATION])

    df_plot = df.merge(
        df_models[["teacher_model", "parameter_size"]],
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

    # Aggregate if averaging
    if args.average:
        df_plot = df_plot.groupby(["teacher_model", "model_size"], as_index=False)[
            "pg_score"
        ].mean()

    # Determine marker sizes
    if args.size_by and not args.average:
        # Scale marker sizes based on the selected attribute
        # Map resource level (3-5) to marker sizes (50-200) for more pronounced differences
        size_values = df_plot[args.size_by].values
        marker_sizes = 50 + (size_values - size_values.min()) * 150 / (
            size_values.max() - size_values.min()
        )
    else:
        marker_sizes = 100

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=args.figsize)
    ax.scatter(df_plot["model_size"], df_plot["pg_score"], alpha=0.6, s=marker_sizes)

    ax.set_xscale("log")
    ax.set_xlabel("Model Size (parameters, log scale)")
    ax.set_ylabel("PG-Score")

    plt.tight_layout()
    plt.savefig(args.output_path, bbox_inches="tight")
    print(f"Plot saved to {args.output_path}")


if __name__ == "__main__":
    main()
