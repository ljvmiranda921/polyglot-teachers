import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot.utils.plot_theme import PLOT_PARAMS, COLORS, OUTPUT_DIR
from plot.utils.model_info import MODEL_INFORMATION

plt.rcParams.update(PLOT_PARAMS)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot model scale")
    parser.add_argument("--input_path", type=Path, required=True, help="Results JSONL file to plot. Must contain the fields `teacher_model`, `target_lang`, and `pg_score`.")
    parser.add_argument("--output_path", type=Path, default=OUTPUT_DIR / "model_scale.pdf", help="Path to save the outputs.")
    parser.add_argument("--figsize", type=lambda s: tuple(map(int, s.split(","))), default=(6, 8), help="Figure size as WIDTH,HEIGHT in inches. Default: 6,8")
    parser.add_argument("--average", action="store_true", help="Plot average pg_score per model instead of individual language points.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df = pd.read_json(args.input_path, lines=True)

    df_models = pd.DataFrame([m.model_dump() for m in MODEL_INFORMATION])
    df_models["teacher_model"] = df_models["name"].str.split("/").str[-1]
    df_plot = df.merge(
        df_models[["teacher_model", "parameter_size"]],
        on="teacher_model",
        how="left",
    )
    df_plot = df_plot[df_plot["parameter_size"] != "Unknown"].copy()
    df_plot["model_size"] = df_plot["parameter_size"].astype(float)

    # Aggregate if averaging
    if args.average:
        df_plot = df_plot.groupby(["teacher_model", "model_size"], as_index=False)[
            "pg_score"
        ].mean()

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=args.figsize)
    ax.scatter(df_plot["model_size"], df_plot["pg_score"], alpha=0.6, s=50)

    ax.set_xscale("log")
    ax.set_xlabel("Model Size (parameters, log scale)")
    ax.set_ylabel("PG-Score")

    plt.tight_layout()
    plt.savefig(args.output_path, bbox_inches="tight")
    print(f"Plot saved to {args.output_path}")


if __name__ == "__main__":
    main()
