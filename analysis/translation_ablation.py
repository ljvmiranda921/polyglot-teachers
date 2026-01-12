import argparse
import logging
import sys
from pathlib import Path

from analysis.utils.plot_theme import FONT_SIZES, PLOT_PARAMS, COLORS, OUTPUT_DIR
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(PLOT_PARAMS)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Analyze translation ablation results and plot a bar chart.")
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the iput JSONL file containing the results. Must have a 'translate_method' field.")
    parser.add_argument("--output_path", type=Path, default=OUTPUT_DIR / "translation_ablation.pdf", help="Path to save the output plot.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df = pd.read_json(args.input_path, lines=True)

    methods = df["translate_method"].unique()
    langs = df["target_lang"].unique()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    x = range(len(methods))
    width = 0.25
    offsets = [-width, 0, width]

    colors_list = [COLORS["warm_blue"], COLORS["warm_orange"], COLORS["warm_green"]]

    for ax, metric, ylabel in [
        (ax1, "pg_score", "PG-Score"),
        (ax2, "result", "Result"),
    ]:
        for i, lang in enumerate(langs):
            lang_data = df[df["target_lang"] == lang]

            values = []
            errors = [] if metric == "result" else None

            for method in methods:
                method_data = lang_data[lang_data["translate_method"] == method]
                if not method_data.empty:
                    values.append(method_data[metric].values[0])
                    if metric == "result":
                        errors.append(method_data["result_stderr"].values[0])
                else:
                    values.append(0)
                    if metric == "result":
                        errors.append(0)

            positions = [xi + offsets[i] for xi in x]
            if metric == "result":
                ax.bar(
                    positions,
                    values,
                    width,
                    label=lang,
                    color=colors_list[i],
                    yerr=errors,
                    capsize=3,
                )
            else:
                ax.bar(positions, values, width, label=lang, color=colors_list[i])

        for j, method in enumerate(methods):
            method_data = df[df["translate_method"] == method]
            avg_val = method_data[metric].mean()
            ax.text(
                j,
                max(method_data[metric]) + 0.15,
                f"{avg_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZES["small"],
            )

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.legend(loc="best")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_path)
    logging.info(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()
