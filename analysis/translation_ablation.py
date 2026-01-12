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
    parser.add_argument("--figsize", type=lambda s: tuple(map(int, s.split(","))), default=(10, 8), help="Figure size as WIDTH,HEIGHT in inches. Default: 10,8")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df = pd.read_json(args.input_path, lines=True)

    methods = ["nllb-translate-both", "translate-then-respond", "translate-synthetic"]
    langs = ["ar", "id", "de"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=args.figsize)

    x = range(len(methods))
    width = 0.6

    avg_values = []
    for method in methods:
        method_data = df[df["translate_method"] == method]
        avg_values.append(method_data["pg_score"].mean())

    ax1.bar(x, avg_values, width, color=COLORS["warm_blue"])
    for i, (method, val) in enumerate(zip(methods, avg_values)):
        ax1.text(i, val + 0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=FONT_SIZES["small"])

    ax1.set_ylabel("Average PG-Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)

    width = 0.25
    offsets = [-width, 0, width]
    colors_list = [COLORS["warm_blue"], COLORS["warm_crest"], COLORS["warm_green"]]

    for i, lang in enumerate(langs):
        lang_data = df[df["target_lang"] == lang]
        values = []
        for method in methods:
            method_data = lang_data[lang_data["translate_method"] == method]
            values.append(method_data["pg_score"].values[0])

        positions = [xi + offsets[i] for xi in x]
        ax2.bar(positions, values, width, label=lang, color=colors_list[i])

    ax2.set_ylabel("PG-Score")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha="right")
    ax2.legend(loc="best")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_path)
    logging.info(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()
