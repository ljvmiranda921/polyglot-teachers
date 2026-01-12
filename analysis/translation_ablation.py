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
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save the output plot.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()


if __name__ == "__main__":
    main()
