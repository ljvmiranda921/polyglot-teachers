import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot.utils.plot_theme import PLOT_PARAMS, COLORS, OUTPUT_DIR

plt.rcParams.update(PLOT_PARAMS)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot model scale")
    parser.add_argument("--input_path", type=Path, required=True, help="Results JSONL file to plot. Must contain the fields `teacher_model`, `target_lang`, and `pg_score`.")
    parser.add_argument("--output_path", type=Path, default=OUTPUT_DIR / "model_scale.pdf", help="Path to save the outputs.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    df = pd.read_jon(args.input_path, lines=True)
    breakpoint()

    fig, ax = plt.subplots(1, 1, figsize=(6, 8))


if __name__ == "__main__":
    main()
