import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils.plot_theme import COLORS, OUTPUT_DIR, PLOT_PARAMS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

plt.rcParams.update(PLOT_PARAMS)


def get_args():
    parser = argparse.ArgumentParser(description="Draw ablation results for Tagalog.")
    # fmt: off
    parser.add_argument("--input_dir", type=Path, help="Directory containing ablation result JSON files.", required=True)
    parser.add_argument("--figsize", type=lambda s: tuple(map(int, s.split(","))), default=(12, 7), help="Figure size as WIDTH,HEIGHT in inches. Default: 6,8")
    # fmt: on
    return parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
