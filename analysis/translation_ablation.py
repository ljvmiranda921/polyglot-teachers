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
    pass


def main():
    pass


if __name__ == "__main__":
    main()
