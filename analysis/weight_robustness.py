import argparse
import json
import pandas as pd
from pathlib import Path

from scipy.stats import spearmanr


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Compute weight robustness of PG-Score")
    parser.add_argument("-i", "--input_path", type=Path)
    # fmt: on
    return parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
