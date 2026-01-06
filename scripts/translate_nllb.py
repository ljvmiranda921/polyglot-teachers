import argparse
import sys
import logging

from datasets import load_dataset
from pathlib import Path
from langcodes import Language


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Translate a dataset using NLLB model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_dataset", type=str, required=True, help="HuggingFace dataset to translate.")
    parser.add_argument("-o", "--output_dataset", type=str, required=True, help="HuggingFace dataset to store the outputs.")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-3.3B", help="The NLLB model to use for translation.")
    parser.add_argument("--target_lang", type=str, required=True, help="The ISO-2 target language code.")
    # fmt: on
    return parser.parse_args()


def main():
    pass


def convert_to_nllb_code(lang_code: str) -> str:
    """Convert ISO language code to NLLB format."""
    lang = Language.get(lang_code)
    lang_3 = lang.to_alpha3()
    script = lang.assume_script().script
    return f"{lang_3}_{script}"


if __name__ == "__main__":
    main()
