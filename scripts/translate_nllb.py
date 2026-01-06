import argparse
import sys
import logging
import json

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
    parser.add_argument("-o", "--output_dataset", type=str, required=True, help="HuggingFace dataset to store the outputs.")
    parser.add_argument("-i", "--input_dataset", type=str, default="ljvmiranda921/tulu-3-sft-subsampled-english-only", help="HuggingFace dataset to translate.")
    parser.add_argument("--translate_model", type=str, default="facebook/nllb-200-3.3B", help="The NLLB model to use for translation.")
    parser.add_argument("--teacher_model", type=str, default="google/gemma-3-27b-it", help="The teacher model to use for generating responses after translation.")
    parser.add_argument("--prompts_key", type=str, default="prompt", help="Field containing the prompt to translate.")
    parser.add_argument("--responses_key", type=str, default="response", help="Field containing the response to translate.")
    parser.add_argument("--strategy", choices=["translate_baseline", "translate_then_respond", "translate_both"], required=True, help="The synthesis strategy to use.")
    parser.add_argument("--append", action="store_true", help="If set, will append to existing output dataset instead of overwriting.")
    parser.add_argument("-l", "--target_lang", type=str, required=True, help="The ISO-2 target language code.")
    parser.add_argument("--limit", default=None, help="If set, then will only run the synthesis strategy on the first N instances.")
    parser.add_argument("--shuffle", default=None, help="If set, will shuffle the dataset using the seed provided before synthesizing. If --limit is set, then THIS command will be run first before shuffling.")
    parser.add_argument("--backend_params", type=str, default=None, help="If set, will pass these additional parameters (in JSON format) to the backend LLM inference calls.")
    parser.add_argument("--generation_params", type=str, default="{'temperature': 0.8, 'top_p': 0.9}", help="If set, will pass these additional generation parameters (in JSON format) to the LLM generation calls.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    # Must not be msde-S1 so that apply_subsampling works correctly
    if "msde-S1" in args.output_dataset:
        raise ValueError("Output dataset cannot be msde-S1-* when using NLLB translation.")  # fmt: skip
    df = load_dataset(args.input_dataset, split="train").to_pandas()

    backend_params = json.loads(args.backend_params) if args.backend_params else None
    generation_params = json.loads(args.generation_params) if args.generation_params else None  # fmt: skip

    if args.shuffle:
        logging.info(f"Shuffling the dataset using seed {args.shuffle}")
        dataset = dataset.shuffle(seed=int(args.shuffle))
    if args.limit:
        logging.info(f"Getting the first {args.limit} instances")
        dataset = dataset.select(range(min(int(args.limit), len(dataset))))


def convert_to_nllb_code(lang_code: str) -> str:
    """Convert ISO language code to NLLB format."""
    lang = Language.get(lang_code)
    lang_3 = lang.to_alpha3()
    script = lang.assume_script().script
    return f"{lang_3}_{script}"


if __name__ == "__main__":
    main()
