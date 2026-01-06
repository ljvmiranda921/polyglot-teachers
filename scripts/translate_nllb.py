import argparse
import sys
import logging
import json

import torch
from datasets import load_dataset
from pathlib import Path
from langcodes import Language
from datasets import Dataset
from bespokelabs.curator.types.curator_response import CuratorResponse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline

from scripts.utils.llm_inference import get_strategy
from scripts.utils.prompts import SYSTEM_PROMPT
from scripts.synthesize_data import (
    filter_by_token_length,
    prepare_output_dataset,
    upload_to_huggingface,
)


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
    parser.add_argument("--strategy", choices=["translate", "nllb_translate_then_respond", "nllb_translate_both"], required=True, help="The synthesis strategy to use.")
    parser.add_argument("--append", action="store_true", help="If set, will append to existing output dataset instead of overwriting.")
    parser.add_argument("-l", "--target_lang", type=str, required=True, help="The ISO-2 target language code.")
    parser.add_argument("--limit", default=None, help="If set, then will only run the synthesis strategy on the first N instances.")
    parser.add_argument("--shuffle", default=None, help="If set, will shuffle the dataset using the seed provided before synthesizing. If --limit is set, then THIS command will be run first before shuffling.")
    parser.add_argument("--backend_params", type=str, default=None, help="If set, will pass these additional parameters (in JSON format) to the backend LLM inference calls.")
    parser.add_argument("--generation_params", type=str, default=None, help="If set, will pass these additional generation parameters (in JSON format) to the LLM generation calls.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    use_lm = False  # tracks whether we'll use an LM
    # Must not be msde-S1 so that apply_subsampling works correctly
    if "msde-S1" in args.output_dataset:
        raise ValueError("Output dataset cannot be msde-S1-* when using NLLB translation.")  # fmt: skip
    dataset = load_dataset(args.input_dataset, split="train")

    if args.shuffle:
        logging.info(f"Shuffling the dataset using seed {args.shuffle}")
        dataset = dataset.shuffle(seed=int(args.shuffle))
    if args.limit:
        logging.info(f"Getting the first {args.limit} instances")
        dataset = dataset.select(range(min(int(args.limit), len(dataset))))
    if args.strategy in ("translate", "nllb_translate_then_respond"):
        logging.info(f"Will load an LM ({args.teacher_model}) due to chosen strategy.")
        backend_params = json.loads(args.backend_params) if args.backend_params else None  # fmt: skip
        generation_params = json.loads(args.generation_params) if args.generation_params else None  # fmt: skip
        use_lm = True

    lang_name = Language.make(args.target_lang).display_name()
    lang_with_script = convert_to_nllb_code(args.target_lang)
    if "Unknown language" in lang_name:
        raise ValueError(f"Unknown language: {args.target_lang}. Please input a two-letter ISO 693-2 code.")  # fmt: skip

    logging.info(f"Using '{args.strategy}' synthesis strategy")
    logging.info(f"No. of instances: {len(dataset)}")

    if use_lm:
        match args.strategy:
            case "translate":
                format_fn, distiller_fn = get_strategy(name=args.strategy)
            case "nllb_translate_then_respond":
                format_fn, distiller_fn = get_strategy(name="respond")

        if args.strategy == "nllb_translate_then_respond":
            df = dataset.to_pandas().rename(
                columns={
                    args.prompts_key: "prompt_en",
                    args.responses_key: "response_en",
                }
            )
            # Translate prompts from English to target language
            texts = df["prompt_en"].tolist()
            nllb_translate(
                texts,
                model_name=args.translate_model,
                lang_code=lang_with_script,
            )

            input_dataset = (
                None  # TODO: replace input_dataset with nllb-translated version
            )

        input_dataset: Dataset = format_fn(dataset, lang_name=lang_name)
        system_prompt = SYSTEM_PROMPT.format(lang_name=lang_name)
        if backend_params and "max_model_length" in backend_params:
            max_model_len = int(backend_params.get("max_model_length"))
            input_dataset = filter_by_token_length(
                input_dataset,
                max_model_len,
                system_prompt=system_prompt,
                prompt_key="synth_prompt",
            )

        distiller = distiller_fn(
            model_name=args.model,
            batch=args.batch_mode,
            system_prompt=system_prompt,
            backend=args.backend,
            backend_params=backend_params,
            generation_params=generation_params,
        )
        curator_response: CuratorResponse = distiller(input_dataset)
        logging.info(f"Data synthesis cost: {curator_response.cost_info.total_cost} USD")  # fmt: skip

        output_dataset = prepare_output_dataset(
            curator_response.dataset,
            input_dataset=input_dataset,
            strategy=args.strategy,
            model=args.model,
        )
        output_dataset = output_dataset.filter(lambda ex: ex["response"] is not None)

        # Upload output to HuggingFace
        logging.info(f"Uploading output dataset to HuggingFace: {args.output_dataset}")
        upload_to_huggingface(
            dataset=output_dataset,
            dataset_name=args.output_dataset,
            append=args.append,
            drop_columns_from_input=None,
        )

    else:
        pass


def nllb_translate(
    texts: list[str], model_name: str, lang_code: str, max_length: int = 1024
) -> list[str]:
    hf_pipeline = pipeline(
        task="translation",
        model=model_name,
        src_lang="eng_Latn",
        tgt_lang="fra_Latn",
        dtype=torch.float16,
        device=0,
    )
    breakpoint()
    inputs = tokenizer(texts, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(lang_code),
        max_length=max_length,
    )
    translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]  # fmt: skip
    breakpoint()
    return translated_texts


def convert_to_nllb_code(lang_code: str) -> str:
    """Convert ISO language code to NLLB format."""
    lang = Language.get(lang_code)
    lang_3 = lang.to_alpha3()
    script = lang.assume_script().script
    return f"{lang_3}_{script}"


if __name__ == "__main__":
    main()
