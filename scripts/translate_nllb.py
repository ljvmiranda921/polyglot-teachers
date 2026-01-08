import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import ctranslate2
import torch
from bespokelabs.curator.types.curator_response import CuratorResponse
from datasets import Dataset, load_dataset
from langcodes import Language
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# For some reason, vllm must be imported before transformers
# https://github.com/vllm-project/vllm/issues/17618
# may the lord have mercy
import vllm
from scripts.synthesize_data import (
    filter_by_token_length,
    prepare_output_dataset,
    upload_to_huggingface,
)
from scripts.utils.llm_inference import get_strategy
from scripts.utils.prompts import SYSTEM_PROMPT

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

CTRANSLATE_MODELS_DIR = Path("ctranslate2")


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
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for NLLB translation.")
    parser.add_argument("--generation_params", type=str, default=None, help="If set, will pass these additional generation parameters (in JSON format) to the LLM generation calls.")
    parser.add_argument("--device", type=str, default="cuda", help="The device to run NLLB translation on.")
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
                model_name = args.teacher_model
            case "nllb_translate_then_respond":
                format_fn, distiller_fn = get_strategy(name="respond")
                model_name = f"{args.translate_model} + {args.teacher_model}"

        if args.strategy == "translate":
            # I still want to see the original English versions
            df = dataset.to_pandas().rename(
                columns={
                    args.prompts_key: "prompt_en",
                    args.responses_key: "response_en",
                }
            )
            df["prompt"] = df["prompt_en"].to_list()  # copy so the template works
            dataset = Dataset.from_pandas(df)

        if args.strategy == "nllb_translate_then_respond":
            df = dataset.to_pandas().rename(
                columns={
                    args.prompts_key: "prompt_en",
                    args.responses_key: "response_en",
                }
            )
            # Translate prompts from English to target language
            texts = df["prompt_en"].tolist()
            df["prompt"] = nllb_translate(
                texts,
                model_name=args.translate_model,
                tgt_lang=lang_with_script,
                batch_size=args.batch_size,
                device=args.device,
            )
            dataset = Dataset.from_pandas(df)

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
            model_name=args.teacher_model,
            batch=False,
            system_prompt=system_prompt,
            backend="vllm",
            backend_params=backend_params,
            generation_params=generation_params,
        )
        curator_response: CuratorResponse = distiller(input_dataset)
        logging.info(f"Data synthesis cost: {curator_response.cost_info.total_cost} USD")  # fmt: skip

    else:
        model_name = args.translate_model
        df = dataset.to_pandas().rename(
            columns={
                args.prompts_key: "prompt_en",
                args.responses_key: "response_en",
            }
        )
        df["prompt"] = nllb_translate(
            df["prompt_en"].tolist(),
            model_name=args.translate_model,
            tgt_lang=lang_with_script,
            batch_size=args.batch_size,
            device=args.device,
        )
        df["response"] = nllb_translate(
            df["response_en"].tolist(),
            model_name=args.translate_model,
            tgt_lang=lang_with_script,
            batch_size=args.batch_size,
            device=args.device,
        )
        dataset = Dataset.from_pandas(df)

    output_dataset = prepare_output_dataset(
        curator_response.dataset,
        input_dataset=input_dataset,
        strategy=args.strategy,
        model=model_name,
        include_input_columns=True,
    )
    output_dataset = output_dataset.filter(lambda ex: ex["response"] is not None)

    # Upload output to HuggingFace
    logging.info(f"Uploading output dataset to HuggingFace: {args.output_dataset}")
    upload_to_huggingface(dataset=output_dataset, dataset_name=args.output_dataset, append=args.append)  # fmt: skip


def nllb_translate(
    texts: list[str],
    model_name: str,
    tgt_lang: str,
    max_length: int = 1024,
    batch_size: int = 128,
    device: str = "cuda",
) -> list[str]:
    """Translate a list of texts using NLLB model with CTranslate2."""

    # If model_name is a local CTranslate2 path, derive the HuggingFace model name for tokenizer
    if model_name.startswith(str(CTRANSLATE_MODELS_DIR)):
        # Extract model variant from path (e.g., "ctranslate2/ct2-nllb-200-3.3B" -> "nllb-200-3.3B")
        model_variant = model_name.split("ct2-")[-1]
        hf_model_name = f"facebook/{model_variant}"
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    else:
        # Use the model_name directly for both tokenizer and translator
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    translator = ctranslate2.Translator(model_name, device=device, compute_type="float16")  # fmt: skip

    translated_texts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch_texts = texts[i : i + batch_size]
        tokenized = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        source_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in tokenized["input_ids"]]  # fmt: skip
        target_prefix = [[tgt_lang]] * len(batch_texts)
        results = translator.translate_batch(
            source_tokens,
            target_prefix=target_prefix,
            max_batch_size=batch_size,
            beam_size=1,
            max_decoding_length=max_length,
        )
        for result in results:
            tokens = result.hypotheses[0]
            translated_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)  # fmt: skip
            translated_texts.append(translated_text)

    logging.info("Deleting translator to free up memory.")
    del translator
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    logging.info(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB. Sleeping for 30 seconds...")  # fmt: skip
    time.sleep(30)

    logging.info(f"Sample translations: {translated_texts[:5]}")
    return translated_texts


def convert_to_nllb_code(lang_code: str) -> str:
    """Convert ISO language code to NLLB format."""
    if lang_code == "ar":
        return "arb_Arab"  # Modern Standard Arabic

    lang = Language.get(lang_code)
    lang_3 = lang.to_alpha3()
    script = lang.assume_script().script
    return f"{lang_3}_{script}"


if __name__ == "__main__":
    main()
