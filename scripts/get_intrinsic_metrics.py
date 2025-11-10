import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset

from scripts.utils.prompts import MR3_EVAL_PROMPT_TEMPLATE

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_intrinsic_metrics():
    """Registry of intrinsic metrics. See implementation of each metric below."""
    return {
        "distinct_ri": _compute_distinct_ri,
        "perplexity": _compute_perplexity,
        "reward_model": _compute_mr3_rubric_score,
    }


def get_args():
    # fmt: off
    description = "Get intrinsic metrics for a given dataset."
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dataset", type=str, required=True, help="HuggingFace dataset to compute intrinsic metrics on.")
    parser.add_argument("--output_path", type=str, required=False, help="Path to save the intrinsic metrics JSON file. If not set, will save to ./metrics/{dataset_name}_{metric}_intrinsic_metrics.json.")
    parser.add_argument("--metrics", type=str, nargs="+", choices=["all"] + list(get_intrinsic_metrics().keys()), help="Intrinsic metric to compute.")
    parser.add_argument("--metric_params", type=str, default=None, help="Additional parameters for the metric in JSON format.")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Will perform a dry run without saving any files and using a small amount of samples (1000).")
    parser.add_argument("--input_dataset_filter", type=str, default=None, help="JSON string representing a filter to apply to the input dataset before finetuning. The keys should be the field names and the values should be the values to filter by. This is an AND operation.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    metrics_to_compute = list(get_intrinsic_metrics().keys()) if "all" in args.metrics else args.metrics  # fmt: skip
    metrics_to_compute = sorted(set(metrics_to_compute))
    logging.info(f"Computing intrinsic metrics: {metrics_to_compute} for dataset: {args.input_dataset}")  # fmt: skip

    output_path = (
        Path("metrics")
        / f"{args.input_dataset.replace('/', '___')}_{'-'.join(metrics_to_compute)}_intrinsic_metrics.json"
        if not args.output_path
        else Path(args.output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.input_dataset, split="train")
    if args.input_dataset_filter:
        filter_dict = json.loads(args.input_dataset_filter)
        logging.info(f"Applying filter to dataset: {filter_dict}")
        dataset = dataset.filter(
            lambda example: all(example[k] == v for k, v in filter_dict.items())
        )
    dataset = dataset.filter(lambda example: all(example[k] is not None for k in dataset.column_names))  # fmt: skip
    if args.dry_run:
        logging.info("Dry run: using a small subset of the dataset (1000 samples).")
        dataset = dataset.shuffle().select(range(1000))

    metric_scores = {}
    for metric in metrics_to_compute:
        metric_fn = get_intrinsic_metrics()[metric]
        logging.info(f"Computing metric: {metric}")
        metric_params = json.loads(args.metric_params) if args.metric_params else {}
        score = metric_fn(dataset, args.dry_run, **metric_params)
        logging.info(f">>> {score}")
        metric_scores[metric] = score

    if not args.dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metric_scores, f, indent=4)
        logging.info(f"Saved intrinsic metrics to: {output_path}")


def _compute_distinct_ri(
    dataset: "Dataset",
    dry_run: bool,
    *,
    embedding_model: str = "nvidia/llama-embed-nemotron-8b",
) -> dict[str, float]:
    """Compute the distinctiveness of the instructions and responses in the dataset."""
    from sentence_transformers import SentenceTransformer

    if "prompt" not in dataset.column_names or "response" not in dataset.column_names:
        raise ValueError("Dataset must contain 'prompt' and 'response' fields!")
    prompts = dataset["prompt"]
    responses = dataset["response"]

    if dry_run:
        embedding_model = "google/embeddinggemma-300m"

    model = SentenceTransformer(embedding_model, trust_remote_code=True)
    metrics = {}
    for k, texts in {"prompts": prompts, "responses": responses}.items():
        embeddings = model.encode(texts, convert_to_tensor=True)
        similarity_matrix = model.similarity(embeddings, embeddings)
        similarity_matrix = torch.nan_to_num(similarity_matrix, nan=0.0)
        n = similarity_matrix.shape[0]
        mask = ~torch.eye(n, dtype=bool)
        average_similarity = similarity_matrix[mask].mean()
        metrics[f"{k}_distinct_ri"] = 1 - average_similarity.item()
    return metrics


def _compute_perplexity(
    dataset,
    *,
    base_model: str = "google/gemma-3-4b-pt",
    batch_size: int = 8,
    save_all_results: bool = True,
) -> dict[str, float]:
    """Compute the perplexity of the responses in the dataset."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    total_perplexity = 0.0
    results = []

    instances = [
        {"prompt": example["prompt"], "response": example["response"]}
        for example in dataset
    ]

    # Reference: https://github.com/neulab/data-agora/blob/d7e66e12b03616caa42d818d5c2a387c127014ab/libs/data-agora/data_agora/core/intrinsic_evaluators.py#L127
    with torch.no_grad():
        for i in range(0, len(instances), batch_size):
            batch = instances[i : i + batch_size]

            batch_inputs = []
            batch_instruction_lens = []
            for instance in batch:
                instruction = instance["prompt"]
                response = instance["response"]
                batch_inputs.append(instruction + response)
                batch_instruction_lens.append(len(tokenizer(instruction)["input_ids"]))

            # Tokenize batch
            inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True).to(device)  # fmt: skip
            labels = inputs["input_ids"].clone()

            # Mask out loss for instruction tokens for each sequence in batch
            for j, inst_len in enumerate(batch_instruction_lens):
                labels[j, :inst_len] = -100

            outputs = model(**inputs, labels=labels)
            losses = outputs.loss.item()

            for j, instance in enumerate(batch):
                perplexity = torch.exp(torch.tensor(losses)).item()
                total_perplexity += perplexity

                result = {
                    "instruction": instance["instruction"],
                    "response": instance["response"],
                    "perplexity": perplexity,
                }

                results.append(result)

    metrics = {"perplexity": total_perplexity / len(instances)}
    if save_all_results:
        metrics["per_instance_perplexity"] = results

    return metrics


def _compute_mr3_rubric_score(
    dataset, *, model="rubricreward/mR3-Qwen3-14B-tgt-prompt-en-thinking"
):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # Setup model. Reference: https://github.com/rubricreward/mr3?tab=readme-ov-file#-using-vllm
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=16384, min_p=0, top_k=20)  # fmt: skip
    llm = LLM(model=model, dtype="bfloat16", max_model_len=32768)

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model)
    list_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switch between thinking and non-thinking modes.
    )

    outputs = llm.generate(list_text, sampling_params)
    print(outputs[0].output.text)
    # TODO: Implement MR3 rubric score computation


if __name__ == "__main__":
    main()
