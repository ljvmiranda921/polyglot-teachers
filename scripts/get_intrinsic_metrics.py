import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from langcodes import Language

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
        "reward_model": _compute_rubric_score,
    }


def get_args():
    # fmt: off
    description = "Get intrinsic metrics for a given dataset."
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dataset", type=str, required=True, help="HuggingFace dataset to compute intrinsic metrics on.")
    parser.add_argument("--output_path", type=str, required=False, help="Path to save the intrinsic metrics JSON file. If not set, will save to ./metrics/{dataset_name}_{metric}_intrinsic_metrics.json.")
    parser.add_argument("--metrics", type=str, nargs="+", choices=["all"] + list(get_intrinsic_metrics().keys()), help="Intrinsic metric to compute.")
    parser.add_argument("--metric_params", type=str, default=None, help="Additional parameters for the metric in JSON format. You need to specify this as 'metric_fn::{\"param1\": value1, \"param2\": value2},metric_fn2::...'.")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Will perform a dry run without saving any files and using a small amount of samples (1000).")
    parser.add_argument("--input_dataset_filter", type=str, default=None, help="JSON string representing a filter to apply to the input dataset before finetuning. The keys should be the field names and the values should be the values to filter by. This is an AND operation.")
    parser.add_argument("--apply_subsampling", action="store_true", default=False, help="Whether to apply subsampling to the dataset before computing metrics. This is to ensure that the number of samples per strategy is roughly the same.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    metrics_to_compute = list(get_intrinsic_metrics().keys()) if "all" in args.metrics else args.metrics  # fmt: skip
    metrics_to_compute = sorted(set(metrics_to_compute))
    logging.info(f"Computing intrinsic metrics: {metrics_to_compute} for dataset: {args.input_dataset}")  # fmt: skip
    metric_params = parse_metric_params(args.metric_params) if args.metric_params else {}  # fmt: skip
    logging.info(f"Using metric parameters: {metric_params}")

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

    # Apply subsampling if specified
    if args.apply_subsampling:
        logging.info("Applying subsampling to the dataset to balance samples.")
        dataset = subsample_per_strategy(dataset)
    if args.dry_run:
        logging.info("Dry run: using a small subset of the dataset (1000 samples).")
        dataset = dataset.shuffle().select(range(1000))

    metric_scores = {}
    for metric in metrics_to_compute:
        metric_fn = get_intrinsic_metrics()[metric]
        logging.info(f"Computing metric: {metric}")
        params = metric_params.get(metric, {})
        score = metric_fn(dataset, args.dry_run, **params)
        logging.info(f">>> {score}")
        metric_scores[metric] = score

    metric_scores["metadata"] = {
        "input_dataset": args.input_dataset,
        "num_samples": len(dataset),
        "input_dataset_filter": args.input_dataset_filter,
    }

    if not args.dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metric_scores, f, indent=4)
        logging.info(f"Saved intrinsic metrics to: {output_path}")


def parse_metric_params(param_str: str) -> dict[str, dict]:
    """Parse metric parameters from a string."""
    metric_params = {}
    for item in param_str.split(","):
        metric_name, params_json = item.split("::", 1)
        metric_params[metric_name] = json.loads(params_json)
    return metric_params


def subsample_per_strategy(dataset: Dataset) -> Dataset:
    pass


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

    metrics["metadata"] = {
        "embedding_model": embedding_model,
    }

    return metrics


def _compute_perplexity(
    dataset,
    dry_run: bool = False,
    *,
    base_model: str = "google/gemma-3-4b-pt",
    batch_size: int = 8,
    save_all_results: bool = True,
) -> dict[str, float]:
    """Compute the perplexity of the responses in the dataset."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dry_run:
        base_model = "google/gemma-3-270m"

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
                    "instruction": instance["prompt"],
                    "response": instance["response"],
                    "perplexity": perplexity,
                }

                results.append(result)

    metrics = {"average_perplexity": total_perplexity / len(instances)}
    if save_all_results:
        metrics["per_instance_perplexity"] = results

    metrics["metadata"] = {
        "base_model": base_model,
        "batch_size": batch_size,
    }

    return metrics


def _compute_rubric_score(
    dataset: Dataset,
    dry_run: bool = False,
    *,
    language: str,
    model: str = "Unbabel/M-Prometheus-14B",
    tensor_parallel_size: int = 1,
    save_all_results: bool = True,
) -> dict:
    from prometheus_eval import PrometheusEval
    from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE
    from prometheus_eval.vllm import VLLM

    from scripts.utils.prompts import M_RUBRIC_PROMPT, get_rubric_criteria

    if dry_run:
        model = "Unbabel/M-Prometheus-3B"

    lang_name = Language.make(language).display_name()
    template = M_RUBRIC_PROMPT.format(language=lang_name)
    rubrics = SCORE_RUBRIC_TEMPLATE.format(**get_rubric_criteria(lang_name))

    model = VLLM(model=model, trust_remote_code=True, tensor_parallel_size=tensor_parallel_size)  # fmt: skip
    judge = PrometheusEval(model=model, absolute_grade_template=template)

    instructions = dataset["prompt"]
    responses = dataset["response"]

    feedbacks, scores = judge.absolute_grade(
        instructions=instructions,
        responses=responses,
        rubric=rubrics,
        params={"temperature": 0.3},  # TODO: figure out best setup
    )

    metrics = {"average_rubric_score": sum(scores) / len(scores)}
    if save_all_results:
        metrics["per_instance_rubric_scores"] = [
            {
                "instruction": inst,
                "response": resp,
                "feedback": fb,
                "score": score,
            }
            for inst, resp, fb, score in zip(instructions, responses, feedbacks, scores)
        ]

    metrics["metadata"] = {
        "language": language,
        "model": model,
    }

    return metrics


if __name__ == "__main__":
    main()
