import argparse
import logging
import sys
from pathlib import Path
import json

from datasets import load_dataset, Dataset


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
    }


def get_args():
    # fmt: off
    description = "Get intrinsic metrics for a given dataset."
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dataset", type=str, required=True, help="HuggingFace dataset to compute intrinsic metrics on.")
    parser.add_argument("--output_path", type=str, required=False, help="Path to save the intrinsic metrics JSON file. If not set, will save to ./metrics/{dataset_name}_{metric}_scores.json.")
    parser.add_argument("--metrics", type=str, nargs="+", choices=["all"] + list(get_intrinsic_metrics().keys()), help="Intrinsic metric to compute.")
    parser.add_argument("--metric_params", type=str, default=None, help="Additional parameters for the metric in JSON format.")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Will perform a dry run without saving any files and using a small amount of samples (1000).")
    parser.add_argument("--input_dataset_filter", type=str, default=None, help="JSON string representing a filter to apply to the input dataset before finetuning. The keys should be the field names and the values should be the values to filter by. This is an AND operation.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    output_path = (
        Path("metrics")
        / f"{args.input_dataset.replace('/', '___')}_intrinsic_metrics.json"
        if not args.output_path
        else Path(args.output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_to_compute = list(get_intrinsic_metrics().keys()) if "all" in args.metrics else args.metrics  # fmt: skip
    logging.info(f"Computing intrinsic metrics: {metrics_to_compute} for dataset: {args.input_dataset}")  # fmt: skip

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
    import torch
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


def _compute_perplexity(dataset, *, base_model: str) -> dict[str, float]:
    pass


if __name__ == "__main__":
    main()
