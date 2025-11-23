import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import pandas as pd
from datasets import Dataset, load_dataset
from langcodes import Language
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)


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
    parser.add_argument("--metric_params", type=str, default=None, help="Additional parameters for the metric in JSON format. You need to specify this as 'metric_fn::{\"param1\": value1, \"param2\": value2}|metric_fn2::...'.")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Will perform a dry run without saving any files and using a small amount of samples (1000).")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to use from the dataset.")
    parser.add_argument("--input_dataset_filter", type=str, default=None, help="JSON string representing a filter to apply to the input dataset before finetuning. The keys should be the field names and the values should be the values to filter by. This is an AND operation.")
    parser.add_argument("--apply_subsampling", action="store_true", default=False, help="Whether to apply subsampling to the dataset before computing metrics. This is to ensure that the number of samples per strategy is roughly the same.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Whether to overwrite the output file if it already exists.")
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
        dataset, subsampling_results = subsample_per_strategy(dataset)
    if args.dry_run:
        logging.info("Dry run: using a small subset of the dataset (1000 samples).")
        dataset = dataset.shuffle().select(range(1000))
    if args.limit is not None:
        logging.info(f"Limiting dataset to {args.limit} samples.")
        dataset = dataset.select(range(args.limit))

    metadata = {
        "input_dataset": args.input_dataset,
        "num_samples": len(dataset),
        "input_dataset_filter": args.input_dataset_filter,
        "subsampling_results": subsampling_results if args.apply_subsampling else None,
    }

    save_scores(
        output_path,
        "metadata",
        metadata,
        append=False,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    for metric in metrics_to_compute:
        metric_fn = get_intrinsic_metrics()[metric]
        logging.info(f"Computing metric: {metric}")
        params = metric_params.get(metric, {})
        score = metric_fn(dataset, args.dry_run, **params)
        logging.info(f">>> {score}")
        save_scores(
            output_path,
            metric,
            score,
            append=True,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        # Sleep for 2 minutes to let vLLM release GPU memory
        time.sleep(120)


def save_scores(
    output_path: Path,
    metric_name: str,
    metric_data: dict,
    append: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    """Save or append metric scores to a JSON file."""
    if dry_run:
        return

    if output_path.exists():
        # If output_path exists, always append to avoid data loss
        append = True

    if append:
        with open(output_path, "r", encoding="utf-8") as f:
            metric_scores = json.load(f)

        # Check if metric already exists and respect overwrite flag
        if metric_name in metric_scores and not overwrite:
            logging.info(
                f"Metric '{metric_name}' already exists in file. Skipping (overwrite=False)."
            )
            return

        metric_scores[metric_name] = metric_data
    else:
        metric_scores = {metric_name: metric_data}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metric_scores, f, indent=4)


def parse_metric_params(param_str: str) -> dict[str, dict]:
    """Parse metric parameters from a string."""
    metric_params = {}
    for item in param_str.split("|"):
        metric_name, params_json = item.split("::", 1)
        metric_params[metric_name] = json.loads(params_json)
    return metric_params


def subsample_per_strategy(
    dataset: Dataset, total_num_samples: int = 10_000, random_state: int = 42
) -> tuple[Dataset, dict[str, int]]:
    """Subsample the dataset to have roughly the same number of samples per strategy.

    Ensures exactly total_num_samples are returned by distributing samples equally
    across strategies, then redistributing any remaining samples to strategies with
    available capacity.
    """
    df = dataset.to_pandas()
    strategies = df["strategy"].unique()
    num_strategies = len(strategies)

    # Equal distribution across strategies
    base_samples_per_strategy = total_num_samples // num_strategies
    subsampled_dfs = []
    samples_taken = {}

    for strategy in strategies:
        strategy_df = df[df["strategy"] == strategy]
        num_samples = min(base_samples_per_strategy, len(strategy_df))
        sampled_df = strategy_df.sample(n=num_samples, random_state=random_state)
        subsampled_dfs.append(sampled_df)
        samples_taken[strategy] = num_samples

    # Redistribute remaining samples to strategies with capacity
    current_total = sum(samples_taken.values())
    remaining = total_num_samples - current_total

    if remaining > 0:
        # Create a pool of strategies that can provide more samples
        available_strategies = [
            (strategy, len(df[df["strategy"] == strategy]) - samples_taken[strategy])
            for strategy in strategies
            if len(df[df["strategy"] == strategy]) > samples_taken[strategy]
        ]
        # Sort by available capacity (descending)
        available_strategies.sort(key=lambda x: x[1], reverse=True)

        # Distribute remaining samples
        for strategy, available in available_strategies:
            if remaining == 0:
                break
            additional = min(remaining, available)
            if additional > 0:
                strategy_df = df[df["strategy"] == strategy]
                # Get samples not already taken
                already_sampled = subsampled_dfs[list(strategies).index(strategy)]
                remaining_pool = strategy_df.drop(already_sampled.index)
                extra_samples = remaining_pool.sample(
                    n=additional, random_state=random_state + 1
                )
                subsampled_dfs[list(strategies).index(strategy)] = pd.concat(
                    [already_sampled, extra_samples]
                )
                remaining -= additional

    subsampled_df = pd.concat(subsampled_dfs).reset_index(drop=True)
    subsampling_results = subsampled_df.strategy.value_counts().to_dict()
    return Dataset.from_pandas(subsampled_df), subsampling_results


def _compute_distinct_ri(
    dataset: "Dataset",
    dry_run: bool,
    *,
    embedding_model: str = "nvidia/llama-embed-nemotron-8b",
    tensor_parallel_size: int = 2,
) -> dict[str, float]:
    """Compute the distinctiveness of the instructions and responses in the dataset."""
    from vllm import LLM
    from vllm.outputs import EmbeddingRequestOutput
    from sentence_transformers.util import cos_sim

    if "prompt" not in dataset.column_names or "response" not in dataset.column_names:
        raise ValueError("Dataset must contain 'prompt' and 'response' fields!")
    prompts = dataset["prompt"]
    responses = dataset["response"]

    if dry_run:
        embedding_model = "google/embeddinggemma-300m"

    model = LLM(
        model=embedding_model,
        task="embed",
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
    )

    metrics = {}
    for k, texts in {"prompts": prompts, "responses": responses}.items():
        outputs: list[EmbeddingRequestOutput] = model.embed(texts)
        embeddings = [output.outputs.embedding for output in outputs]

        # Compute cosine similarity
        similarity_matrix = cos_sim(embeddings, embeddings)
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
    model_name: str = "Unbabel/M-Prometheus-14B",
    provider: str = "transformers",
    tensor_parallel_size: int = 1,
    save_all_results: bool = True,
    openai_api_key: str = "EMPTY",
    max_concurrent_requests: int = 4,
) -> dict:
    from typing import Literal

    import outlines
    from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE
    from pydantic import BaseModel

    from scripts.utils.prompts import M_RUBRIC_PROMPT, get_rubric_criteria

    if dry_run:
        model_name = "Unbabel/M-Prometheus-3B"

    # Prepare inputs
    lang_name = Language.make(language).display_name()
    rubrics = SCORE_RUBRIC_TEMPLATE.format(**get_rubric_criteria(lang_name))
    template = M_RUBRIC_PROMPT.format(language=lang_name)
    instructions = dataset["prompt"]
    responses = dataset["response"]
    inputs = [
        template.format(instruction=inst, response=resp, rubric=rubrics)
        for inst, resp in zip(instructions, responses)
    ]

    class Feedback(BaseModel):
        score: Literal[1, 2, 3, 4, 5]
        feedback: str

    if provider == "llamacpp":
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download

        # Extract filename from model name
        # (e.g., "M-Prometheus-3B-Q4_K_M-GGUF" -> "m-prometheus-3b-q4_k_m.gguf")
        model_basename = model_name.split("/")[-1]
        if model_basename.upper().endswith("-GGUF"):
            model_basename = model_basename[:-5]  # Remove "-GGUF"
        model_filename = model_basename.lower() + ".gguf"

        logging.info(f"Downloading GGUF model {model_filename} from {model_name}...")
        model_path = hf_hub_download(
            repo_id=model_name,
            filename=model_filename,
            cache_dir="./models/",
        )
        logging.info(f"Model downloaded to {model_path}")

        model = outlines.from_llamacpp(Llama(str(model_path), n_ctx=8192))
        generator = outlines.Generator(model, output_type=Feedback)
        results = []
        for input in tqdm(inputs):
            raw_output = generator(input, max_tokens=2048)
            try:
                results.append(Feedback.model_validate_json(raw_output))
            except Exception as e:
                logging.error(f"Validation error: {raw_output} | Error: {e}")
                results.append(Feedback(score=1, feedback="Invalid output"))
    elif provider == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = outlines.from_transformers(
            AutoModelForCausalLM.from_pretrained(model_name, device_map="auto"),
            AutoTokenizer.from_pretrained(model_name),
        )
        generator = outlines.Generator(model, output_type=Feedback)
        # Based on: https://github.com/ljvmiranda921/prometheus-eval/blob/dbbfb22a705af8c17dbf9f3217d2616935e8d948/libs/prometheus-eval/prometheus_eval/utils.py#L20-L24
        raw_outputs = generator.batch(inputs, max_new_tokens=2048)
        results = []
        for output in raw_outputs:
            try:
                results.append(Feedback.model_validate_json(output))
            except Exception as e:
                logging.error(f"Validation error: {output} | Error: {e}")
                results.append(Feedback(score=1, feedback="Invalid output"))
    elif provider == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
        sampling_params = SamplingParams(
            max_tokens=4096,
            guided_decoding=GuidedDecodingParams(json=Feedback.model_json_schema()),
        )
        raw_outputs = llm.generate(inputs, sampling_params=sampling_params)
        results = []
        for output in raw_outputs:
            try:
                results.append(Feedback.model_validate_json(output.outputs[0].text))
            except Exception as e:
                logging.error(f"Validation error: {output} | Error: {e}")
                results.append(Feedback(score=1, feedback="Invalid output"))

    elif provider == "openai_server":
        import asyncio
        from openai import AsyncOpenAI

        async def process_with_openai(inputs, base_url, api_key, max_concurrent):
            """Process inputs asynchronously using OpenAI-compatible server."""
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_single(input_text, index):
                async with semaphore:
                    try:
                        response = await client.beta.chat.completions.parse(
                            model=base_url,
                            messages=[
                                {
                                    "role": "user",
                                    "content": input_text,
                                }
                            ],
                            response_format=Feedback,
                            max_tokens=2048,
                        )
                        parsed = response.choices[0].message.parsed
                        return index, parsed
                    except Exception as e:
                        logging.error(f"Error processing input {index}: {e}")
                        return index, Feedback(score=1, feedback=f"Error: {str(e)}")

            # Create tasks for all inputs with their indices
            tasks = [process_single(inp, i) for i, inp in enumerate(inputs)]

            # Process with progress bar
            results_with_indices = []
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Processing with OpenAI",
                miniters=100,
            ):
                result = await coro
                results_with_indices.append(result)

            # Sort by original index to maintain order
            results_with_indices.sort(key=lambda x: x[0])
            return [result for _, result in results_with_indices]

        logging.info(
            f"Using OpenAI-compatible server at {model_name} with {max_concurrent_requests} concurrent requests"
        )
        results = asyncio.run(
            process_with_openai(
                inputs, model_name, openai_api_key, max_concurrent_requests
            )
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider}. Must be 'transformers', 'llamacpp', 'vllm', or 'openai_server'"
        )

    feedbacks = [res.feedback for res in results]
    scores = [res.score for res in results]

    not_none_scores = [s for s in scores if s is not None]
    metrics = {"average_rubric_score": sum(not_none_scores) / len(not_none_scores)}
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
        "model": model_name,
        "provider": provider,
    }

    return metrics


if __name__ == "__main__":
    main()
