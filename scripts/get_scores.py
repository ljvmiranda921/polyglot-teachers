"""Get the PG-Score for a given dataset."""

import argparse
import logging
import sys
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

# Setup cache
CACHE_DIR = Path("data/mtep-cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_INT = CACHE_DIR / "intrinsic_metrics.jsonl"
CACHE_EXT = CACHE_DIR / "extrinsic_metrics.jsonl"


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Get the PG-Score for a given dataset.")
    parser.add_argument("-i", "--intrinsic", type=str, default="ljvmiranda921/mtep-intrinsic-metrics", help="Huggingface Dataset containing the intrinsic metrics.")
    parser.add_argument("-e", "--extrinsic", type=str, default="details_", help="Search string for getting HuggingFace datasets with student model performance.")
    parser.add_argument("--intrinsic_kwargs", type=str, default='{"directory_path": "csd3", "local_path": "data"}', help="Extra arguments to pass when processing the intrinsic metrics.")
    parser.add_argument("--extrinsic_kwargs", type=str, default='{"directory_path": "csd3", "local_path": "data"}', help="Extra arguments to pass when processing the extrinsic metrics.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    df_int = get_intrinsic_metrics(
        repo_id=args.intrinsic,
        **json.loads(args.intrinsic_kwargs),
    )
    df_ext = get_extrinsic_metrics(
        repo_search_str=args.extrinsic,
        **json.loads(args.extrinsic_kwargs),
    )

    # ext_metrics = (
    #     pd.read_json(CACHE_EXT, lines=True)
    #     if args.use_cache
    #     else get_extrinsic_metrics(repo_search_str=args.extrinsic, **json.loads(args.extrinsic_kwargs))  # fmt: skip
    # )

    breakpoint()


def get_intrinsic_metrics(
    repo_id: str,
    *,
    use_cache: bool = False,
    directory_path: str = "csd3",
    local_path: str = "data",
    cache_results: bool = False,
) -> pd.DataFrame:
    """Downloads all JSON files containing the metrics and returns a collated DataFrame"""
    if not use_cache:
        _local_path = Path(local_path)
        _local_path.mkdir(parents=True, exist_ok=True)
        data_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=_local_path,
            allow_patterns=[directory_path + "/*"],
        )
        metrics_dir = Path(data_path) / directory_path

        metrics_data = []
        for file in metrics_dir.glob("*.json"):
            filename = file.stem
            parts = filename.replace("msde-S1-", "").replace("_intrinsic_metrics", "")
            language, model_raw = parts.split("_", 1)
            model = model_raw.replace("__", "/")

            with file.open("r") as f:
                data = json.load(f)

            metrics_data.append(
                # fmt: off
                {
                    "language": language,
                    "model": model,
                    "prompts_distinct_ri": data.get("distinct_ri", {}).get("prompts_distinct_ri"),
                    "responses_distinct_ri": data.get("distinct_ri", {}).get("responses_distinct_ri"),
                    "rubric_score": data.get("reward_model", {}).get("average_rubric_score"),
                    "perplexity": data.get("perplexity", {}).get("average_perplexity"),
                }
                # fmt: on
            )

        df = (
            pd.DataFrame(metrics_data)
            .sort_values(by=["language", "model"])
            .reset_index(drop=True)
        )

        if cache_results:
            df.to_json(CACHE_INT, orient="records", line=True)
            logging.info(f"Saved intrinsic metrics to {CACHE_INT}")

    else:
        logging.info(f"Using cache from {CACHE_INT}. Ignoring other kwargs...")
        df = pd.read_json(CACHE_INT, lines=True)

    return df


if __name__ == "__main__":
    main()
