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

CACHE_DIR = Path("data/mtep-cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Get the PG-Score for a given dataset.")
    parser.add_argument("-i", "--intrinsic", type=str, default="ljvmiranda921/mtep-intrinsic-metrics", help="Huggingface Dataset containing the intrinsic metrics.")
    parser.add_argument("-e", "--extrinsic", type=str, default="details_", help="Search string for getting HuggingFace datasets with student model performance.")
    parser.add_argument("--intrinsic_kwargs", type=str, default='{"directory_path": "csd3", "local_path": "data"}', help="Extra arguments to pass when processing the intrinsic metrics.")
    parser.add_argument("--use_cache", action="store_true", default=False, help="If set, will just use the saved results in cache (data/fcache).")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    if args.intrinsic_kwargs:
        int_kwargs = json.loads(args.intrinsic_kwargs)
    intrinsic_metrics: pd.DataFrame = get_intrinsic_metrics(repo_id=args.intrinsic, **int_kwargs)  # fmt: skip


def get_intrinsic_metrics(
    repo_id: str,
    *,
    directory_path: str,
    local_path: str,
    cache_results: bool = False,
) -> pd.DataFrame:
    """Downloads all JSON files containing the metrics and returns a collated DataFrame"""
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
        cache_path = CACHE_DIR / "intrinsic_metrics.jsonl"
        df.to_json(CACHE_DIR / "intrinsic_metrics.jsonl", orient="records", line=True)
        logging.info(f"Saved intrinsic metrics to {cache_path}")


if __name__ == "__main__":
    main()
