"""Get the PG-Score for a given dataset."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import DownloadMode, load_dataset
from huggingface_hub import list_datasets, snapshot_download
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfxethighperformance
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

# Setup cache
CACHE_DIR = Path("data/mtep-cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_INT = CACHE_DIR / "intrinsic_metrics.jsonl"
CACHE_EXT = CACHE_DIR / "extrinsic_metrics.jsonl"

# Define metrics to track for each benchmark
METRICS_TASK_MAP = {
    "global_mmlu_lite": "acc",
    "mrewardbench_mcf": "weighted_acc",
    "mgsm_custom": "extractive_match",
}


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Get the PG-Score for a given dataset.")
    parser.add_argument("-i", "--intrinsic", type=str, default="ljvmiranda921/mtep-intrinsic-metrics", help="Huggingface Dataset containing the intrinsic metrics.")
    parser.add_argument("-e", "--extrinsic", type=str, default="details_", help="Search string for getting HuggingFace datasets with student model performance.")
    parser.add_argument("--intrinsic_kwargs", type=str, default='{"directory_path": "csd3", "local_path": "data"}', help="Extra arguments to pass when processing the intrinsic metrics.")
    parser.add_argument("--extrinsic_kwargs", type=str, default="{}", help="Extra arguments to pass when processing the extrinsic metrics.")
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
            df.to_json(CACHE_INT, orient="records", lines=True)
            logging.info(f"Saved intrinsic metrics to {CACHE_INT}")

    else:
        logging.info(f"Using cache from {CACHE_INT}. Ignoring other kwargs...")
        df = pd.read_json(CACHE_INT, lines=True)

    return df


def get_extrinsic_metrics(
    repo_search_str: str,
    *,
    use_cache: bool = False,
    hf_org: str = "ljvmiranda921",
    force_redownload: bool = False,
    cache_results: bool = False,
):
    if not use_cache:
        hf_dataset_ids = [dataset.id for dataset in list_datasets(search=repo_search_str, author=hf_org)]  # fmt: skip
        logging.info(f"Found {len(hf_dataset_ids)} datasets using search string: '{repo_search_str}'")  # fmt: skip

        _dfs: list[pd.DataFrame] = []
        for hf_dataset_id in hf_dataset_ids:
            _dfs.append(
                _process_results(hf_dataset_id, force_redownload=force_redownload)
            )
        df = pd.concat(_dfs).reset_index(drop=True)
        if cache_results:
            df.to_json(CACHE_EXT, orient="records", lines=True)
            logging.info(f"Saved extrinsic metrics to {CACHE_EXT}")

    else:
        df = pd.read_json(CACHE_EXT, lines=True)

    return df


def _process_results(dataset_id: str, force_redownload: bool = False) -> pd.DataFrame:
    """Parse a dataset ID and output a dataframe containing the relevant fields

    Based from: https://huggingface.co/docs/lighteval/en/saving-and-reading-results
    """
    logging.info(f"Parsing results from dataset {dataset_id}")
    model_info = _parse_model_info(dataset_id)
    logging.info(model_info)

    ds = load_dataset(
        dataset_id,
        "results",
        trust_remote_code=True,
        download_mode=(
            DownloadMode.FORCE_REDOWNLOAD
            if force_redownload
            else DownloadMode.REUSE_CACHE_IF_EXISTS
        ),
    )

    metrics = []
    _tasks_checked = []
    eval_runs = sorted(ds.keys())  # from oldest to newest
    for eval_run in tqdm(eval_runs, desc="Processing eval runs"):
        df = ds[eval_run].to_pandas()
        for task, result in json.loads(df.results.iloc[0]).items():
            if task != "all":
                if task in _tasks_checked:
                    # Skip, so we don't have duplicate results
                    pass
                else:
                    _tasks_checked.append(task)
                    task_dict = _parse_eval_str(task)
                    task_dict.update({"result": result.get(METRICS_TASK_MAP.get(task_dict.get("task")))})  # fmt: skip
                    task_dict.update({"result_stderr": result.get(METRICS_TASK_MAP.get(task_dict.get("task")) + "_stderr")})  # fmt: skip
                    task_dict.update({"raw_result": result})
                    metrics.append(task_dict)

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.assign(**model_info)
    return metrics_df


def _parse_eval_str(task_str: str) -> dict[str, str | int]:
    task_lang, n_shots = task_str.split("|")
    task, lang = task_lang.split(":")
    return {"task": task, "eval_lang": lang, "n_shots": int(n_shots)}


def _parse_model_info(dataset_id: str) -> dict[str, str | bool]:
    # dataset_id: 'ljvmiranda921/details_msde-allenai_Olmo-3-1025-7B-lora-4bit-msde-S1-es_Llama-3_1-8B-Instruct' -> 'allenai_Olmo-3-1025-7B-lora-4bit-msde-S1-es_Llama-3_1-8B-Instruct'
    prefix = "details_msde-"
    relevant_part = dataset_id.split(prefix, 1)[1] if prefix in dataset_id else dataset_id  # fmt: skip
    # from: 'allenai_Olmo-3-1025-7B-lora-4bit-msde-S1-es_Llama-3_1-8B-Instruct'
    # parts: ['allenai_Olmo-3-1025-7B-lora-4bit', 'es_Llama-3_1-8B-Instruct']
    # model_info_raw: 'allenai_Olmo-3-1025-7B-lora-4bit'
    # lang_and_teacher: 'es_Llama-3_1-8B-Instruct'
    model_info_raw, lang_and_teacher = relevant_part.split("-msde-S1-")
    # Extract language and teacher model
    lang_teacher_parts = lang_and_teacher.split("_", 1)
    language = lang_teacher_parts[0]
    teacher_model_raw = lang_teacher_parts[1] if len(lang_teacher_parts) > 1 else ""
    teacher_model = teacher_model_raw.replace("_", ".")
    # Check for lora/qlora in the model info
    is_lora_model = "lora" in model_info_raw.lower()
    is_qlora_model = is_lora_model and "4bit" in model_info_raw.lower() or "8bit" in model_info_raw.lower()  # fmt: skip
    # Remove lora/qlora suffix to get clean base model name
    base_model_raw = model_info_raw
    if "-lora" in base_model_raw.lower():
        idx = base_model_raw.lower().find("-lora")
        base_model_raw = base_model_raw[:idx]
    # Replace only the first underscore with slash (org/model)
    base_model = (
        base_model_raw.replace("_", "/", 1) if "_" in base_model_raw else base_model_raw
    )

    return {
        "base_model": base_model,
        "teacher_model": teacher_model,
        "lora": is_lora_model,
        "qlora": is_qlora_model,
        "target_lang": language,
    }


def get_base_model_results(dataset_id: str):
    pass


def get_ref_model_results(dataset_id: str):
    pass


if __name__ == "__main__":
    main()
