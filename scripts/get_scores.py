"""Get the PG-Score for a given dataset."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from datasets import DownloadMode, load_dataset
from huggingface_hub import list_datasets, snapshot_download
from sklearn.preprocessing import StandardScaler
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
    parser.add_argument("-e", "--extrinsic", type=str, default="msde-allenai_Olmo-3-1025-7B", help="Search string for getting HuggingFace datasets with student model performance.")
    parser.add_argument("--output_file", type=str, default="pg_scores.jsonl", help="Output file to save the PG-Scores.")
    parser.add_argument("--intrinsic_kwargs", type=str, default='{"directory_path": "csd3", "local_path": "data"}', help="Extra arguments to pass when processing the intrinsic metrics.")
    parser.add_argument("--extrinsic_kwargs", type=str, default="{}", help="Extra arguments to pass when processing the extrinsic metrics.")
    parser.add_argument("--ref_model_results", type=str, default="ljvmiranda921/details_allenai__Olmo-3-7B-Instruct-SFT_private", help="Huggingface Dataset containing the reference model results.")
    parser.add_argument("--base_model_results", type=str, default="ljvmiranda921/details_allenai__Olmo-3-1025-7B_private", help="Huggingface Dataset containing the base model results.")
    parser.add_argument("--show_per_language", action="store_true", help="Whether to show per-language PG-Scores.")
    parser.add_argument("--add_metadata", type=str, default="{}", help="Additional metadata to add to the output JSONL file. Must be valid JSON. Will be added as a field for each row.")
    parser.add_argument("--append", action="store_true", help="Whether to append to the output file if it exists.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    # Get intrinsic metrics
    df_int = get_intrinsic_metrics(repo_id=args.intrinsic, **json.loads(args.intrinsic_kwargs))  # fmt: skip
    df_int_computed = compute_intrinsic_zscore(df_int)

    # Get extrinsic metrics
    df_ext = get_extrinsic_metrics(repo_search_str=args.extrinsic, **json.loads(args.extrinsic_kwargs))  # fmt: skip
    df_ext_computed = compute_extrinsic_pgr(
        df_ext,
        df_base=_process_results(
            args.base_model_results,
            model_info={
                "model_name": "allenai/OLMo-3-1025-7B",
                "lora": False,
                "qlora": False,
            },
        ),
        df_ref=_process_results(
            args.ref_model_results,
            model_info={
                "model_name": "allenai/OLMo-3-7B-Instruct-SFT",
                "lora": False,
                "qlora": False,
            },
        ),
    )

    # Merge intrinsic and extrinsic metrics
    # Extract model name without org from intrinsic data
    df_int_for_merge = df_int_computed.copy()
    df_int_for_merge["model_short"] = df_int_for_merge["model"].str.split("/").str[-1]

    df_merged = df_ext_computed.merge(
        df_int_for_merge[["model_short", "language", "z_score"]],
        left_on=["teacher_model", "target_lang"],
        right_on=["model_short", "language"],
        how="left",
    ).drop(columns=["model_short", "language"])
    # Compute PG-Score as average of PGR and z-score
    df_merged["pg_score"] = (df_merged["pgr"] + df_merged["z_score"]) / 2

    # Add additional metadata if provided
    additional_metadata = json.loads(args.add_metadata)
    for key, value in additional_metadata.items():
        df_merged[key] = value

    print("\n====== PGR (grouped-by teacher model across languages) ======")
    group_by_cols = (
        ["teacher_model", "target_lang"]
        if args.show_per_language
        else ["teacher_model"]
    )
    print(
        df_merged.groupby(group_by_cols)
        .agg({"pg_score": "mean", "result": "mean", "pgr": "mean"})
        .reset_index()
        .to_markdown(index=False)
    )
    if args.append and (CACHE_DIR / args.output_file).exists():
        df_existing = pd.read_json(CACHE_DIR / args.output_file, lines=True)
        df_merged = pd.concat([df_existing, df_merged]).reset_index(drop=True)
    df_merged.to_json(CACHE_DIR / args.output_file, orient="records", lines=True)
    logging.info(f"Saved PG-Scores to {CACHE_DIR / args.output_file}")


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
        logging.info(f"Using cache from {CACHE_EXT}. Ignoring other kwargs...")
        df = pd.read_json(CACHE_EXT, lines=True)

    return df


def _process_results(
    dataset_id: str,
    force_redownload: bool = False,
    model_info: Optional[dict] = {},
) -> pd.DataFrame:
    """Parse a dataset ID and output a dataframe containing the relevant fields

    Based from: https://huggingface.co/docs/lighteval/en/saving-and-reading-results
    """
    logging.info(f"Parsing results from dataset {dataset_id}")
    _model_info = model_info if model_info else _parse_model_info(dataset_id)
    logging.info(_model_info)

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
    metrics_df = metrics_df.assign(**_model_info)
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
    model_info_raw, lang_and_teacher = relevant_part.split("-msde-T1-")
    # Extract language and teacher model
    lang_teacher_parts = lang_and_teacher.split("_", 1)
    language = lang_teacher_parts[0]
    teacher_model_raw = lang_teacher_parts[1] if len(lang_teacher_parts) > 1 else ""
    teacher_model = teacher_model_raw.replace("_", ".")

    # Filtering
    # If .generate, .translate, and .respond suffixes exist, remove them
    to_remove_method = [".generate", ".translate", ".respond"]
    # If .sz{number} suffix exists, remove it
    to_remove_number = [f".sz{num}k" for num in [1, 5, 10, 25, 50]]
    # if .translate.ablation and other exists, remove it
    to_remove_method += [
        ".translate.ablation",
        ".nllb.translate.both",
        ".nllb.ttr",
    ]

    for suffix in to_remove_method + to_remove_number:
        if teacher_model.endswith(suffix):
            teacher_model = teacher_model[: -len(suffix)]
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


def compute_extrinsic_pgr(
    df_ext: pd.DataFrame, df_base: pd.DataFrame, df_ref: pd.DataFrame
) -> pd.DataFrame:
    """Compute PGR scores for extrinsic metrics."""

    def _cagg(group):
        matching = group[group["target_lang"] == group["eval_lang"]]
        data = matching if len(matching) > 0 else group
        return data[["result", "result_stderr"]].mean()

    df_ext_avg = df_ext.groupby(["teacher_model", "target_lang"]).apply(_cagg).reset_index()  # fmt: skip
    df_base_avg = df_base.groupby(["eval_lang"]).agg({"result": "mean"}).reset_index()
    df_ref_avg = df_ref.groupby(["eval_lang"]).agg({"result": "mean"}).reset_index()

    df_merged = df_ext_avg.merge(
        df_base_avg.rename(columns={"result": "base_perf"}),
        left_on="target_lang",
        right_on="eval_lang",
        how="left",
    ).drop(columns=["eval_lang"])
    df_merged = df_merged.merge(
        df_ref_avg.rename(columns={"result": "ref_perf"}),
        left_on="target_lang",
        right_on="eval_lang",
        how="left",
    ).drop(columns=["eval_lang"])
    df_merged["pgr"] = df_merged.apply(compute_pgr, axis=1)
    return df_merged.drop(columns=["base_perf", "ref_perf"])


def compute_intrinsic_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Compute intrinsic quality scores using z-score normalization."""
    data = np.column_stack(
        [
            df["prompts_distinct_ri"],
            df["responses_distinct_ri"],
            df["rubric_score"],
            -np.log1p(df["perplexity"]),
        ]
    )
    scaler = StandardScaler()
    normalized = scaler.fit_transform(data)
    df = df.copy()
    df["z_score"] = normalized.mean(axis=1)
    return df


def compute_pgr(
    row,
    result_col: str = "result",
    base_col: str = "base_perf",
    ref_col: str = "ref_perf",
):
    """Compute the PGR (Performance Gain Recovered) for a given row.
    Ref: https://arxiv.org/abs/2412.03679
    """
    return (row[result_col] - row[base_col]) / (row[ref_col] - row[base_col])


if __name__ == "__main__":
    main()
