"""Create the final polyglot-teachers-sft dataset by combining all languages,
filtered to only include the gemma-3-27b model.

Uses streaming to avoid loading entire datasets into memory at once.
Each language is processed one at a time and written as a parquet shard,
then all shards are uploaded together.
"""

import argparse
import logging
import sys
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

LANGUAGES = ["ar", "de", "id", "ja", "es", "cs"]
SOURCE_DATASET = "ljvmiranda921/msde-S1-{language}"
TARGET_MODEL = "google/gemma-3-27b-it"
COLUMNS_TO_KEEP = ["id", "source", "language", "strategy", "source_id", "synth_prompt", "model", "prompt", "response", "messages"]


def get_args():
    parser = argparse.ArgumentParser(description="Create the final polyglot-teachers-sft dataset.")
    parser.add_argument("--output_dataset", type=str, default="ljvmiranda921/polyglot-teachers-sft", help="HuggingFace dataset path to save to.")
    parser.add_argument("--cache_dir", type=str, default="data/final_dataset_cache", help="Local cache directory for intermediate parquet shards.")
    parser.add_argument("--languages", nargs="+", type=str, default=LANGUAGES, help="Languages to include.")
    return parser.parse_args()


def process_language(language: str, cache_dir: Path) -> Dataset:
    """Stream a single language dataset, filter to target model, and save a parquet shard."""
    dataset_name = SOURCE_DATASET.format(language=language)
    logging.info(f"Streaming {dataset_name}, filtering for model={TARGET_MODEL}")

    ds = load_dataset(dataset_name, split="train", streaming=True)
    filtered = ds.filter(lambda row: row["model"] == TARGET_MODEL)

    # Materialize to a local parquet file one batch at a time
    shard_path = cache_dir / f"{language}.parquet"
    if shard_path.exists():
        logging.info(f"  Shard already exists at {shard_path}, loading from cache")
        return Dataset.from_parquet(str(shard_path))

    rows = []
    for row in filtered:
        rows.append({col: row[col] for col in COLUMNS_TO_KEEP if col in row})

    shard_ds = Dataset.from_list(rows)
    shard_ds.to_parquet(str(shard_path))
    logging.info(f"  {language}: {len(shard_ds)} rows saved to {shard_path}")
    return shard_ds


def main():
    args = get_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    shards = []
    for lang in args.languages:
        shard = process_language(lang, cache_dir)
        shards.append(shard)

    final_ds = concatenate_datasets(shards)
    logging.info(f"Final dataset: {len(final_ds)} rows across {len(args.languages)} languages")

    # Show language distribution
    from collections import Counter

    lang_counts = Counter(final_ds["language"])
    for lang, count in sorted(lang_counts.items()):
        logging.info(f"  {lang}: {count}")

    # Push to hub
    logging.info(f"Pushing to {args.output_dataset}...")
    try:
        final_ds.push_to_hub(args.output_dataset, private=True)
        logging.info("Done!")
    except Exception:
        logging.exception("Failed to push to hub. Dataset saved locally in cache dir.")


if __name__ == "__main__":
    main()
