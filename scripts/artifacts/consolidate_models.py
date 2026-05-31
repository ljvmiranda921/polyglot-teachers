"""Consolidate the Polyglot Teachers models into a repo."""

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, get_collection, snapshot_download

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logger.addHandler(_handler)

COLLECTION = "ljvmiranda921/polyglot-teachers-6904893eee7a3a73e171c837"
TARGET = "ljvmiranda921/Polyglot-SFT-Multilingual"
IGNORE = [".cache/*", ".git*", "*.lock"]


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Copy every model in a collection into branches of one repo.")
    parser.add_argument("--collection", default=COLLECTION, help="Source collection slug.")
    parser.add_argument("--target", default=TARGET, help="Target repo that will hold one branch per model.")
    parser.add_argument("--workdir", default=None, help="Where to download snapshots. Defaults to a temp dir.")
    parser.add_argument("--extra", nargs="+", default=[], metavar="MODEL_ID", help="Extra model IDs to include on top of the collection.")
    parser.add_argument("--private", action="store_true", help="Create the target repo as private.")
    parser.add_argument("--dry_run", action="store_true", help="List what would be done without creating or uploading anything.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    api = HfApi()

    collection = get_collection(args.collection)
    models = [item.item_id for item in collection.items if item.item_type == "model"]

    # Append any ad-hoc models passed on the command line, skipping duplicates.
    for model_id in args.extra:
        if model_id not in models:
            models.append(model_id)

    # The collection may already list the consolidated repo itself; never copy
    # the target into one of its own branches.
    models = [model_id for model_id in models if model_id != args.target]

    if not models:
        logger.info("No models found in collection. Nothing to do.")
        return

    mapping = [(model_id.split("/")[-1], model_id) for model_id in models]

    logger.info(f"Found {len(models)} models in '{args.collection}':")
    for branch, original in mapping:
        logger.info(f"{original} will become branch '{branch}'")

    if args.dry_run:
        logger.info(f"Dry run. Would create '{args.target}' with the branches above.")
        return

    api.create_repo(args.target, repo_type="model", private=args.private, exist_ok=True)
    logger.info(f"Ensured target repo: {args.target}")

    workdir = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.mkdtemp(prefix="polyglot_"))
    )
    workdir.mkdir(parents=True, exist_ok=True)

    for branch, original in mapping:
        snapshot_dir = workdir / branch
        try:
            logger.info(f"[{branch}] Creating branch on {args.target}")
            api.create_branch(args.target, branch=branch, exist_ok=True)

            logger.info(f"[{branch}] Downloading {original}")
            snapshot_download(
                original,
                local_dir=str(snapshot_dir),
                ignore_patterns=IGNORE,
            )

            logger.info(f"[{branch}] Uploading to {args.target}@{branch}")
            api.upload_folder(
                folder_path=str(snapshot_dir),
                repo_id=args.target,
                revision=branch,
                commit_message=f"Copy {original}",
                ignore_patterns=IGNORE,
            )
            logger.info(f"[{branch}] Done")
        except Exception:
            logger.exception(f"[{branch}] Failed")
        finally:
            shutil.rmtree(snapshot_dir, ignore_errors=True)

    logger.info(
        f"Finished. View branches at https://huggingface.co/{args.target}/tree/main"
    )


if __name__ == "__main__":
    main()
