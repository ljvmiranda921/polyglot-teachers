"""Delete all datasets matching the pattern ljvmiranda921/details_msde*.

WARNING: This script is DESTRUCTIVE and IRREVERSIBLE. It will permanently
delete datasets from HuggingFace Hub. Use --dry_run first to see what
would be deleted.
"""

import argparse
import logging
import sys
import time

from huggingface_hub import HfApi

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

PATTERN = "details_msde"
AUTHOR = "ljvmiranda921"


def get_args():
    parser = argparse.ArgumentParser(
        description="Delete all datasets matching ljvmiranda921/details_msde*."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only list datasets that would be deleted, don't actually delete.",
    )
    return parser.parse_args()


def confirm(prompt: str) -> bool:
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response == "y"


def main():
    args = get_args()
    api = HfApi()

    repos = list(api.list_datasets(author=AUTHOR, search=PATTERN))
    matching = [r for r in repos if PATTERN in r.id]

    if not matching:
        logger.info("No matching datasets found. Nothing to do.")
        return

    logger.info(f"Found {len(matching)} datasets matching '{AUTHOR}/{PATTERN}*'")

    if args.dry_run:
        for r in matching:
            print(f"  {r.id}")
        logger.info(f"Dry run complete. {len(matching)} datasets would be deleted.")
        return

    # Double confirmation
    print(f"\n{'='*60}")
    print(f"  WARNING: You are about to DELETE {len(matching)} datasets.")
    print(f"  This action is IRREVERSIBLE.")
    print(f"{'='*60}\n")

    if not confirm("Are you sure you want to delete these datasets?"):
        logger.info("Aborted.")
        return

    if not confirm("Are you REALLY sure? This cannot be undone."):
        logger.info("Aborted.")
        return

    deleted = 0
    for r in matching:
        try:
            api.delete_repo(r.id, repo_type="dataset")
            logger.info(f"  Deleted {r.id}")
            deleted += 1
        except Exception:
            logger.exception(f"  Failed to delete {r.id}")
        time.sleep(0.5)

    logger.info(f"Done. Deleted {deleted}/{len(matching)} datasets.")


if __name__ == "__main__":
    main()
