"""Export a model from a branch of ljvmiranda921/msde-sft-dev to a new HuggingFace model repository.

Downloads all files from the specified branch, generates a model card from the
template, and uploads everything to the target repository.
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_handler)

SOURCE_REPO = "ljvmiranda921/msde-sft-dev"
TEMPLATE_PATH = Path(__file__).parent.parent / "assets" / "TEMPLATE_MODEL_CARD.md"

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "es": "Spanish",
    "id": "Indonesian",
    "ja": "Japanese",
    "tl": "Tagalog",
}


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Export a model branch to a new HuggingFace repository.")
    parser.add_argument("--branch", type=str, required=True, help="Branch name in ljvmiranda921/msde-sft-dev to export.")
    parser.add_argument("--output_repo", type=str, required=True, help="Target HuggingFace model repository (e.g. ljvmiranda921/Polyglot-Gemma3-4B-SFT-ar).")
    parser.add_argument("--language", type=str, required=True, choices=list(LANGUAGE_NAMES.keys()), help="Language code for the model card.")
    parser.add_argument("--private", action="store_true", help="Make the target repository private.")
    # fmt: on
    return parser.parse_args()


def render_model_card(language: str) -> str:
    """Render the template model card with the given language."""
    template = TEMPLATE_PATH.read_text()
    language_name = LANGUAGE_NAMES[language]
    return template.format(language=language, language_name=language_name)


def main():
    args = get_args()
    api = HfApi()

    # Download all files from the source branch
    logger.info(f"Downloading {SOURCE_REPO} branch={args.branch}")
    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = snapshot_download(
            SOURCE_REPO,
            revision=args.branch,
            local_dir=tmpdir,
        )
        local_path = Path(local_dir)
        logger.info(f"Downloaded to {local_path}")

        # Write model card
        readme_path = local_path / "README.md"
        model_card = render_model_card(args.language)
        readme_path.write_text(model_card)
        logger.info(f"Generated model card for language={args.language}")

        # Create repo and upload
        api.create_repo(args.output_repo, exist_ok=True, private=args.private)
        logger.info(f"Uploading to {args.output_repo}...")
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=args.output_repo,
            commit_message=f"Upload model from {SOURCE_REPO} branch {args.branch}",
        )
        logger.info(f"Done! Model available at https://huggingface.co/{args.output_repo}")


if __name__ == "__main__":
    main()
