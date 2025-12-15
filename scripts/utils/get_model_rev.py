"""Get model revisions from Hugging Face Hub."""

import argparse
import sys
from huggingface_hub import HfApi


def get_model_revisions(
    model_id: str, search_str: str = None, include_main: bool = False
) -> list[str]:
    """Get all revisions for a model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b")
        search_str: Optional search string to filter revisions
        include_main: Whether to include the "main" branch in results

    Returns:
        List of revision strings (branch names or tags)
    """
    api = HfApi()

    try:
        # Get model info which includes all refs (branches and tags)
        refs = api.list_repo_refs(model_id, repo_type="model")

        # Collect all revision names
        revisions = []

        # Add branches
        for branch in refs.branches:
            revisions.append(branch.name)

        # Add tags
        for tag in refs.tags:
            revisions.append(tag.name)

        # Filter out "main" if requested
        if not include_main:
            revisions = [r for r in revisions if r != "main"]

        # Filter by search string if provided
        if search_str:
            revisions = [r for r in revisions if search_str in r]

        return sorted(revisions)

    except Exception as e:
        print(f"Error fetching revisions for {model_id}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Get model revisions from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Get all revisions (excluding main)
    python scripts/utils/get_model_rev.py --hf-model-id "meta-llama/Llama-2-7b"

    # Get revisions containing "step"
    python scripts/utils/get_model_rev.py --hf-model-id "user/model" --search-str "step"

    # Include main branch in results
    python scripts/utils/get_model_rev.py --hf-model-id "user/model" --include-main

    # Use in shell script for evaluation
    for rev in $(python scripts/utils/get_model_rev.py --hf-model-id "user/model" --search-str "checkpoint"); do
        echo "Evaluating revision: $rev"
        lighteval vllm "model_name=user/model,revision=$rev" ...
    done
        """,
    )

    parser.add_argument(
        "--hf-model-id",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., 'meta-llama/Llama-2-7b')",
    )

    parser.add_argument(
        "--search-str",
        type=str,
        default=None,
        help="Filter revisions by this search string (case-sensitive substring match)",
    )

    parser.add_argument(
        "--include-main",
        action="store_true",
        help="Include the 'main' branch in results (excluded by default)",
    )

    parser.add_argument(
        "--delimiter",
        type=str,
        default="\n",
        help="Delimiter between revision names (default: newline)",
    )

    args = parser.parse_args()

    # Get revisions
    revisions = get_model_revisions(
        model_id=args.hf_model_id,
        search_str=args.search_str,
        include_main=args.include_main,
    )

    # Print results
    if not revisions:
        print(
            f"No revisions found for model '{args.hf_model_id}' with search string '{args.search_str}'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(args.delimiter.join(revisions))


if __name__ == "__main__":
    main()
