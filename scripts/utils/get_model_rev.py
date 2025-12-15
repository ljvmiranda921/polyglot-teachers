import argparse
import sys
from huggingface_hub import HfApi


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Get model revisions from Hugging Face Hub")
    parser.add_argument("--hf_model_id", type=str, required=True, help="HuggingFace model ID (e.g., 'meta-llama/Llama-2-7b')")
    parser.add_argument("--search_str", type=str, default=None, help="Filter revisions by this search string (case-sensitive substring match)")
    parser.add_argument("--include_main", action="store_true", help="Include the 'main' branch in results (excluded by default)")
    parser.add_argument("--delimiter", type=str, default="\n", help="Delimiter between revision names (default: newline)")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    revisions = get_model_revisions(
        model_id=args.hf_model_id,
        search_str=args.search_str,
        include_main=args.include_main,
    )

    if not revisions:
        print(
            f"No revisions found for model '{args.hf_model_id}' with search string '{args.search_str}'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(args.delimiter.join(revisions))


def get_model_revisions(
    model_id: str, search_str: str = None, include_main: bool = False
) -> list[str]:
    """Get all revisions for a model from HuggingFace Hub."""
    api = HfApi()

    try:
        refs = api.list_repo_refs(model_id, repo_type="model")
        revisions = []
        for branch in refs.branches:
            revisions.append(branch.name)
        for tag in refs.tags:
            revisions.append(tag.name)
        if not include_main:
            revisions = [r for r in revisions if r != "main"]
        if search_str:
            revisions = [r for r in revisions if search_str in r]
        return sorted(revisions)
    except Exception as e:
        print(f"Error fetching revisions for {model_id}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
