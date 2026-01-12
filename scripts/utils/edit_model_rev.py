"""Edit model revisions in a Hugging Face model repository."""

import argparse
import sys
from typing import Literal

from huggingface_hub import HfApi


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Rename a revision (branch or tag) in a Hugging Face model repository.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID (e.g., 'username/model-name')")
    parser.add_argument("--old_revision", type=str, required=True, help="Current name of the revision to rename")
    parser.add_argument("--new_revision", type=str, required=True, help="New name for the revision")
    parser.add_argument("--type", type=str, choices=["branch", "tag"], default="branch", help="Type of revision to rename.")
    parser.add_argument("--no_delete_old", action="store_true", help="Keep the old revision instead of deleting it")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace authentication token (uses cached token if not provided)")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    result = rename_revision(
        model_id=args.model_id,
        old_revision=args.old_revision,
        new_revision=args.new_revision,
        revision_type=args.type,
        delete_old=not args.no_delete_old,
        token=args.token,
    )

    if result["success"]:
        print(result["message"])
        sys.exit(0)
    else:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)


def rename_revision(
    model_id: str,
    old_revision: str,
    new_revision: str,
    revision_type: Literal["branch", "tag"] = "branch",
    delete_old: bool = True,
    token: str = None,
) -> dict:
    """Rename a revision in a Hugging Face model repository."""
    api = HfApi(token=token)

    try:
        refs = api.list_repo_refs(model_id, repo_type="model")

        old_commit_sha = None
        if revision_type == "branch":
            for branch in refs.branches:
                if branch.name == old_revision:
                    old_commit_sha = branch.target_commit
                    break
        else:
            for tag in refs.tags:
                if tag.name == old_revision:
                    old_commit_sha = tag.target_commit
                    break

        if old_commit_sha is None:
            return {
                "success": False,
                "error": f"{revision_type.capitalize()} '{old_revision}' not found in repository '{model_id}'",
            }

        if revision_type == "branch":
            for branch in refs.branches:
                if branch.name == new_revision:
                    return {
                        "success": False,
                        "error": f"Branch '{new_revision}' already exists in repository '{model_id}'",
                    }
        else:
            for tag in refs.tags:
                if tag.name == new_revision:
                    return {
                        "success": False,
                        "error": f"Tag '{new_revision}' already exists in repository '{model_id}'",
                    }

        # Create the new revision pointing to the same commit
        if revision_type == "branch":
            api.create_branch(
                repo_id=model_id,
                branch=new_revision,
                revision=old_commit_sha,
                repo_type="model",
            )
        else:
            api.create_tag(
                repo_id=model_id,
                tag=new_revision,
                revision=old_commit_sha,
                repo_type="model",
            )

        old_deleted = False
        # Delete the old revision if requested
        if delete_old:
            if revision_type == "branch":
                api.delete_branch(
                    repo_id=model_id, branch=old_revision, repo_type="model"
                )
            else:
                api.delete_tag(repo_id=model_id, tag=old_revision, repo_type="model")
            old_deleted = True

        return {
            "success": True,
            "new_revision": new_revision,
            "old_revision_deleted": old_deleted,
            "message": f"Successfully renamed {revision_type} '{old_revision}' to '{new_revision}'"
            + (f" and deleted old {revision_type}" if old_deleted else ""),
        }

    except Exception as e:
        return {"success": False, "error": f"Error renaming {revision_type}: {str(e)}"}


if __name__ == "__main__":
    main()
