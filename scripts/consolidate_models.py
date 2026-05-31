"""Consolidate the Polyglot Teachers models into a repo."""

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from string import Template

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

README = Template("""---
library_name: transformers
license: other
license_name: mixed
pipeline_tag: text-generation
language:
- ar
- es
- cs
- de
- id
- tl
base_model:
- allenai/Olmo-3-1025-7B
- google/gemma-3-4b-pt
datasets:
- ljvmiranda921/PolyglotTeachers-SFT-Synth
tags:
- multilingual
- synthetic
- sft
---

# Polyglot SFT (Multilingual)

These are per-language models supervised fine-tuned on the synthetic data
generated in the [Polyglot Teachers](https://huggingface.co/collections/ljvmiranda921/polyglot-teachers)
project (see [ljvmiranda921/PolyglotTeachers-SFT-Synth](https://huggingface.co/datasets/ljvmiranda921/PolyglotTeachers-SFT-Synth)).

Load a specific model by passing the branch as the `revision`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "${target}"
branch = "${example_branch}"  # pick any branch below
model = AutoModelForCausalLM.from_pretrained(repo, revision=branch)
tokenizer = AutoTokenizer.from_pretrained(repo, revision=branch)
```

## Branches

| Branch | Original model |
| --- | --- |
${branch_table}

## Licensing

This repo holds models under different licenses; each branch follows its base
model's license:

- `Polyglot-OLMo3-7B-SFT-*` (base [allenai/Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B)) — Apache-2.0
- `Polyglot-Gemma3-4B-SFT-*` (base [google/gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt)) — [Gemma license](https://ai.google.dev/gemma/terms)

## Acknowledgements

LJVM and AK acknowledge the support of the UKRI Frontier Grant EP/Y031350/1 ([EQUATE](https://gtr.ukri.org/projects?ref=EP%2FY031350%2F1)).
This work was performed using joint resources provided by the [Cambridge Service for Data Driven Discovery (CSD3)](https://hpc.cam.ac.uk/high-performance-computing) EP/T022159/1 and the [Isambard AI National AI Research Resource (AIRR)](https://www.bristol.ac.uk/research/centres/bristol-supercomputing/#isambard-ai) ST/AIRR/I-A-I/1023, and the Microsoft Research Grant.
LJVM would also like to thank Songbo Hu, Chen Cecilia Liu, Millicent Ochieng, and Felermino Ali for helpful and productive discussions on the project.

## Citation

```bibtex
@misc{miranda2026polyglotteachersevaluatinglanguage,
    title={Polyglot Teachers: Evaluating Language Models for Multilingual Synthetic Data Generation},
    author={Lester James V. Miranda and Ivan Vulić and Anna Korhonen},
    year={2026},
    eprint={2604.11290},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2604.11290},
}
```
""")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Copy every model in a collection into branches of one repo.")
    parser.add_argument("--collection", default=COLLECTION, help="Source collection slug.")
    parser.add_argument("--target", default=TARGET, help="Target repo that will hold one branch per model.")
    parser.add_argument("--workdir", default=None, help="Where to download snapshots. Defaults to a temp dir.")
    parser.add_argument("--private", action="store_true", help="Create the target repo as private.")
    parser.add_argument("--dry_run", action="store_true", help="List what would be done without creating or uploading anything.")
    # fmt: on
    return parser.parse_args()


def build_readme(target: str, mapping: list[tuple[str, str]]) -> str:
    """mapping is a list of (branch, original_model_id)."""
    rows = [
        f"| `{branch}` | [{original}](https://huggingface.co/{original}) |"
        for branch, original in mapping
    ]
    return README.substitute(
        target=target,
        example_branch=mapping[0][0],
        branch_table="\n".join(rows),
    )


def main():
    args = get_args()
    api = HfApi()

    collection = get_collection(args.collection)
    models = [item.item_id for item in collection.items if item.item_type == "model"]

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
    api.upload_file(
        path_or_fileobj=build_readme(args.target, mapping).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.target,
        commit_message="Add branch index to main",
    )
    logger.info(f"Ensured target repo and README: {args.target}")

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
