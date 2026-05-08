import argparse
import csv
import random
from collections import defaultdict

from datasets import load_dataset


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Sample N (prompt, response) pairs per model from a HuggingFace dataset for annotation.")
    parser.add_argument("--dataset", type=str, default="ljvmiranda921/msde-S1-tl", help="HuggingFace dataset path.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to sample from.")
    parser.add_argument("--n_per_model", type=int, default=50, help="Number of instances to sample per model.")
    parser.add_argument("--output", type=str, default="annotation_sample.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)

    ds = load_dataset(args.dataset, split=args.split)

    by_model: dict[str, list[int]] = defaultdict(list)
    for idx, model in enumerate(ds["model"]):
        by_model[model].append(idx)

    sampled_indices = []
    for model, indices in by_model.items():
        k = min(args.n_per_model, len(indices))
        if k < args.n_per_model:
            print(f"warning: {model} only has {len(indices)} rows, sampling {k}")
        sampled_indices.extend(random.sample(indices, k))

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt", "response", "model", "fluency", "cultural accuracy"],
        )
        writer.writeheader()
        for idx in sampled_indices:
            row = ds[idx]
            writer.writerow(
                {
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "model": row["model"],
                    "fluency": "",
                    "cultural accuracy": "",
                }
            )


if __name__ == "__main__":
    main()
