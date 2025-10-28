import uuid
import logging
import sys
import argparse
import hashlib
import random

import pandas as pd
from datasets import Dataset, load_dataset


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

LANG_MAPPING = {
    "Spanish": "es",
    "German": "de",
    "Indonesian": "id",
    "Czech": "cs",
    "Japanese": "ja",
}


def get_data_processors():
    """Registry of dataset processors. See implementation of each processor below."""
    return {
        "allenai/WildChat-4.8M": _process_wildchat,
        "openai/gsm8k": _process_gsm8k,
        "Magpie-Align/Magpie-Pro-300K-Filtered": _process_magpie_pro_300k,
        "nvidia/Helpsteer3": _process_nvidia_helpsteer3,
        #    "HuggingFaceH4/Multilingual-Thinking": _process_huggingfaceh4,
    }


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Create a seed dataset from a series of datasets.")
    parser.add_argument("--output_dataset", type=str, required=True, help="HuggingFace dataset path to save the seed dataset to.")
    parser.add_argument("--exclude", nargs="+", type=str, default=[], help="List of dataset names to exclude from the seed dataset.")
    parser.add_argument("--include", nargs="+", type=str, default=[], help="List of dataset names to exclusively include in the seed dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--num_instances", type=int, default=200_000, help="Limit the number of instances by sampling to this value. Useful for very large datasets that don't fit to memory.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    all_dfs = []
    for dataset_name, processor in get_data_processors().items():
        if args.include:
            if dataset_name not in args.include:
                continue
        elif dataset_name in args.exclude:
            logging.info(f"Skipping excluded dataset: {dataset_name}")
            continue

        logging.info(f"Processing dataset: {dataset_name}")
        df = processor(num_instances=args.num_instances, seed=args.seed)
        all_dfs.append(df)

    seed_dataset_df = pd.concat(all_dfs, ignore_index=True)
    breakpoint()


def _process_wildchat(num_instances: int, seed: int) -> pd.DataFrame:
    """Process the allenai/WildChat-4.8M dataset that contains multilingual prompt-response pairs."""
    wildchat_4_8m = load_dataset("allenai/WildChat-4.8M", split="train", streaming=True)
    sampled = wildchat_4_8m.shuffle(seed=seed).take(num_instances)

    sampled_df = pd.DataFrame(list(sampled))
    filtered_df = sampled_df[(sampled_df["language"].isin(LANG_MAPPING.keys()))]

    # Transform to desired format
    wildchat_df = pd.DataFrame(
        {
            "id": [uuid.uuid4().hex for _ in range(len(filtered_df))],
            "source": "allenai/WildChat-4.8M",
            "conversation": filtered_df["conversation"].values,
            "language": filtered_df["language"].map(LANG_MAPPING).values,
            "strategy": [["generate", "respond"] for _ in range(len(filtered_df))],
            "source_id": filtered_df["conversation_hash"].values,
        }
    )

    wildchat_df["prompt"] = wildchat_df.conversation.apply(lambda x: x[0]["content"])
    wildchat_df["response"] = wildchat_df.conversation.apply(lambda x: x[1]["content"])
    wildchat_df = wildchat_df.drop(columns=["conversation"])  # No longer needed
    return wildchat_df.reset_index(drop=True)


def _process_gsm8k(num_instances: int, seed: int) -> pd.DataFrame:
    """Process the openai/gsm8k dataset for math word problems."""
    gsm8k_df = load_dataset("openai/gsm8k", "main", split="train").to_pandas()
    gsm8k_df["source_id"] = gsm8k_df["question"].apply(
        lambda x: hashlib.md5(x.encode()).hexdigest()
    )
    gsm8k_df["id"] = [uuid.uuid4().hex for _ in range(len(gsm8k_df))]
    gsm8k_df = gsm8k_df.rename(columns={"question": "prompt", "answer": "response"})
    gsm8k_df["source"] = "openai/gsm8k"
    gsm8k_df["language"] = "en"
    gsm8k_df["strategy"] = [["translate"] for _ in range(len(gsm8k_df))]
    return gsm8k_df.reset_index(drop=True)


def _process_magpie_pro_300k(num_instances: int, seed: int) -> pd.DataFrame:
    """Process the Magpie-Align/Magpie-Pro-300K-Filtered dataset for general chat text."""
    magpie_pro_300k = load_dataset(
        "Magpie-Align/Magpie-Pro-300K-Filtered", split="train", streaming=True
    )
    sampled = magpie_pro_300k.shuffle(seed=seed).take(num_instances)
    sampled_df = pd.DataFrame(list(sampled))

    # Transform to desired format
    magpie_pro_300k_df = pd.DataFrame(
        {
            "id": [uuid.uuid4().hex for _ in range(len(sampled_df))],
            "source": "Magpie-Align/Magpie-Pro-300K-Filtered",
            "conversations": sampled_df["conversations"].values,
            "language": "en",
            "strategy": [["translate"] for _ in range(len(sampled_df))],
            "source_id": sampled_df["uuid"].values,
        }
    )

    # fmt: off
    magpie_pro_300k_df["prompt"] = magpie_pro_300k_df.conversations.apply(lambda x: x[0]["value"])
    magpie_pro_300k_df["response"] = magpie_pro_300k_df.conversations.apply(lambda x: x[1]["value"])
    magpie_pro_300k_df = magpie_pro_300k_df.drop(columns=["conversations"])  # No longer needed
    # fmt: on
    return magpie_pro_300k_df.reset_index(drop=True)


def _process_nvidia_helpsteer3(num_instances: int, seed: int) -> pd.DataFrame:
    """Process the nvidia/helpsteer3 dataset for `generation` strategy. We get the prompt and preferred response from the dataset."""
    helpsteer3_ds = load_dataset("nvidia/helpsteer3", "preference", split="train")
    filtered_df = helpsteer3_ds.filter(lambda x: x["domain"] == "multilingual").to_pandas()  # fmt: skip
    filtered_df = filtered_df[filtered_df.language.isin([lang.lower() for lang in LANG_MAPPING.keys()])].reset_index(drop=True)  # fmt: skip
    filtered_df["prompt"] = filtered_df["context"].apply(lambda x: x[0]["content"])

    def _get_preferred_response(row):
        if row["overall_preference"] < 0:
            pref = "response1"
        elif row["overall_preference"] == 0:
            pref = random.choice(["response1", "response2"])
        elif row["overall_preference"] > 0:
            pref = "response2"
        return row[pref]

    filtered_df["response"] = filtered_df.apply(
        lambda x: _get_preferred_response(x), axis=1
    )

    helpsteer3_df = pd.DataFrame(
        {
            "id": [uuid.uuid4().hex for _ in range(len(filtered_df))],
            "source": "nvidia/Helpsteer3",
            "prompt": filtered_df["prompt"].values,
            "response": filtered_df["response"].values,
            "language": filtered_df["language"]
            .map(lambda x: LANG_MAPPING[x.capitalize()])
            .values,
            "strategy": [["generate"] for _ in range(len(filtered_df))],
        }
    )

    helpsteer3_df["source_id"] = helpsteer3_df["prompt"].apply(
        lambda x: hashlib.md5(x.encode()).hexdigest()
    )
    return helpsteer3_df.reset_index(drop=True)


def _process_huggingfaceh4(num_instances: int, seed: int) -> pd.DataFrame:
    """Process the HuggingFaceH4/Multilingual-Thinking dataset for multilingual reasoning tasks.

    We just need the prompts in the 'user' column. We can filter based on the `reasoning_language` column.
    The reason we're doing this is because from a cursory check, the prompts are culturally-adapted, so might be useful.
    """
    mt_df = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train").to_pandas()  # fmt: skip
    filtered_df = mt_df[mt_df["reasoning_language"].isin(LANG_MAPPING.keys())]

    huggingface_h4_df = pd.DataFrame(
        {
            "id": [uuid.uuid4().hex for _ in range(len(filtered_df))],
            "source": "HuggingFaceH4/Multilingual-Thinking",
            "prompt": filtered_df["user"].values,
            "response": filtered_df["final"].values,
            "language": "en",
            "strategy": [["translate"] for _ in range(len(filtered_df))],
            "source_id": filtered_df["uuid"].values,
        }
    )
    return huggingface_h4_df.reset_index(drop=True)


if __name__ == "__main__":
    main()
