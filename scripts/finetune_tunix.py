import argparse
import sys
import os
import json
import logging
from pathlib import Path

import jax
import numpy as np
from flax import nnx  # neural network lib for jax
import jax.numpy as jnp  # numpy commands in TPU
from orbax import checkpoint as ocp  # checkpointing
import qwix  # quantization
import optax  # gradient and optimization library
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models import gemma3, qwen2, qwen3, llama3
from huggingface_hub import snapshot_download
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import utils
from tunix.sft.utils import show_hbm_usage

# from tunix.models.gemma3 import model as gemma_lib
# from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
# from tunix.models.gemma3 import params as gemma_params

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    description = "Perform finetuning using Tunix on TPUs"
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dataset", type=str, required=True, help="HuggingFace dataset to use for finetuning. Must contain a 'messages' field in the OpenAI format.")
    parser.add_argument("--base_model", type=str, default="google/gemma-3-270m", help="Base model to use for finetuning.")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run. This will be used to identify the model in TrackIO and also as a revision to the HuggingFace model in --output_model_name. Will be added as a suffix to a timestamp.")
    parser.add_argument("--output_model_name", type=str, default="ljvmiranda921/msde-sft-dev", help="Name of the output model (HuggingFace ID) to save after finetuning.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for finetuning.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for finetuning.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to finetune for.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the model.")
    parser.add_argument("--use_lora", action="store_true", help="If set, will use LoRA for finetuning.")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    checkpoints_dir = {
        "full_ckpt": Path(args.checkpoints_dir) / "full_ckpts",
        "lora_ckpt": Path(args.checkpoints_dir) / "lora_ckpts",
        "profiling": Path(args.checkpoints_dir) / "profiling",
    }
    for _, ckpt_path in checkpoints_dir.items():
        ckpt_path.mkdir(parents=True, exist_ok=True)


def get_device_info() -> list:
    """Get TPU device information and set up mesh configuration."""
    devices = jax.devices()
    num_tpus = len(devices)

    logging.info(f"Number of TPU devices: {num_tpus}")
    for idx, device in enumerate(devices):
        logging.info(f"Device {idx}: {device}")

    if num_tpus == 8:
        mesh_counts = (1, 4)
    elif num_tpus == 1:
        mesh_counts = (1, 1)
    else:
        raise ValueError(f"Unsupported number of TPUs: {num_tpus}")

    mesh_config = [mesh_counts, ("fsdp", "tp")]
    return mesh_config


def get_model_and_tokenizer(model_name: str):
    local_model_path = snapshot_download(
        repo_id=model_name,
        ignore_patterns=["*.pth"],  # Ignore PyTorch .pth weight files
    )

    eos_tokens = []
    generation_config_path = Path(local_model_path, "generation_config.json")
    if generation_config_path.exists():
        with generation_config_path.open("r") as f:
            generation_configs = json.load(f)
        eos_tokens = generation_configs.get("eos_token_id", [])
        logging.info(f"Using EOS token IDs: {eos_tokens}")

    show_hbm_usage("Before loading model and tokenizer")


if __name__ == "__main__":
    main()
