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
from huggingface_hub import snapshot_download
from tunix.sft import metrics_logger, peft_trainer, utils
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.models.gemma3 import params as gemma_params

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
    parser.add_argument("--use_tokenizer", type=str, default="gs://gemma-data/tokenizers/tokenizer_gemma3.model", help="Path to tokenizer to use. If not set, will use the tokenizer associated with --base_model.")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run. This will be used to identify the model in TrackIO and also as a revision to the HuggingFace model in --output_model_name. Will be added as a suffix to a timestamp.")
    parser.add_argument("--output_model_name", type=str, default="ljvmiranda921/msde-sft-dev", help="Name of the output model (HuggingFace ID) to save after finetuning.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for finetuning.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for finetuning.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to finetune for.")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0, overrides num_epochs to set the maximum number of training steps to perform.")
    parser.add_argument("--eval_steps", type=int, default=20, help="Number of steps between evaluations.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the model.")
    parser.add_argument("--quantize", action="store_true", help="If set, will quantize the model to 4-bit using QWIX.")
    parser.add_argument("--use_lora", action="store_true", help="If set, will use LoRA for finetuning.")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    # Setup checkpoints and sharding mesh
    checkpoints_dir = {
        "full_ckpt": Path(args.checkpoints_dir) / "full_ckpts",
        "lora_ckpt": Path(args.checkpoints_dir) / "lora_ckpts",
        "profiling": Path(args.checkpoints_dir) / "profiling",
    }
    for _, ckpt_path in checkpoints_dir.items():
        ckpt_path.mkdir(parents=True, exist_ok=True)

    mesh_config = get_device_info()
    mesh: jax.sharding.Mesh = jax.make_mesh(*mesh_config, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_config[0]))  # fmt: skip

    # Initialize the baes model and tokenizer
    base_model, tokenizer, eos_tokens = get_model_and_tokenizer(
        model_name=args.base_model,
        mesh=mesh,
        tokenizer_path=args.use_tokenizer if args.use_tokenizer else args.base_model,
    )
    model = (
        get_lora_model(base_model, mesh=mesh, quantize=args.quantize)
        if args.use_lora
        else base_model
    )
    nnx.display(model)


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


def get_model_and_tokenizer(
    model_name: str,
    *,
    mesh: jax.sharding.Mesh,
    tokenizer_path: str,
):
    """Load model and tokenizer from HuggingFace Hub.

    The tokenizer_path can be a GCS path (e.g., gs://gemma-data/tokenizers/tokenizer_gemma3.model)
    or a HugingFace Hub repo ID (in case you're using a different model family).

    NOTE: Currently only supports Gemma-3 models.
    """
    logging.warning("Currently only Gemma-3 models are supported.")

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

    utils.show_hbm_usage("Before loading model and tokenizer")

    # TODO: Add support for other model families
    if "gemma-3-270m" in model_name:
        model_config = gemma_lib.ModelConfig.gemma3_270m()
    elif "gemma-3-1b" in model_name:
        model_config = gemma_lib.ModelConfig.gemma3_1b()
    elif "gemma-3-4b" in model_name:
        model_config = gemma_lib.ModelConfig.gemma3_4b()
    elif "gemma-3-12b" in model_name:
        model_config = gemma_lib.ModelConfig.gemma3_12b()
    elif "gemma-3-27b" in model_name:
        model_config = gemma_lib.ModelConfig.gemma3_27b()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    with mesh:
        base_model = params_safetensors_lib.create_model_from_safe_tensors(local_model_path, (model_config), mesh)  # fmt: skip

    # Load tokenizer
    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=tokenizer_path)
    if tokenizer.eos_id() not in eos_tokens:
        eos_tokens.append(tokenizer.eos_id())
        print(f"Using EOS token IDs: {eos_tokens}")

    return base_model, tokenizer, eos_tokens


def get_lora_model(
    base_model,
    *,
    mesh: jax.sharding.Mesh,
    quantize: bool,
    lora_r: int = 16,
    lora_alpha: float = 2.0,
):
    """Wrap the base model with LoRA adapters and optionally quantize it."""
    if quantize:
        lora_provider = qwix.LoraProvider(
            module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
            rank=lora_r,
            alpha=lora_alpha,
            weight_qtype="nf4",
            tile_size=128,
        )
    else:
        lora_provider = qwix.LoraProvider(
            module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
            rank=lora_r,
            alpha=lora_alpha,
        )

    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(base_model, lora_provider, **model_input)

    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded_state)

    return lora_model


if __name__ == "__main__":
    main()
