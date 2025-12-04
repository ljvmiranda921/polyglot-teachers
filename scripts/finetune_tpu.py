import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import datasets
import jax
import jax.numpy as jnp  # numpy commands in TPU
import numpy as np
import optax  # gradient and optimization library
import qwix  # quantization
from dotenv import load_dotenv
from flax import nnx  # neural network lib for jax
from grain import python as grain
from huggingface_hub import HfApi, snapshot_download
from orbax import checkpoint as ocp  # checkpointing
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params as gemma_params
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.sft import metrics_logger, peft_trainer, utils

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

CHAT_TEMPLATES = {
    "gemma-3": {
        "prefix": "<start_of_turn>user\n",
        "suffix": "\n<end_of_turn>\n<start_of_turn>model\n",
    }
}


def get_args():
    # fmt: off
    description = "Perform finetuning using Tunix on TPUs"
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dataset", type=str, required=True, help="HuggingFace dataset to use for finetuning. Must contain a 'messages' field in the OpenAI format.")
    parser.add_argument("--base_model", type=str, default="google/gemma-3-270m", help="Base model to use for finetuning.")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run. This will be used to identify the model in TrackIO and also as a revision to the HuggingFace model in --output_model_name. Will be added as a suffix to a timestamp.")
    parser.add_argument("--use_tokenizer", type=str, default="gs://gemma-data/tokenizers/tokenizer_gemma3.model", help="Path to tokenizer to use. If not set, will use the tokenizer associated with --base_model.")
    parser.add_argument("--output_model_name", type=str, default="ljvmiranda921/msde-sft-dev", help="Name of the output model (HuggingFace ID) to save after finetuning.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for finetuning.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for finetuning.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to finetune for.")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0, overrides num_epochs to set the maximum number of training steps to perform.")
    parser.add_argument("--eval_steps", type=int, default=20, help="Number of steps between evaluations.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the model.")
    parser.add_argument("--validation_split", default=None, help="Name of the validation split in the dataset. If not set, will use 5%% of the training set as validation.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank to use for finetuning.")
    parser.add_argument("--lora_alpha", type=float, default=2.0, help="LoRA alpha to use for finetuning.")
    parser.add_argument("--quantize", action="store_true", help="If set, will quantize the model to 4-bit using QWIX.")
    parser.add_argument("--use_lora", action="store_true", help="If set, will use LoRA for finetuning.")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--input_dataset_filter", type=str, default=None, help="JSON string representing a filter to apply to the input dataset before finetuning. The keys should be the field names and the values should be the values to filter by. This is an AND operation.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token is None:
        raise ValueError("HuggingFace token not found in HF_TOKEN environment variable.")  # fmt: skip

    # Set-up the run name
    run_name = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-msde-{args.base_model.replace('/', '___')}"
    if args.use_lora:
        run_name += "-lora"
    elif args.quantize:
        run_name += "-qlora-nf4"
    if args.run_name:
        # Append custom run name suffix
        run_name += f"-{args.run_name}"
    logging.info(f"Starting finetuning run: {run_name}")

    if not args.use_lora:
        logging.warning("Finetuning without LoRA is not recommended due to high memory usage.")  # fmt: skip

    # Setup checkpoints and sharding mesh
    checkpoints_dir = {
        "full_ckpt": Path(args.checkpoints_dir) / "full_ckpts",
        "lora_ckpt": Path(args.checkpoints_dir) / "lora_ckpts",
        "profiling": Path(args.checkpoints_dir) / "profiling",
        "log_dir": Path(args.checkpoints_dir) / "train-logs" / f"train-{datetime.now().strftime('%Y%m%dT%H%M%S')}",  # fmt: skip
    }
    for _, ckpt_path in checkpoints_dir.items():
        ckpt_path.mkdir(parents=True, exist_ok=True)

    mesh_config = get_device_info()
    mesh: jax.sharding.Mesh = jax.make_mesh(*mesh_config, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_config[0]))  # fmt: skip

    # Initialize the model and tokenizer
    base_model, tokenizer, eos_tokens, local_model_path = get_model_and_tokenizer(
        model_name=args.base_model,
        mesh=mesh,
        tokenizer_path=args.use_tokenizer if args.use_tokenizer else args.base_model,
        tokenizer_type="sentencepiece" if args.use_tokenizer else "huggingface",
        hf_token=hf_token,
    )
    model = (
        get_lora_model(
            base_model,
            mesh=mesh,
            quantize=args.quantize,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
        if args.use_lora
        else base_model
    )
    nnx.display(model)

    # Load the dataset
    dataset_filter = json.loads(args.input_dataset_filter) if args.input_dataset_filter else None  # fmt: skip
    train_ds, eval_ds = get_dataset(
        dataset_name=args.input_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_seq_length=args.max_seq_length,
        validation_split_name=args.validation_split,
        chat_template_name="gemma-3",
        input_dataset_filter=dataset_filter,
        seed=args.seed,
    )

    # Setup training options
    full_logging_options = metrics_logger.MetricsLoggerOptions(
        log_dir=checkpoints_dir["log_dir"], flush_every_n_steps=20
    )
    training_config = peft_trainer.TrainingConfig(
        eval_every_n_steps=args.eval_steps,
        max_steps=args.max_steps,
        metrics_logging_options=full_logging_options,
        checkpoint_root_directory=checkpoints_dir["lora_ckpt"] if args.use_lora else checkpoints_dir["full_ckpt"],  # fmt: skip
    )
    trainer = peft_trainer.PeftTrainer(
        model=model,
        optimizer=optax.adamw(learning_rate=args.learning_rate),
        training_config=training_config,
    ).with_gen_model_input_fn(gen_model_input_fn)

    with mesh:
        trainer.train(train_ds, eval_ds)

    save_finetuned_model(
        run_name=run_name,
        model=model,
        local_model_path=local_model_path,
        output_hf_name=args.output_model_name,
        token=hf_token,
        lora_r=args.lora_r if args.use_lora else None,
        lora_alpha=args.lora_alpha if args.use_lora else None,
    )


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
    tokenizer_type: str = "sentencepiece",
    hf_token: Optional[str] = None,
):
    """Load model and tokenizer from HuggingFace Hub.

    The tokenizer_path can be a GCS path (e.g., gs://gemma-data/tokenizers/tokenizer_gemma3.model)
    or a HugingFace Hub repo ID (in case you're using a different model family).
    For the former, use tokenizer_type="sentencepiece", for the latter, use tokenizer_type="huggingface".

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
    tokenizer = tokenizer_lib.Tokenizer(
        tokenizer_type=tokenizer_type,
        tokenizer_path=tokenizer_path,
        hf_access_token=hf_token,
    )
    if tokenizer.eos_id() not in eos_tokens:
        eos_tokens.append(tokenizer.eos_id())
        print(f"Using EOS token IDs: {eos_tokens}")

    return base_model, tokenizer, eos_tokens, local_model_path


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


def get_dataset(
    dataset_name: str,
    *,
    tokenizer: tokenizer_lib.Tokenizer,
    batch_size: int,
    num_epochs: int,
    max_seq_length: int,
    input_dataset_filter: Optional[dict] = None,
    validation_split_name: Optional[str] = None,
    chat_template_name: str = "gemma-3",
    seed: int = 42,
) -> tuple[grain.DataLoader, grain.DataLoader]:
    """Load dataset and build data loaders for training and evaluation.

    Some datasets won't have their own validation dataset so if it's not
    provided, we take 5% of the training set randomly (seeded).
    """
    if validation_split_name:
        train_ds, eval_ds = datasets.load_dataset(
            dataset_name, split=("train", validation_split_name)
        )
    else:
        logging.warning("No validation split name provided. Using 5%% of the training set as validation.")  # fmt: skip
        full_ds = datasets.load_dataset(dataset_name, split="train", trust_remote_code=True)  # fmt: skip
        full_ds = full_ds.shuffle(seed=seed)
        train_size = int(0.95 * len(full_ds))
        train_ds, eval_ds = full_ds.select(range(train_size)), full_ds.select(range(train_size, len(full_ds)))  # fmt: skip

    if input_dataset_filter:
        filter_dict = json.loads(input_dataset_filter)
        logging.info(f"Applying filter to dataset: {filter_dict}")
        dataset = dataset.filter(
            lambda example: all(example[k] == v for k, v in filter_dict.items())
        )

    if chat_template_name not in CHAT_TEMPLATES:
        raise ValueError(f"Unsupported chat template name: {chat_template_name}")

    input_template = CHAT_TEMPLATES.get(chat_template_name)
    train_loader = _build_data_loader(
        data_source=train_ds,
        batch_size=batch_size,
        num_epochs=num_epochs,
        max_seq_len=max_seq_length,
        tokenizer=tokenizer,
        input_template=input_template,
    )
    eval_loader = _build_data_loader(
        data_source=eval_ds,
        batch_size=batch_size,
        num_epochs=1,
        max_seq_len=max_seq_length,
        tokenizer=tokenizer,
        input_template=input_template,
    )
    return train_loader, eval_loader


def _build_data_loader(
    data_source: grain.RandomAccessDataSource,
    batch_size: int,
    num_epochs: int | None,
    max_seq_len: int,
    tokenizer: tokenizer_lib.Tokenizer,
    input_template: dict[str, str],
) -> grain.DataLoader:
    return grain.DataLoader(
        data_source=data_source,
        sampler=grain.IndexSampler(
            num_records=len(data_source),
            num_epochs=num_epochs,
            shard_options=grain.NoSharding(),
        ),
        operations=[
            _Tokenize(tokenizer, input_template),
            _BuildTrainInput(max_seq_len, tokenizer.pad_id()),
            _FilterOverlength(max_seq_len),
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ],
    )


class _Tokenize(grain.MapTransform):
    """Tokenize the input."""

    def __init__(
        self, tokenizer: tokenizer_lib.Tokenizer, input_template: dict[str, str]
    ):
        self._tokenizer: tokenizer_lib.Tokenizer = tokenizer
        self._input_template = input_template

    def map(self, element: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize the input."""
        src_tokens = self._tokenizer.tokenize(
            example=element["prompt"],
            prefix=self._input_template["prefix"],
            suffix=self._input_template["suffix"],
            add_eos=False,
        )
        dst_tokens = self._tokenizer.tokenize(element["response"], add_eos=True)
        return src_tokens, dst_tokens


class _BuildTrainInput(grain.MapTransform):
    """Build a TrainingInput from a tuple of source and destination tokens."""

    def __init__(self, max_seq_len: int, pad_value: int | bool):
        self._max_seq_len = max_seq_len
        self._pad_value = pad_value

    def map(self, tokens: tuple[np.ndarray, np.ndarray]) -> peft_trainer.TrainingInput:
        src_tokens, dst_tokens = tokens

        # The input sequence fed to the model is simply the concatenation of the
        # source and the destination.
        tokens = np.concat([src_tokens, dst_tokens], axis=0)

        # To prevent the model from updating based on the source (input)
        # tokens, add a target mask to each input.
        q_mask = np.zeros_like(src_tokens, dtype=np.bool)
        a_mask = np.ones_like(dst_tokens, dtype=np.bool)
        mask = np.concat([q_mask, a_mask], axis=0)

        # If the input tokens sequence is smaller than the target sequence size,
        # then pad it with pad tokens.
        tokens = self._pad_up_to_max_len(tokens, self._pad_value)

        # Don't want to perform the backward pass on the pad tokens.
        mask = self._pad_up_to_max_len(mask, 0)

        return peft_trainer.TrainingInput(input_tokens=tokens, input_mask=mask)

    def _pad_up_to_max_len(
        self, input_tensor: np.ndarray, pad_value: int
    ) -> np.ndarray:
        """Pad the given tensor up to sequence length of a batch."""
        seq_len = input_tensor.shape[0]
        to_pad = np.maximum(self._max_seq_len - seq_len, 0)
        return np.pad(
            input_tensor,
            [[0, to_pad]],
            mode="constant",
            constant_values=pad_value,
        )


class _FilterOverlength(grain.FilterTransform):
    """Filter out overlength examples."""

    def __init__(self, max_seq_len: int):
        self._max_seq_len = max_seq_len

    def filter(self, element: peft_trainer.TrainingInput) -> bool:
        return element.input_tokens.shape[0] <= self._max_seq_len


def gen_model_input_fn(x: peft_trainer.TrainingInput) -> dict[str, Any]:
    pad_mask = x.input_tokens != tokenizer.pad_id
    positions = utils.build_positions_from_mask(pad_mask)
    attention_mask = utils.make_causal_attn_mask(pad_mask)
    return {
        "input_tokens": x.input_tokens,
        "input_mask": x.input_mask,
        "positions": positions,
        "attention_mask": attention_mask,
    }


def save_finetuned_model(
    run_name: str,
    *,
    model,
    local_model_path: str,
    output_hf_name: str,
    token: str,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[float] = None,
):
    output_dir = Path("tunix-models") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    gemma_params.save_lora_merged_model_as_safetensors(
        local_model_path=str(local_model_path),
        output_dir=str(output_dir),
        lora_model=model,
        rank=lora_r,
        alpha=lora_alpha,
    )
    print("\n" + "=" * 60)
    print("Model saved successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    print("\nSaved files:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
        print(f"  {f:<30} {size:>10.2f} MB")

    commit_message = f"ckpt for {run_name}"
    try:
        api = HfApi()
        api.create_branch(
            repo_id=output_hf_name,
            branch=run_name,
            repo_type="model",
            exist_ok=True,
            token=token,
        )
        api.upload_folder(
            folder_path=output_dir,
            repo_id=output_hf_name,
            repo_type="model",
            path_in_repo=".",
            revision=run_name,
            commit_message=commit_message,
            token=token,
        )
    except Exception as e:
        logging.error(f"Upload failed, keeping local directory: {e}")
        raise
    else:
        # Delete local directory to save some space in the cluster
        shutil.rmtree(output_dir)
        logging.info(f"Successfully uploaded and deleted local directory: {output_dir}")


if __name__ == "__main__":
    main()
