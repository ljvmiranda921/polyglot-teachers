import argparse
import logging
import json
import os
import sys
from datetime import datetime

# Unsloth must be imported before torch and trl so that it can patch them properly.
from unsloth import FastLanguageModel  # isort: skip
from unsloth.chat_templates import CHAT_TEMPLATES, get_chat_template  # isort: skip
import torch
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    description = "Finetune a model on a dataset using Unsloth."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_dataset", type=str, required=True, help="HuggingFace dataset to use for finetuning. Must contain a 'messages' field in the OpenAI format.")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B", help="Base model to use for finetuning.")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run. This will be used to identify the model in TrackIO and also as a revision to the HuggingFace model in --output_model_name.")
    parser.add_argument("--chat_template", type=str, choices=list(CHAT_TEMPLATES.keys()), default="llama-3.1", help="Chat template to use for formatting the messages.")
    parser.add_argument("--output_model_name", type=str, default="ljvmiranda921/msde-unsloth-sft", help="Name of the output model (HuggingFace ID) to save after finetuning.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs to finetune for.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for finetuning.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the model.")
    parser.add_argument("--use_lora", action="store_true", help="If set, will use LoRA for finetuning.")
    parser.add_argument("--load_in_4bit", action="store_true", help="If set, will load the model in 4-bit precision to save memory.")
    parser.add_argument("--save_mode", choices=["merged_16bit", "merged_4bit", "lora"], default="merged_16bit", help="Precision for saving the finetuned model.")
    parser.add_argument("--input_dataset_filter", type=str, default=None, help="JSON string representing a filter to apply to the input dataset before finetuning. The keys should be the field names and the values should be the values to filter by. This is an AND operation.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("Please set the HF_TOKEN env variable!")

    run_name = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-msde-{args.base_model.replace('/', '___')}"
    if args.run_name:
        # Append custom run name suffix
        run_name += f"-{args.run_name}"
    logging.info(f"Starting finetuning run: {run_name}")

    model, tokenizer = get_model_and_tokenizer(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        use_lora=args.use_lora,
        token=hf_token,
    )

    dataset = prepare_training_data(
        dataset_name=args.input_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        input_dataset_filter=args.input_dataset_filter,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,  # Can make training 5x faster for short sequences.
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="trackio",
            # TrackIO reporting parameters
            project="sft-runs",
            trackio_space_id="msde-logging",
            run_name=run_name,
            hub_private_repo=True,
        ),
    )

    # Train the model and report stats and usage
    start_gpu_memory, max_memory = show_gpu_info()
    trainer_stats = trainer.train()
    show_training_stats(start_gpu_memory, max_memory, trainer_stats)

    # Save the finetuned model
    save_finetuned_model(
        run_name,
        model=model,
        tokenizer=tokenizer,
        output_hf_name=args.output_model_name,
        save_precision=args.save_mode,
        token=hf_token,
    )


def get_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 2048,
    dtype=None,
    load_in_4bit: bool = False,
    use_lora: bool = False,
    lora_r: int = 16,
    token: str = None,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=token,
    )

    if use_lora:
        logging.info("Applying LoRA adapters for finetuning.")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    return model, tokenizer


def prepare_training_data(
    dataset_name: str,
    tokenizer,
    chat_template: str,
    input_dataset_filter: str,
    messages_key: str = "messages",
) -> Dataset:
    """Apply chat template to the post-training dataset. Expects a 'messages' field in the dataset."""

    def _formatting_prompts_func(examples):
        messages = examples[messages_key]
        texts = [
            tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False
            )
            for msg in messages
        ]
        return {"text": texts}

    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    dataset = load_dataset(dataset_name, split="train")
    if input_dataset_filter:
        filter_dict = json.loads(input_dataset_filter)
        logging.info(f"Applying filter to dataset: {filter_dict}")
        dataset = dataset.filter(
            lambda example: all(example[k] == v for k, v in filter_dict.items())
        )

    dataset = dataset.map(_formatting_prompts_func, batched=True)
    return dataset


def show_gpu_info() -> tuple[float, float]:
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logging.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logging.info(f"{start_gpu_memory} GB of memory reserved.")
    return start_gpu_memory, max_memory


def show_training_stats(start_gpu_memory: float, max_memory: float, trainer_stats):
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    # fmt: off
    logging.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logging.info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    logging.info(f"Peak reserved memory = {used_memory} GB.")
    logging.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logging.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logging.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    # fmt: on


def save_finetuned_model(
    run_name: str,
    *,
    model,
    tokenizer,
    output_hf_name: str,
    save_precision: str,
    token: str,
):
    if save_precision in ["merged_16bit", "merged_4bit"]:
        model.push_to_hub_merged(
            output_hf_name,
            tokenizer,
            revision=run_name,
            save_method=save_precision,
            token=token,
        )
    elif save_precision == "lora":
        model.push_to_hub(output_hf_name, revision=run_name, token=token)
        tokenizer.push_to_hub(output_hf_name, revision=run_name, token=token)
    else:
        raise ValueError(f"Unknown save_precision: {save_precision}")


if __name__ == "__main__":
    main()
