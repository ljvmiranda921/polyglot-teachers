#!/bin/bash
#SBATCH --job-name=sft
#! change to gpu:4 to use all 4 GPU cards on a GPU node.
#SBATCH --nodelist=ltl-gpu05
#SBATCH --gres=gpu:1
#SBATCH --output=gpu-%j.log

usage: finetune_unsloth.py [-h] --input_dataset INPUT_DATASET [--base_model BASE_MODEL] --run_name RUN_NAME
                           [--chat_template {unsloth,zephyr,chatml,mistral,llama,vicuna,vicuna_old,vicuna old,alpaca,gemma,gemma_chatml,gemma2,gemma2_chatml,llama-3,llama3,phi-3,phi-35,phi-3.5,llama-3.1,llama-31,llama-3.2,llama-3.3,llama-32,llama-33,qwen-2.5,qwen-25,qwen25,qwen2.5,phi-4,gemma-3,gemma3,qwen-3,qwen3,gemma-3n,gemma3n,gpt-oss,gptoss,qwen3-instruct,qwen3-thinking,lfm-2,starling,yi-chat}]
                           [--output_model_name OUTPUT_MODEL_NAME] [--num_epochs NUM_EPOCHS] [--learning_rate LEARNING_RATE]
                           [--max_seq_length MAX_SEQ_LENGTH] [--use_lora] [--load_in_4bit] [--save_mode {merged_16bit,merged_4bit,lora}]


# Parse arguments                    
INPUT_DATASET=${1:-"ljvmiranda921/msde-S1-es"}
BASE_MODEL=${2:-"unsloth/gemma-3-1b-pt"}
CHAT_TEMPLATE=${3:-"gemma-3n"}
RUN_NAME=${3:-"test"}

source .venv/bin/activate
python -m scripts.finetune_unsloth --help
python -m scripts.finetune_unsloth \
    --input_dataset ${INPUT_DATASET} \
    --run_name ${RUN_NAME} \
    --chat_template ${CHAT_TEMPLATE} \
    --num_epochs 2 \
    --learning_rate 2e-5 \
    --max_seq_length 2048 \
    --use_lora \
    --load_in_4bit \
    --save_mode merged_16bit \
    --input_dataset_filter '{"model": "meta-llama/Llama-3.1-8B-Instruct"}'