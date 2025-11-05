#!/bin/bash
#SBATCH --job-name=sft
#! change to gpu:4 to use all 4 GPU cards on a GPU node.
#SBATCH --nodelist=ltl-gpu05
#SBATCH --gres=gpu:1
#SBATCH --output=gpu-%j.log

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
    --base_model ${BASE_MODEL} \
    --chat_template ${CHAT_TEMPLATE} \
    --num_epochs 2 \
    --learning_rate 2e-5 \
    --max_seq_length 2048 \
    --use_lora \
    --load_in_4bit \
    --save_mode merged_16bit \
    --input_dataset_filter '{"model": "meta-llama/Llama-3.1-8B-Instruct"}'