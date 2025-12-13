#!/bin/bash
# Job execution script for finetuning student model with unsloth

# Parse arguments
INPUT_DATASET=${1:-"ljvmiranda921/msde-S1-es"}
BASE_MODEL=${2:-"unsloth/gemma-3-1b-pt"}
CHAT_TEMPLATE=${3:-"gemma-3"}
TEACHER_MODEL_FULL=${4:-"meta-llama/Llama-3.1-8B-Instruct"}

# Extract dataset name and teacher model for run_name
DATASET_NAME=$(basename ${INPUT_DATASET})
TEACHER_MODEL=$(basename ${TEACHER_MODEL_FULL})
RUN_NAME="${DATASET_NAME}_${TEACHER_MODEL}"
INPUT_DATASET_FILTER="{\"model\": \"${TEACHER_MODEL_FULL}\"}"

python -m scripts.finetune_unsloth --help
python -m scripts.finetune_unsloth \
    --input_dataset ${INPUT_DATASET} \
    --run_name ${RUN_NAME} \
    --base_model ${BASE_MODEL} \
    --chat_template ${CHAT_TEMPLATE} \
    --num_epochs 2 \
    --learning_rate 5e-6 \
    --max_seq_length 2048 \
    --use_lora \
    --load_in_4bit \
    --save_mode merged_16bit \
    --input_dataset_filter "${INPUT_DATASET_FILTER}"
