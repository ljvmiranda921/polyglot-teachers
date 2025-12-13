#!/bin/bash
# Job execution script for finetuning student model with unsloth

OMP_NUM_THREADS=16

MODELS=(
    "meta-llama/Llama-3.1-70B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "CohereLabs/aya-expanse-32b"
    "cohere-command-a"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    "google/gemma-3-4b-it"
    "gpt-4o-mini-2024-07-18"
    "ibm-granite/granite-4.0-1b"
    "ibm-granite/granite-4.0-micro"
)

LANGUAGES=(ar cs de es id ja)

MODEL=${MODELS[SLURM_ARRAY_TASK_ID % ${#MODELS[@]}]}
LANGUAGE=${LANGUAGES[SLURM_ARRAY_TASK_ID / ${#MODELS[@]}]}

echo "Finetuning student model with teacher: ${MODEL} and language: ${LANGUAGE}"

# Set parameters based on arrays
INPUT_DATASET="ljvmiranda921/msde-S1-${LANGUAGE}"
BASE_MODEL=${1:-"allenai/Olmo-3-1025-7B"}
CHAT_TEMPLATE=${2:-"llama-3.1"}
TEACHER_MODEL_FULL="${MODEL}"

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
    --learning_rate 5e-5 \
    --max_seq_length 16384 \
    --use_lora \
    --load_in_4bit \
    --apply_subsampling \
    --max_train_samples 10500 \
    --save_mode merged_16bit \
    --input_dataset_filter "${INPUT_DATASET_FILTER}"
