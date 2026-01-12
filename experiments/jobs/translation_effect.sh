#!/bin/bash
# Job execution script for NLLB translation experiments
# Task IDs 0-2: Strategies translate, nllb_translate_then_respond, nllb_translate_both with language ar (Arabic)
# Task IDs 3-5: Strategies translate, nllb_translate_then_respond, nllb_translate_both with language de (German)
# Task IDs 6-8: Strategies translate, nllb_translate_then_respond, nllb_translate_both with language id (Indonesian)

OMP_NUM_THREADS=16
MODEL="google/gemma-3-27b-it"

STRATEGIES=("translate" "nllb_translate_then_respond" "nllb_translate_both")
LANGUAGES=(ar de id)

STRATEGY=${STRATEGIES[SLURM_ARRAY_TASK_ID % ${#STRATEGIES[@]}]}
LANGUAGE=${LANGUAGES[SLURM_ARRAY_TASK_ID / ${#STRATEGIES[@]}]}

echo "Finetuning student model with teacher: ${MODEL}, language: ${LANGUAGE}, strategy: ${STRATEGY}"

INPUT_DATASET="ljvmiranda921/msde-T1-${LANGUAGE}"
BASE_MODEL=${1:-"allenai/Olmo-3-1025-7B"}
CHAT_TEMPLATE=${2:-"llama-3.1"}
TEACHER_MODEL_FULL="${MODEL}"

# Extract dataset name and teacher model for run_name
DATASET_NAME=$(basename ${INPUT_DATASET})
TEACHER_MODEL=$(basename ${TEACHER_MODEL_FULL})

RUN_NAME="${DATASET_NAME}_${TEACHER_MODEL}_${STRATEGY}"
INPUT_DATASET_FILTER="{\"strategy\": \"${STRATEGY}\"}"

python -m scripts.finetune_unsloth --help
python -m scripts.finetune_unsloth \
    --input_dataset ${INPUT_DATASET} \
    --run_name ${RUN_NAME} \
    --base_model ${BASE_MODEL} \
    --chat_template ${CHAT_TEMPLATE} \
    --num_epochs 2 \
    --learning_rate 5e-5 \
    --max_seq_length 16384 \
    --apply_subsampling \
    --max_train_samples 10000 \
    --save_mode merged_16bit \
    --use_lora \
    --load_in_4bit \
    --input_dataset_filter "${INPUT_DATASET_FILTER}"

