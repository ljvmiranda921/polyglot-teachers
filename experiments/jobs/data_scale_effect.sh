#!/bin/bash
# Job execution script for finetuning student model with unsloth
# Task IDs 0-4: Data scales 1000, 5000, 10000, 25000, 50000 with language ar (Arabic)
# Task IDs 5-9: Data scales 1000, 5000, 10000, 25000, 50000 with language de (German)
# Task IDs 10-14: Data scales 1000, 5000, 10000, 25000, 50000 with language id (Indonesian)

OMP_NUM_THREADS=16
MODEL="google/gemma-3-27b-it"


DATA_SCALES=(1000 5000 10000 25000 50000)
LANGUAGES=(ar de id)

DATA_SCALE=${DATA_SCALES[SLURM_ARRAY_TASK_ID % ${#DATA_SCALES[@]}]}
LANGUAGE=${LANGUAGES[SLURM_ARRAY_TASK_ID / ${#DATA_SCALES[@]}]}

echo "Finetuning student model with teacher: ${MODEL}, language: ${LANGUAGE}, data scale: ${DATA_SCALE}"

INPUT_DATASET="ljvmiranda921/msde-S1-${LANGUAGE}"
BASE_MODEL=${1:-"allenai/Olmo-3-1025-7B"}
CHAT_TEMPLATE=${2:-"llama-3.1"}
TEACHER_MODEL_FULL="${MODEL}"

# Extract dataset name and teacher model for run_name
DATASET_NAME=$(basename ${INPUT_DATASET})
TEACHER_MODEL=$(basename ${TEACHER_MODEL_FULL})
DATA_SCALE_FORMATTED=$((DATA_SCALE / 1000))k
RUN_NAME="${DATASET_NAME}_${TEACHER_MODEL}_sz${DATA_SCALE_FORMATTED}"
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
    --apply_subsampling \
    --max_train_samples ${DATA_SCALE} \
    --save_mode merged_16bit \
    --use_lora \
    --load_in_4bit \
    --input_dataset_filter "${INPUT_DATASET_FILTER}"

