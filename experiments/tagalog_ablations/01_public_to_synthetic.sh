#!/bin/bash

INPUT_DATASET_PUBLIC="ljvmiranda921/10k-Public-tl"
RUN_NAME="tgl_10k-Public-tl"
BASE_MODEL="google/gemma-3-4b-pt"
CHAT_TEMPLATE="gemma-3"

echo "Finetuning student model using this dataset: ${INPUT_DATASET_PUBLIC}"
python -m scripts.finetune_unsloth \
    --input_dataset ${INPUT_DATASET_PUBLIC} \
    --run_name ${RUN_NAME} \
    --base_model ${BASE_MODEL} \
    --chat_template ${CHAT_TEMPLATE} \
    --num_epochs 2 \
    --learning_rate 5e-5 \
    --max_seq_length 16384 \
    --max_train_samples 10000 \
    --save_mode "merged_16bit" \
    --use_lora \
    --load_in_4bit

sleep 60

RUN_NAME="tgl_10k-GPT-4om"
INPUT_DATASET_SYNTH="ljvmiranda921/msde-S1-tl"
TEACHER_MODEL_FULL="gpt-4o-mini-2024-07-18"
INPUT_DATASET_FILTER="{\"model\": \"${TEACHER_MODEL_FULL}\"}"

echo "Finetuning student model using this dataset: ${INPUT_DATASET_SYNTH} and teacher: ${TEACHER_MODEL_FULL}"


python -m scripts.finetune_unsloth \
    --input_dataset ${INPUT_DATASET_SYNTH} \
    --run_name ${RUN_NAME} \
    --base_model ${BASE_MODEL} \
    --chat_template ${CHAT_TEMPLATE} \
    --num_epochs 2 \
    --learning_rate 5e-5 \
    --max_seq_length 16384 \
    --apply_subsampling \
    --max_train_samples 10000 \
    --save_mode "merged_16bit" \
    --input_dataset_filter "${INPUT_DATASET_FILTER}" \
    --use_lora \
    --load_in_4bit
