#!/bin/bash

RUN_NAME="tgl_10k-AyaExp"
BASE_MODEL="google/gemma-3-4b-pt"
CHAT_TEMPLATE="gemma-3"
TEACHER_MODEL_FULL="CohereLabs/aya-expanse-32b"
INPUT_DATASET="ljvmiranda921/msde-S1-tl"
INPUT_DATASET_FILTER="{\"model\": \"${TEACHER_MODEL_FULL}\"}"

torchrun --nproc_per_node 2 -m scripts.finetune_unsloth \
    --input_dataset ${INPUT_DATASET} \
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
