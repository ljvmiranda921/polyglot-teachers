#!/bin/bash

RUN_NAME="tgl_25k-Gemma3-Big"
BASE_MODEL="google/gemma-3-12b-pt"

#RUN_NAME="tgl_25k-Gemma3-Ultra"
#BASE_MODEL="google/gemma-3-27b-pt"

CHAT_TEMPLATE="gemma-3"
TEACHER_MODEL_FULL="google/gemma-3-27b-it"
INPUT_DATASET_FILTER="{\"model\": \"${TEACHER_MODEL_FULL}\"}"

python -m scripts.finetune_unsloth \
    --input_dataset ${INPUT_DATASET_SYNTH} \
    --run_name ${RUN_NAME} \
    --base_model ${BASE_MODEL} \
    --chat_template ${CHAT_TEMPLATE} \
    --num_epochs 2 \
    --learning_rate 5e-5 \
    --max_seq_length 16384 \
    --apply_subsampling \
    --max_train_samples 25000 \
    --save_mode "merged_16bit" \
    --input_dataset_filter "${INPUT_DATASET_FILTER}" \
    --use_lora \
    --load_in_4bit
