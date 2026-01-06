#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

STRATEGIES=("translate", "nllb_translate_then_respond", "nllb_translate_both")
LANGUAGES=(ar de id)

STRATEGY=${STRATEGIES[SLURM_ARRAY_TASK_ID % ${#STRATEGIES[@]}]}
LANGUAGE=${LANGUAGES[SLURM_ARRAY_TASK_ID] / ${#STRATEGIES[@]}}

echo "Creating synthetic dataset for language: ${LANGUAGE} using strategy: ${STRATEGY}"

python -m scripts.translate_nllb --help
python -m scripts.translate_nllb \
    --input_dataset ljvmiranda921/tulu-3-sft-subsampled-english-only \
    --output_dataset ljvmiranda921/msde-T1-${LANGUAGE} \
    --target_lang ${LANGUAGE} \
    --strategy $STRATEGY \
    --translate_model "facebook/nllb-200-3.3B" \
    --teacher_model "google/gemma-3-27b-it" \
    --batch_size 16 \
    --backend_params '{"tensor_parallel_size":1,"gpu_memory_utilization":0.7, "max_model_length":4096, "require_all_responses": false}' \ 
    --generation_params '{"temperature": 0.8, "top_p": 0.9}'
