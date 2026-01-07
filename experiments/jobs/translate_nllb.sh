#!/bin/bash
# Job execution script for NLLB translation experiments
# Task IDs 0-2: Strategies translate, nllb_translate_then_respond, nllb_translate_both with language ar (Arabic)
# Task IDs 3-5: Strategies translate, nllb_translate_then_respond, nllb_translate_both with language de (German)
# Task IDs 6-8: Strategies translate, nllb_translate_then_respond, nllb_translate_both with language id (Indonesian)

export CUDA_VISIBLE_DEVICES=0,1

STRATEGIES=("translate" "nllb_translate_then_respond" "nllb_translate_both")
LANGUAGES=(ar de id)

STRATEGY=${STRATEGIES[SLURM_ARRAY_TASK_ID % ${#STRATEGIES[@]}]}
LANGUAGE=${LANGUAGES[SLURM_ARRAY_TASK_ID / ${#STRATEGIES[@]}]}

echo "Creating synthetic dataset for language: ${LANGUAGE} using strategy: ${STRATEGY}"

python -m scripts.translate_nllb --help
python -m scripts.translate_nllb \
    --input_dataset ljvmiranda921/tulu-3-sft-subsampled-english-only \
    --output_dataset ljvmiranda921/msde-T1-${LANGUAGE} \
    --target_lang ${LANGUAGE} \
    --strategy ${STRATEGY} \
    --limit 15000 \
    --shuffle 21 \
    --translate_model "facebook/nllb-200-3.3B" \
    --teacher_model "google/gemma-3-27b-it" \
    --backend_params '{"tensor_parallel_size":2,"gpu_memory_utilization":0.80, "max_model_length":4096, "require_all_responses": false}' \
    --generation_params '{"temperature": 0.8, "top_p": 0.9}'
