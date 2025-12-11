#!/bin/bash
# Job execution script for creating synthetic datasets

# Parse arguments
MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
BACKEND=${2:-"vllm"}
STRATEGY=${3:-"generate"}
LIMIT=${4:-10000}
LANGUAGES=(ar cs de es id ja)
LANGUAGE=${LANGUAGES[SLURM_ARRAY_TASK_ID]}

python -m scripts.synthesize_data --help
python -m scripts.synthesize_data --input_dataset ljvmiranda921/msde-seed-S1 \
    --output_dataset ljvmiranda921/msde-S1-${LANGUAGE} \
    --target_lang ${LANGUAGE} \
    --strategy ${STRATEGY} \
    --has_prefilter \
    --limit ${LIMIT} \
    --backend ${BACKEND} \
    --model ${MODEL} \
    --shuffle 921 \
    --append \
    --backend_params '{"tensor_parallel_size":2,"gpu_memory_utilization":0.7, "max_model_length":4096, "require_all_responses": false}' \
    --generation_params '{"temperature": 0.8, "top_p": 0.9}'
