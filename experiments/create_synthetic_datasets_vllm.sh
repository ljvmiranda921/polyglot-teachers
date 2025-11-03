#!/bin/bash -l
#SBATCH --job-name=synthesize-data
#SBATCH --gres=gpu:1,gpu-ram:24G
#SBATCH --time=00:10:00
#SBATCH --output=gpu-%j.log

# Parse arguments
MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
LANGUAGE=${2:-"id"}

source .venv/bin/activate
.venv/bin/python -m scripts.synthesize_data --help
.venv/bin/python -m scripts.synthesize_data --input_dataset ljvmiranda921/msde-seed-S1 \
    --output_dataset ljvmiranda921/test-synthesize-${LANGUAGE} \
    --target_lang ${LANGUAGE} \
    --strategy generate \
    --has_prefilter \
    --limit 10000 \
    --backend vllm \
    --model ${MODEL}