#!/bin/bash -l
#SBATCH --job-name=lj-test
#SBATCH --gres=gpu:1,gpu-ram:24G
#SBATCH --time=00:10:00
#SBATCH --output=gpu-%j.log

python -m scripts.synthesize_data --help

# .venv/bin/python -m scripts.synthesize_data --input_dataset ljvmiranda921/msde-seed-S1 \
#     --output_dataset ljvmiranda921/test-synthesize-id \
#     --target_lang id \
#     --strategy generate \
#     --has_prefilter \
#     --limit 1000 \
#     --backend vllm \
#     --model meta-llama/Llama-3.1-8B-Instruct