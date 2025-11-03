#!/bin/bash
#SBATCH --job-name=synthesize-data
#! change to gpu:4 to use all 4 GPU cards on a GPU node.
#SBATCH --nodelist=ltl-gpu05
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=gpu-%j.log

# Parse arguments
MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
LANGUAGE=${2:-"id"}

source .venv/bin/activate
python -m scripts.synthesize_data --help
python -m scripts.synthesize_data --input_dataset ljvmiranda921/msde-seed-S1 \
    --output_dataset ljvmiranda921/test-synthesize-${LANGUAGE} \
    --target_lang ${LANGUAGE} \
    --strategy generate \
    --has_prefilter \
    --limit 500 \
    --backend vllm \
    --model ${MODEL} \
    --backend_params '{"tensor_parallel_size":2,"gpu_memory_utilization":0.7, "max_model_length":4096}'