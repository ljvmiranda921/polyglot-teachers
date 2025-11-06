#!/bin/bash
#SBATCH --account=KORHONEN-SL3-GPU
#SBATCH --job-name=synthesize-data
#SBATCH --partition ampere
#SBATCH --nodes 1
#! change to gpu:4 to use all 4 GPU cards on a GPU node.
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=gpu-%j.log

# Module setup: cluster environment and recent python.
module purge
module load rhel8/default-amp

# Parse arguments
MODEL=${1:-"meta-llama/Llama-3.1-70B-Instruct"}
BACKEND=${2:-"vllm"}
STRATEGY=${3:-"generate"}
LIMIT=${4:-10000}
LANGUAGES=(ar cs de es id ja)
LANGUAGE=${LANGUAGES[SLURM_ARRAY_TASK_ID]}

source .venv/bin/activate
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
    --no_cache \
    --append \
    --backend_params '{"tensor_parallel_size":4,"gpu_memory_utilization":0.7, "max_model_length":4096, "require_all_responses": false}'
