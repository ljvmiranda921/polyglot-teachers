#!/bin/bash
#SBATCH --account=KORHONEN-SL3-GPU
#SBATCH --job-name=metrics
#SBATCH --partition ampere
#SBATCH --nodes 1
#! change to gpu:4 to use all 4 GPU cards on a GPU node.
#SBATCH --gres=gpu:2
#SBATCH --output=gpu-%j.log
#SBATCH --time=10:00:00

# Module setup: cluster environment and recent python.
. /etc/profile.d/modules.sh 
module purge
module load rhel8/default-amp
module load python/3.11.9/gcc/abrhyqg7  # This should have dev headers

export HF_HOME=/home/ljvm2/rds/hpc-work/hpc_cache/huggingface
export HF_HUB_CACHE=/home/ljvm2/rds/hpc-work/hpc_cache/hf_hub
export VLLM_CACHE_ROOT=/home/ljvm2/rds/hpc-work/hpc_cache/vllm
export OMP_NUM_THREADS=16

MODELS=(
    "gpt-4o-mini-2024-08-06"
    "meta-llama/Llama-3.1-70B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"

)

LANGUAGES=(ar cs de es id ja)


source .venv/bin/activate
python -m scripts.get_intrinsic_metrics --help

# Compute across models (get 3.5k samples for each strategy)
python -m scripts.get_intrinsic_metrics --input_dataset ljvmiranda921/msde-S1-ar \
    --metrics all \
    --output_path /home/ljvm2/rds/hpc-work/multilingual-teacher-eval/metrics/msde-S1-ar_intrinsic_metrics.json \
    --metric_params '{"reward_model::{"language":"ar", "tensor_parallel_size": 2}}' \
    --input_dataset_filter '{"model": ${MODEL}}' \
    --apply_subsampling 
