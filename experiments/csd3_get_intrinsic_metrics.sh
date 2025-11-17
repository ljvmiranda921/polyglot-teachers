#!/bin/bash
#SBATCH --account=KORHONEN-SL3-GPU
#SBATCH --job-name=metrics
#SBATCH --partition ampere
#SBATCH --nodes 1
#! change to gpu:4 to use all 4 GPU cards on a GPU node.
#SBATCH --gres=gpu:2
#SBATCH --output=gpu-%j.log
#SBATCH --time=10:00:00
#! Array computation is 11 models and 6 languages = 66 combinations
#SBATCH --array=0-65%4  

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
    "meta-llama/Llama-3.1-70B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "CohereLabs/aya-expanse-32b"
    "cohere-command-a"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    "google/gemma-3-4b-it"
    "gpt-4o-mini-2024-07-18"
    "ibm-granite/granite-4.0-1b"
    "ibm-granite/granite-4.0-micro"
    "mistralai/Mistral-Small-24B-Instruct-2501"
)

LANGUAGES=(ar cs de es id ja)

MODEL=${MODELS[SLURM_ARRAY_TASK_ID % ${#MODELS[@]}]}
LANGUAGE=${LANGUAGES[SLURM_ARRAY_TASK_ID / ${#MODELS[@]}]}

echo "Computing intrinsic metrics for model: ${MODEL} and language: ${LANGUAGE}"

source .venv/bin/activate
python -m scripts.get_intrinsic_metrics --help

# Compute across models (get 3.5k samples for each strategy)
python -m scripts.get_intrinsic_metrics --input_dataset ljvmiranda921/msde-S1-${LANGUAGE} \
    --metrics all \
    --output_path /home/ljvm2/rds/hpc-work/multilingual-teacher-eval/metrics/msde-S1-${LANGUAGE}_${MODEL//\//__}_intrinsic_metrics.json \
    --metric_params '{"reward_model::":{"language":"'"${LANGUAGE}"'","tensor_parallel_size":2}}' \
    --input_dataset_filter '{"model":"'"${MODEL}"'"}' \
    --apply_subsampling 
