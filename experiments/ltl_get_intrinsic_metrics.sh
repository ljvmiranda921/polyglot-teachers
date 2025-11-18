#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --nodelist=ltl-gpu05
#! change to gpu:4 to use all 4 GPU cards on a GPU node.
#SBATCH --gres=gpu:2
#SBATCH --output=gpu-%j.log
#SBATCH --time=10:00:00
#! Array computation is 11 models and 6 languages = 66 combinations
#SBATCH --array=0-65%8

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
# Run each metric separately to allow vLLM to release memory between runs
METRIC_PARAMS_SM='distinct_ri::{"embedding_model":"google/embeddinggemma-300m"}|reward_model::{"language": "'"$LANGUAGE"'", "tensor_parallel_size": 2, "model": "Unbabel/M-Prometheus-3B"}|perplexity::{"base_model":"google/gemma-3-270m","batch_size":64}'
METRIC_PARAMS_LG='distinct_ri::{"embedding_model":"nvidia/llama-embed-nemotron-8b","tensor_parallel_size":2}|reward_model::{"language": "'"$LANGUAGE"'", "tensor_parallel_size": 2, "model": "Unbabel/M-Prometheus-14B"}|perplexity::{"base_model":"google/gemma-3-270m","batch_size":32}'

METRIC_PARAMS="$METRIC_PARAMS_LG"

INPUT_FILTER='{"model": "'"$MODEL"'"}'
OUTPUT_PATH="/home/ljvm2/rds/hpc-work/dev/multilingual-teacher-eval/metrics/msde-S1-${LANGUAGE}_${MODEL//\//__}_intrinsic_metrics.json"

# Run distinct_ri metric
python -m scripts.get_intrinsic_metrics --input_dataset ljvmiranda921/msde-S1-${LANGUAGE} \
    --metrics distinct_ri \
    --output_path "$OUTPUT_PATH" \
    --metric_params "$METRIC_PARAMS" \
    --input_dataset_filter "$INPUT_FILTER" \
    --apply_subsampling

# Run reward_model metric
python -m scripts.get_intrinsic_metrics --input_dataset ljvmiranda921/msde-S1-${LANGUAGE} \
    --metrics reward_model \
    --output_path "$OUTPUT_PATH" \
    --metric_params "$METRIC_PARAMS" \
    --input_dataset_filter "$INPUT_FILTER" \
    --apply_subsampling

# Run perplexity metric
python -m scripts.get_intrinsic_metrics --input_dataset ljvmiranda921/msde-S1-${LANGUAGE} \
    --metrics perplexity \
    --output_path "$OUTPUT_PATH" \
    --metric_params "$METRIC_PARAMS" \
    --input_dataset_filter "$INPUT_FILTER" \
    --apply_subsampling 
