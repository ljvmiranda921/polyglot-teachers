#!/bin/bash
# Recommended SBATCH --array: 14 tasks x 9 models (e.g., 14x9=126)

# Run those with all languages available first
TASKS=(
    "lighteval|global_mmlu_lite:de|0|0"
    "lighteval|global_mmlu_lite:es|0|0"
    "lighteval|global_mmlu_lite:ja|0|0"
    "lighteval|mrewardbench_mcf:de|0|0"
    "lighteval|mrewardbench_mcf:es|0|0"
    "lighteval|mrewardbench_mcf:ja|0|0"
    "lighteval|mgsm:de|5|0"
    "lighteval|mgsm:es|5|0"
    "lighteval|mgsm:ja|5|0"
    "lighteval|global_mmlu_lite:ar|0|0"
    "lighteval|mrewardbench_mcf:ar|0|0"
    "lighteval|global_mmlu_lite:id|0|0"
    "lighteval|mrewardbench_mcf:id|0|0"
    "lighteval|mrewardbench_mcf:cs|0|0"
)

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "CohereLabs/aya-expanse-32b"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    "google/gemma-3-4b-it"
    "ibm-granite/granite-4.0-1b"
    "ibm-granite/granite-4.0-micro"
    "mistralai/Mistral-Small-24B-Instruct-2501"
    "meta-llama/Llama-3.1-70B-Instruct"
)

TASK=${TASKS[SLURM_ARRAY_TASK_ID % ${#TASKS[@]}]}
MODEL=${MODELS[SLURM_ARRAY_TASK_ID / ${#TASKS[@]}]}

echo "Evaluating model: ${MODEL} on task: ${TASK}"

# Reference for gsm8k setup: https://github.com/huggingface/lighteval/issues/686
source .venv/bin/activate
lighteval vllm --help
lighteval vllm "model_name=${MODEL},tensor_parallel_size=2,gpu_memory_utilization=0.9,max_model_length=4096,dtype=bfloat16,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" "${TASK}" \
    --custom-tasks scripts/lighteval_tasks.py \
    --output-dir lighteval-results \
    --results-path-template '{output_dir}/{org}___{model}' \
    --use-chat-template \
    --no-public-run \
    --results-org "ljvmiranda921" \
    --push-to-hub \
    --save-details 
